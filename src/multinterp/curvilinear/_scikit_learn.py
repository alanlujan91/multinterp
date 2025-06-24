from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR

from multinterp.grids import _CurvilinearGrid
from multinterp.rectilinear._multi import MultivariateInterp
from multinterp.utilities import update_mc_kwargs


class PipelineCurvilinearInterp(_CurvilinearGrid, MultivariateInterp):
    """Curvilinear Interpolator using a pipeline of sklearn models."""

    def __init__(self, values, grids, pipeline, options=None) -> None:
        """Initialize a PipelineCurvilinearInterp object.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a curvilinear grid.
        grids : np.ndarray
            Functional coordinates on a curvilinear grid.
        pipeline : sklearn.pipeline.Pipeline
            Pipeline of sklearn models.

        """
        # for now, only support scipy
        super().__init__(values, grids, backend="scipy")
        self.mc_kwargs = update_mc_kwargs(options)
        self.pipeline = pipeline

        x_train = np.reshape(self.grids, (self.ndim, -1))
        y_train = np.mgrid[[slice(0, dim) for dim in self.shape]]
        y_train = np.reshape(y_train, (self.ndim, -1))

        self.models = [make_pipeline(*pipeline) for _ in range(self.ndim)]
        for dim in range(self.ndim):
            self.models[dim].fit(x_train, y_train[dim])

    def _get_coordinates(self, args):
        """Apply the sklearn pipeline to each dimension of arguments.

        Parameters
        ----------
        args : np.ndarray
            Values to interpolate for each dimension.

        Returns
        -------
        np.ndarray
            Interpolated values.

        """
        x_test = np.reshape(args, (self.ndim, -1))
        return np.array([m.predict(x_test).reshape(args[0].shape) for m in self.models])


class _PreprocessingCurvilinearInterp(PipelineCurvilinearInterp):
    """Abstract class for PipelineCurvilinearInterp with preprocessing."""

    def __init__(
        self,
        values,
        grids,
        pipeline,
        std=False,
        mc_options=None,
        pp_options=None,
    ) -> None:
        """Initialize a _PreprocessingCurvilinearInterp object. Preprocessing options
        includes standardization, polynomial features, and spline features.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a curvilinear grid.
        grids : np.ndarray
            Functional coordinates on a curvilinear grid.
        pipeline : sklearn.pipeline.Pipeline
            Pipeline of sklearn models.
        std : bool, optional
            Standardize data by removing the mean and scaling to unit variance,
            by default False
        preprocessing_options : dict, optional
            Can be None, or a dictionary with key "feature".
            If "feature" is "pol", then "degree" must be specified.
            If "feature" is "spl", then "degree" and "n_knots" must be specified.

        Raises
        ------
        AttributeError
            Feature not recognized.

        """
        self.std = std

        if pp_options is None:
            pp_options = {}

        self.pp_options = pp_options

        feature = pp_options.get("feature", None)

        if feature and isinstance(feature, str):
            degree = pp_options.get("degree", 3)
            assert isinstance(degree, int), "Degree must be an integer."
            if feature.startswith("pol"):
                pipeline.insert(0, PolynomialFeatures(degree))
            elif feature.startswith("spl"):
                n_knots = pp_options.get("n_knots", 5)
                assert isinstance(n_knots, int), "n_knots must be an integer."
                pipeline.insert(0, SplineTransformer(n_knots=n_knots, degree=degree))
            else:
                msg = f"Feature {feature} not recognized."
                raise AttributeError(msg)
        else:
            msg = f"Feature {feature} not recognized."
            raise AttributeError(msg)

        if std:
            pipeline.insert(0, StandardScaler())

        super().__init__(values, grids, pipeline, mc_options)


class RegressionCurvilinearInterp(_PreprocessingCurvilinearInterp):
    """Generalized Regression for each dimension of the curvilinear grid.
    Use regression to map from the curvilinear grid to an index grid.
    Then use map_coordinates to interpolate on the index grid.
    """

    def __init__(
        self,
        values,
        grids,
        model="elastic-net-cv",
        mc_options=None,
        pp_options=None,
        mod_options=None,
    ) -> None:
        """Initialize a GeneralizedRegressionCurvilinearInterp object.
        The model determines the regression used for each dimension.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a curvilinear grid.
        grids : np.ndarray
            Functional coordinates on a curvilinear grid.
        model : str, optional
            One of "elastic-net", "elastic-net-cv", "kernel-ridge", "svr", "sgd",
            "gaussian-process", by default "elastic-net".
        options : dict, optional
            Options for the model, by default None.

        Raises
        ------
        AttributeError
            Model is not implemented.

        """
        if mod_options is None:
            mod_options = {}

        self.model = model
        self.mod_options = mod_options

        if model == "elastic-net":
            pipeline = [ElasticNet(**mod_options)]
        elif model == "elastic-net-cv":
            pipeline = [ElasticNetCV(**mod_options)]
        elif model == "kernel-ridge":
            pipeline = [KernelRidge(**mod_options)]
        elif model == "svr":
            pipeline = [SVR(**mod_options)]
        elif model == "sgd":
            pipeline = [SGDRegressor(**mod_options)]
        elif model == "gaussian-process":
            pipeline = [GaussianProcessRegressor(**mod_options)]
        else:
            msg = f"Model {model} not implemented. Consider using `PipelineCurvilinearInterp`."
            raise AttributeError(msg)

        super().__init__(
            values,
            grids,
            pipeline,
            mc_options=mc_options,
            pp_options=pp_options,
        )
