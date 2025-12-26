from __future__ import annotations

import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MultiDeviceKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch.optim import Adam

from multinterp.backend.LBFGS import FullBatchLBFGS
from multinterp.grids import _UnstructuredGrid

DEVICE_COUNT = torch.cuda.device_count()


class _SimpleExactGPModel(ExactGP):
    """A simple Gaussian Process model."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood,
    ):
        """Initialize the GP model.

        Args:
        ----
            train_x: Training input data
            train_y: Training output data
            likelihood: Likelihood function

        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through the GP model.

        Args:
        ----
            x: Input data

        Returns:
        -------
            Distribution over the input data

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class _PipelineExactGPModel(ExactGP):
    """A Gaussian Process model for data extraction."""

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        mean_module=None,
        covar_module=None,
        distribution=None,
    ):
        """Initialize the GP model.

        Args:
        ----
            train_x: Training input data
            train_y: Training output data
            likelihood: Likelihood function (default: GaussianLikelihood)
            mean_module: Mean function (default: ConstantMean)
            covar_module: Covariance function (default: ScaleKernel(RBFKernel))
            distribution: Distribution function (default: MultivariateNormal)

        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or ScaleKernel(RBFKernel())
        self.distribution = distribution or MultivariateNormal

    def forward(self, x):
        """Forward pass through the GP model.

        Args:
        ----
            x: Input data

        Returns:
        -------
            Distribution over the input data

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.distribution(mean_x, covar_x)


class _PipelineGPUExactGPModel(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        mean_module=None,
        covar_module=None,
        distribution=None,
        n_devices=DEVICE_COUNT,
        output_device=None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or ConstantMean()
        base_covar_module = covar_module or ScaleKernel(RBFKernel())
        self.distribution = distribution or MultivariateNormal

        assert (
            0 < n_devices <= DEVICE_COUNT
        ), f"n_devices must be between 1 and {DEVICE_COUNT}"

        if n_devices == 1:
            self.covar_module = base_covar_module
        else:
            self.covar_module = MultiDeviceKernel(
                base_covar_module,
                device_ids=range(n_devices),
                output_device=output_device or torch.device("cuda:0"),
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.distribution(mean_x, covar_x)


class GaussianProcessRegression(_UnstructuredGrid):
    def __init__(self, values, grids):
        super().__init__(values, grids, backend="torch")

        self.grids = self.grids.T

        self._likelihood = GaussianLikelihood()
        self._model = _PipelineGPUExactGPModel(
            self.grids,
            self.values,
            likelihood=self._likelihood,
            n_devices=2,
        )

        self._to_cuda()
        self._train()

    def _to_cuda(self):
        self.grids = self.grids.cuda()
        self.values = self.values.cuda()
        self._model = self._model.cuda()
        self._likelihood = self._likelihood.cuda()

    def _train(self, training_iter=50, _preconditioner_size=100):
        _train_lbfgs(
            self._model,
            self._likelihood,
            self.grids,
            self.values,
            training_iter,
            verbose=True,
        )

    def __call__(self, *args):
        args = torch.as_tensor(
            args,
            device=self.grids.device,
        ).T

        # Get into evaluation (predictive posterior) mode
        self._model.eval()
        self._likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self._model(args)


def _train_simple(
    model,
    likelihood,
    train_x,
    train_y,
    training_iter=50,
    verbose=False,
    n_skip=10,
):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = Adam(model.parameters(), lr=0.1)
    # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        if verbose and i % n_skip == 0:
            _loss = loss.item()
            _lengthscale = model.covar_module.module.base_kernel.lengthscale.item()
            _noise = model.likelihood.noise.item()

        optimizer.step()


def _eval_simple(model, likelihood, test_x):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(test_x))


def _train_lbfgs(
    model,
    likelihood,
    train_x,
    train_y,
    preconditioner_size=100,
    training_iter=50,
    verbose=False,
    n_skip=10,
):
    model.train()
    likelihood.train()

    optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            return -mll(output, train_y)

        loss = closure()
        loss.backward()

        for i in range(training_iter):
            options = {"closure": closure, "current_loss": loss, "max_ls": 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)

            # Print progress every n_skip steps if verbose is True
            if verbose and i % n_skip == 0:
                _loss = loss.item()
                _lengthscale = model.covar_module.module.base_kernel.lengthscale.item()
                _noise = model.likelihood.noise.item()

            if fail:
                if verbose:
                    pass
                break


def _train_pipeline(
    train_x,
    train_y,
    model,
    likelihood,
    optimizer,
    mll,
    training_iter=50,
    verbose=False,
    n_skip=10,
):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        if verbose and i % n_skip == 0:
            _loss = loss.item()
            _lengthscale = model.covar_module.module.base_kernel.lengthscale.item()
            _noise = model.likelihood.noise.item()

        optimizer.step()
