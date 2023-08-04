from multinterp.core import (
    MC_KWARGS,
    _RegularGridInterp,
    import_backends,
    JAX_MC_KWARGS,
)
from multinterp.backend._scipy import scipy_get_coordinates, scipy_map_coordinates
from multinterp.backend._numba import numba_get_coordinates, numba_map_coordinates

AVAILABLE_BACKENDS, BACKEND_MODULES = import_backends()


def get_methods():
    GET_COORDS = {"scipy": scipy_get_coordinates, "numba": numba_get_coordinates}
    MAP_COORDS = {"scipy": scipy_map_coordinates, "numba": numba_map_coordinates}

    try:
        from multinterp.backend._cupy import cupy_get_coordinates, cupy_map_coordinates

        GET_COORDS["cupy"] = cupy_get_coordinates
        MAP_COORDS["cupy"] = cupy_map_coordinates
    except ImportError:
        pass

    try:
        from multinterp.backend._jax import jax_get_coordinates, jax_map_coordinates

        GET_COORDS["jax"] = jax_get_coordinates
        MAP_COORDS["jax"] = jax_map_coordinates

    except ImportError:
        pass

    return GET_COORDS, MAP_COORDS


GET_COORDS, MAP_COORDS = get_methods()


class MultivariateInterp(_RegularGridInterp):
    """
    Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy, cupy, or jax.
    """

    def __init__(self, values, grids, backend="scipy", options=None):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        backend : str, optional
            One of "scipy", "numba", "cupy", or "jax". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, grids, backend=backend)

        self.mc_kwargs = MC_KWARGS if backend != "jax" else JAX_MC_KWARGS
        if options:
            self.mc_kwargs = MC_KWARGS.copy()
            intersection = self.mc_kwargs.keys() & options.keys()
            self.mc_kwargs.update({key: options[key] for key in intersection})

    def _get_coordinates(self, args):
        """
        For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.
        """

        return GET_COORDS[self.backend](self.grids, args)

    def _map_coordinates(self, coords):
        """
        Uses coordinates to interpolate on the regular grid with
        `map_coordinates` from scipy or cupy, depending on backend.

        Parameters
        ----------
        coordinates : np.ndarray
            Index coordinates for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each coordinate.
        """

        return MAP_COORDS[self.backend](self.values, coords, **self.mc_kwargs)
