from __future__ import annotations

import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MultiDeviceKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

from multinterp.backend.LBFGS import FullBatchLBFGS
from multinterp.grids import _UnstructuredGrid

N_DEVICES = torch.cuda.device_count()


class _SimpleExactGPModel(ExactGP):
    """
    A simple Gaussian Process model.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood,
    ):
        """
        Initialize the GP model.

        Args:
            train_x: Training input data
            train_y: Training output data
            likelihood: Likelihood function
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass through the GP model.

        Args:
            x: Input data

        Returns:
            Distribution over the input data
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class _PipelineExactGPModel(ExactGP):
    """
    A Gaussian Process model for data extraction.
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        mean_module=None,
        covar_module=None,
        distribution=None,
    ):
        """
        Initialize the GP model.

        Args:
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
        """
        Forward pass through the GP model.

        Args:
            x: Input data

        Returns:
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
        n_devices=N_DEVICES,
        output_device=None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or ConstantMean()
        base_covar_module = covar_module or ScaleKernel(RBFKernel())
        self.distribution = distribution or MultivariateNormal

        assert (
            0 < n_devices <= N_DEVICES
        ), f"n_devices must be between 1 and {N_DEVICES}"

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

        self._likelihood = GaussianLikelihood()
        self._model = _PipelineGPUExactGPModel(
            self.grids[0],
            self.values,
            likelihood=self._likelihood,
            n_devices=2,
        )

        self._train()

    def _train(self, training_iter=50, preconditioner_size=100):
        self._model.train()
        self._likelihood.train()

        self.grids = self.grids.cuda()
        self.values = self.values.cuda()
        self._model = self._model.cuda()
        self._likelihood = self._likelihood.cuda()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = FullBatchLBFGS(self._model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        with gpytorch.settings.max_preconditioner_size(preconditioner_size):

            def closure():
                optimizer.zero_grad()
                output = self._model(self.grids[0])
                return -mll(output, self.values)

            loss = closure()
            loss.backward()

            for _i in range(training_iter):
                options = {"closure": closure, "current_loss": loss, "max_ls": 10}
                loss, _, _, _, _, _, _, fail = optimizer.step(options)

                # print(
                #     "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                #     % (
                #         i + 1,
                #         training_iter,
                #         loss.item(),
                #         self._model.covar_module.module.base_kernel.lengthscale.item(),
                #         self._model.likelihood.noise.item(),
                #     )
                # )

                if fail:
                    # print("Convergence reached!")
                    break

        self._model.eval()
        self._likelihood.eval()

    def __call__(self, *args):
        return self._model(*args)


def train(
    train_x,
    train_y,
    model,
    likelihood,
    n_devices,
    preconditioner_size,
    n_training_iter,
    verbose=False,
    n_skip=10,
):
    model.train()
    likelihood.train()

    optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            return -mll(output, train_y)

        loss = closure()
        loss.backward()

        lengthscale = model.covar_module.module.base_kernel.lengthscale.item()
        noise = model.likelihood.noise.item()

        for i in range(n_training_iter):
            options = {"closure": closure, "current_loss": loss, "max_ls": 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)

            if verbose:
                if i % n_skip == 0:  # Print progress every 10 steps if verbose is True
                    print(
                        f"Iter {i+1}/{n_training_iter} - Loss: {loss.item():.3f}   lengthscale: {lengthscale:.3f}   noise: {noise:.3f}"
                    )

            if fail:
                print("Convergence reached!")
                break

    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood
