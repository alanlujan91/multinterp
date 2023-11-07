from __future__ import annotations

from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch.optim import Adam

from multinterp.grids import _UnstructuredGrid


class _SimpleExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GaussianProcessRegression(_UnstructuredGrid):
    def __init__(self, values, grids):
        super().__init__(values, grids, backend="torch")

        self._likelihood = GaussianLikelihood()
        self._model = _SimpleExactGPModel(self.grids[0], self.values, self._likelihood)

        self._train()

    def _train(self, training_iter=50):
        self._model.train()
        self._likelihood.train()

        optimizer = Adam(self._model.parameters(), lr=0.1)

        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from self._model
            output = self._model(self.grids[0])
            # Calc loss and backprop gradients
            loss = -mll(output, self.values)
            loss.backward()
            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    self._model.covar_module.base_kernel.lengthscale.item(),
                    self._model.likelihood.noise.item(),
                )
            )
            optimizer.step()

        self._model.eval()
        self._likelihood.eval()

    def __call__(self, *args):
        return self._model(*args)
