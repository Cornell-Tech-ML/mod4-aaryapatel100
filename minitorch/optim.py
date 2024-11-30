from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Resets the gradients of all model parameters to zero.

        This function iterates over all parameters stored within the optimizer.
        For each parameter, if it has an associated derivative or gradient,
        these are set to None, effectively resetting them to zero. This is a
        necessary step before performing a new optimization step, ensuring that
        gradients do not accumulate from previous updates.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Performs a single optimization step.

        For each parameter, this function updates the parameter by moving in the
        direction of the negative gradient. The learning rate is the magnitude of
        the update.

        Args:
        ----
            None

        Returns:
        -------
            None

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
