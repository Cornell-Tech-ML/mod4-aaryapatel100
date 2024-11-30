from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the function to the given scalar inputs.

        Args:
        ----
            *vals (ScalarLike): Input scalar-like values.

        Returns:
        -------
            Scalar: Result of applying the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass for the addition operation.

        Args:
        ----
            ctx (Context): The context to store any information for backward computation.
            a (float): The first input scalar.
            b (float): The second input scalar.

        Returns:
        -------
            float: The result of adding `a` and `b`.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the addition function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative with respect to the output.

        Returns:
        -------
            Tuple[float, ...]: A tuple containing the derivatives with respect to both inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the logarithm function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The logarithm of `a`.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the logarithm function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input `a`.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of multiplying `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the multiplication function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients with respect to both inputs.

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The inverse of `a`.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the inverse function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input `a`.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the negation function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The negation of `a`.

        """
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the negation function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input `a`.

        """
        return float(operators.neg(d_output))


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        res = operators.sigmoid(a)
        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the sigmoid function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input `a`.

        """
        (sigmoid_a,) = ctx.saved_values
        return d_output * sigmoid_a * (1 - sigmoid_a)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying ReLU to `a`.

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the ReLU function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input `a`.

        """
        (a,) = ctx.saved_values
        return float(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The exponential of `a`.

        """
        exp_a = operators.exp(a)
        ctx.save_for_backward(exp_a)
        return exp_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the exponential function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input `a`.

        """
        (exp_a,) = ctx.saved_values
        return d_output * exp_a


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the less than function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of comparing if `a` is less than `b`.

        """
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the less than function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, ...]: Always returns (0, 0) since less than is non-differentiable.

        """
        return 0, 0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the equality function.

        Args:
        ----
            ctx (Context): The context to store information for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of comparing if `a` is equal to `b`.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the equality function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            d_output (float): The derivative of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, ...]: Always returns (0, 0) since equality is non-differentiable.

        """
        return 0, 0
