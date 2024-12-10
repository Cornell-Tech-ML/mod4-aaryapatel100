from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """A wrapper around numba's njit which always inlines the function, and
    also provides type hints.

    Args:
    ----
        fn: The function to compile with njit.
        **kwargs: Additional keyword arguments to pass to njit.

    Returns:
    -------
        The njitted version of the function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # Implementation for Task 4.1 using tensor_zip and tensor_reduce
    for batch in prange(batch_):
        for out_channel in prange(out_channels):
            for w in prange(out_width):
                # Position to write in the output
                out_pos = (
                    batch * out_strides[0]
                    + out_channel * out_strides[1]
                    + w * out_strides[2]
                )
                acc = 0.0
                for in_channel in range(in_channels):
                    for k in range(kw):
                        if not reverse:
                            if w + k < width:
                                # Forward pass: standard convolution
                                in_pos = (
                                    batch * s1[0] + in_channel * s1[1] + (w + k) * s1[2]
                                )
                                weight_pos = (
                                    out_channel * s2[0] + in_channel * s2[1] + k * s2[2]
                                )
                                acc += input[in_pos] * weight[weight_pos]
                        else:
                            # Backward pass: reversed convolution
                            if w - k >= 0 and w < width:
                                in_pos = batch * s1[0] + in_channel * s1[1] + w * s1[2]
                                weight_pos = (
                                    out_channel * s2[0] + in_channel * s2[1] + k * s2[2]
                                )
                                acc += input[in_pos] * weight[weight_pos]
                out[out_pos] = acc


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the 1D convolution function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            grad_output (Tensor): The derivative of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple where the first element is the gradient with respect to the input tensor `t1`, and the second element is the gradient with respect to the weight tensor `t2`.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides

    # Main convolution loop with parallel processing
    for b in prange(batch_):
        for oc in prange(out_channels):
            for h in range(height):
                for w in range(width):
                    # Calculate output position once
                    out_pos = (
                        b * out_strides[0]
                        + oc * out_strides[1]
                        + h * out_strides[2]
                        + w * out_strides[3]
                    )
                    if out_pos >= out_size:
                        continue

                    acc = 0.0
                    for ic in range(in_channels):
                        for i in range(kh):
                            for j in range(kw):
                                # Calculate positions
                                h_pos = h + i if not reverse else h - (kh - 1) + i
                                w_pos = w + j if not reverse else w - (kw - 1) + j

                                if (
                                    h_pos < 0
                                    or h_pos >= height
                                    or w_pos < 0
                                    or w_pos >= width
                                ):
                                    continue

                                # Calculate input position
                                in_pos = (
                                    b * s1[0]
                                    + ic * s1[1]
                                    + h_pos * s1[2]
                                    + w_pos * s1[3]
                                )

                                # Calculate weight position
                                weight_pos = (
                                    oc * s2[0] + ic * s2[1] + i * s2[2] + j * s2[3]
                                )

                                # Add to accumulator
                                acc += input[in_pos] * weight[weight_pos]

                    out[out_pos] = acc


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the 2D convolution function during the backward pass.

        Args:
        ----
            ctx (Context): The context storing information from the forward pass.
            grad_output (Tensor): The derivative of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple of two tensors, each of which is the gradient with respect to the input `input` and `weight` respectively.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
