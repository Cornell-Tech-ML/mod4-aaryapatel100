from typing import Tuple, Optional

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor, zeros


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw

    # Create a view of the input tensor that's reshaped for pooling
    # We want to group the elements that will be pooled together
    input_reshaped = input.contiguous()

    # Reshape to batch x channel x new_height x kernel_height x new_width x kernel_width
    input_strided = input_reshaped.view(batch, channel, new_height, kh, new_width, kw)

    # Combine the kernel dimensions into one dimension
    # Final shape: batch x channel x new_height x new_width x (kernel_height * kernel_width)
    output = input_strided.permute(0, 1, 2, 4, 3, 5).contiguous()
    output = output.view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    # Calculate mean and reshape to remove the last dimension
    pooled = output.mean(4)
    return pooled.view(pooled.shape[0], pooled.shape[1], new_height, new_width)


def argmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply argmax over

    Returns:
    -------
        A tensor of the same size as input with 1.0 in the argmax position and 0 elsewhere.

    """
    if dim is None:
        max_vals = input.f.max_reduce(input, 0)
    else:
        max_vals = input.f.max_reduce(input, int(input._ensure_tensor(dim).item()))
    return max_vals == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass of max.

        Args:
        ----
            ctx: context
            input: input tensor
            dim: dimension to apply max over

        Returns:
        -------
            Tuple of Tensor maximum value of each row and Tensor argmax value of each row

        """
        ctx.save_for_backward(input, dim)
        return input.f.max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass of max.

        Args:
        ----
            ctx: context
            grad_output: gradient of output

        Returns:
        -------
            Tuple of input gradient and None

        """
        input, dim = ctx.saved_values
        # use 1-hot tensor to gradient (only the max value should be remaining)
        one_hot = argmax(input, int(dim.item()))
        return grad_output * one_hot, tensor([0.0])


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the max over the dimension 'dim' of the input tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply max over

    Returns:
    -------
        A tensor of the same size as input with values in the range (0,inf) that represent the maximum value over the given dimension.

    """
    # Doesn't use same methodology as argmax for calculating max_vals
    # using max_reduce because Max.apply stores gradient for autodiff.
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0.0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Computes the softmax of the input tensor over the given dimension.

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax over

    Returns:
    -------
        A tensor of the same size as input with values in the range (0,1) that sum to 1 over the given dimension.

    """
    e = (input - max(input, dim)).exp()
    return e / e.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Computes the log of the softmax of the input tensor over the given dimension.

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax over

    Returns:
    -------
        A tensor of the same size as input with values in the range (-inf,0)

    """
    max_val = max(input, dim)
    # input values - LSE using max
    return input - ((input - max_val).exp()).sum(dim).log() - max_val


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling 2D.

    Args:
    ----
        input: input tensor with shape batch x channel x height x width
        kernel: tuple representing the height and width of the pooling window

    Returns:
    -------
        A tensor with shape batch x channel x new_height x new_width after applying max pooling.

    """
    output, new_height, new_width = tile(input, kernel)
    # Calculate mean and reshape to remove the last dimension
    pooled = max(output, 4)
    return pooled.view(pooled.shape[0], pooled.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Randomly sets elements to zero with probability rate.

    Args:
    ----
        input: input tensor
        rate: probability of dropping out
        ignore: if true, dropout is not applied

    Returns:
    -------
        tensor with dropout applied

    """
    if ignore or rate == 0.0:
        return input
    elif rate == 1.0:
        return zeros(input.shape)
    return input * (rand(input.shape) > rate)
