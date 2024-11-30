# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from numpy._core.defchararray import index
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    """A wrapper around numba's CUDA device JIT compilation.

    This function compiles the given function `fn` to run on a CUDA-capable GPU.
    It uses numba's `jit` with `device=True`, allowing `fn` to be called
    from other CUDA device functions or kernels.

    Args:
    ----
        fn: The function to compile for CUDA device execution.
        **kwargs: Additional keyword arguments to pass to the JIT compiler.

    Returns:
    -------
        The device-jitted version of the function, callable within CUDA kernels.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003
    """A wrapper around numba's CUDA JIT compilation.

    This function compiles the given function `fn` to run on a CUDA-capable GPU.
    It uses numba's `jit` with `target="cuda"`, allowing `fn` to be called
    from other CUDA device functions or kernels.

    Args:
    ----
        fn: The function to compile for CUDA execution.
        **kwargs: Additional keyword arguments to pass to the JIT compiler.

    Returns:
    -------
        The jitted version of the function, callable within CUDA kernels.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """CUDA higher-order tensor reduce function.

        This function compiles a given reduction function to run on a CUDA-capable GPU.
        It reduces a tensor `a` along a specified dimension `dim` using the provided
        binary operator `fn`, starting from the value `start`.

        Args:
        ----
            fn: A binary function that maps two floats to a float, used for reduction.
            start: The starting value for the reduction.

        Returns:
        -------
            A function that takes a tensor `a` and an integer `dim`, and returns
            a new tensor reduced along the given dimension.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        """CUDA Tensor map function.

        Requirements:

        * All data must be first moved to shared memory.
        * Only read each cell in `in_storage` once.
        * Only write to global memory once per kernel.

        Should work for any tensor shapes that broadcast as long as ::

          assert len(out_shape) == len(in_shape)

        Returns
        -------
            None : Fills in `out`

        """
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)

            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = float(a[i])
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    # Reduce within the block
    if i < size:
        for j in [1, 2, 4, 8, 16]:
            if pos % (2 * j) == 0:
                cache[pos] += cache[pos + j]
                cuda.syncthreads()
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]
            
    # step = 1
    # while step < BLOCK_DIM:
    #     if pos % (2 * step) == 0 and pos + step < BLOCK_DIM:
    #         cache[pos] += cache[pos + step]
    #     step *= 2
    #     cuda.syncthreads()

    # if pos == 0:
    #     out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Compute the sum of all values in a tensor in parallel on the GPU.

    Args:
    ----
        a (Tensor): the tensor to sum

    Returns:
    -------
        TensorData: A tensor with a single element containing the sum of all values in the input tensor.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                a_pos = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[a_pos]
                cuda.syncthreads()
                x = 0
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                        cuda.syncthreads()
                    x += 1
            if pos == 0:
                out[o] = cache[0]
    
        # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # # Only proceed if thread is within output size
        # if i < out_size:
        #     # Convert linear index to multidimensional output index
        #     to_index(i, out_shape, out_index)

        #     # Initialize cache with reduce_value
        #     local_reduce = reduce_value

        #     # Reduce along the specified dimension
        #     reduce_size = a_shape[reduce_dim]
        #     for j in range(reduce_size):
        #         # Set the reduce dimension index
        #         out_index[reduce_dim] = j

        #         # Get position and update cache
        #         in_pos = index_to_position(out_index, a_strides)
        #         local_reduce = fn(local_reduce, a_storage[in_pos])

        #     # Write final result to output
        #     cache[pos] = local_reduce
        #     cuda.syncthreads()
        #     out[pos] = cache[pos]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Get thread indices
    tx = cuda.threadIdx.x  # Thread x-coordinate
    ty = cuda.threadIdx.y  # Thread y-coordinate

    # Each thread computes one element of the output matrix
    if tx < size and ty < size:
        # Initialize output value
        tmp = 0.0

        # Load input matrices into shared memory
        a_shared[tx, ty] = a[tx * size + ty]
        b_shared[tx, ty] = b[tx * size + ty]

        # Ensure all threads have loaded their data
        cuda.syncthreads()

        # Compute dot product for this element
        for k in range(size):
            tmp += a_shared[tx, k] * b_shared[k, ty]

        # Write result to global memory
        out[tx * size + ty] = tmp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """A practice square MM kernel to prepare for matmul.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute
    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # raise NotImplementedError("Need to implement for Task 3.4")
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # Loading in shared arrays

    value = 0.0
    for k_start in range(0, a_shape[2], BLOCK_DIM):
        k = k_start + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride
                + i * a_strides[1]
                + k * a_strides[2]
            ]
        k = k_start + pi
        if j < b_shape[2] and k < b_shape[1]:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride
                + k * b_strides[1]
                + j * b_strides[2]
            ]
        cuda.syncthreads()
        
        for k in range(BLOCK_DIM):
            if (k_start + k) < a_shape[2]:
                value += a_shared[pi, k] * b_shared[k, pj]
        
    if i < out_shape[1] and j < out_shape[2]:
        pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[pos] = value
    
    # MY LOGIC
    # # For loop through # of blocks horizontally in tensor A (a_shape[-1] // BLOCK_DIM)
    # # + 1 is because range is exclusive
    # value = 0.0
    # for block in range((a_shape[-1] // BLOCK_DIM) + 1):
    #     # Only compute if within output dimensions
    #     if i < a_shape[-2] and block * BLOCK_DIM + pj < a_shape[-1]:
    #         # Calculate position of the value being moved to storage using strides (slide in dir. of pj)
    #         pos = (
    #             batch * a_batch_stride
    #             + i * a_strides[-2]
    #             + (block * BLOCK_DIM + pj) * a_strides[-1]
    #         )
    #         a_shared[pi, pj] = a_storage[pos]
    #     else:
    #         a_shared[pi, pj] = 0.0

    #     # Do the same thing for b_shared using its dimensions but instead slide in other direction (moving in dir. of pi)
    #     if block * BLOCK_DIM + pi < b_shape[-2] and j < b_shape[-1]:
    #         # Calculate position of the value being moved to storage using strides
    #         pos = (
    #             batch * b_batch_stride
    #             + (block * BLOCK_DIM + pi) * b_strides[-2]
    #             + j * b_strides[-1]
    #         )
    #         b_shared[pi, pj] = b_storage[pos]
    #     else:
    #         b_shared[pi, pj] = 0.0

    #     cuda.syncthreads()
    #     # Do necessary dot product calculations between these internal shared blocks
    #     # (thus using local positioning, pi and pj, before current blocks slide)
    #     for k in range(BLOCK_DIM):
    #         value += a_shared[pi, k] * b_shared[k, pj]
    #     cuda.syncthreads()

    # # Check if thread is within dimensions and find position in global out and assign calculated value
    # if i < out_shape[-2] and j < out_shape[-1]:
    #     pos = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
    #     out[pos] = value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
