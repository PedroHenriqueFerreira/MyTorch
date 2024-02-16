from typing import Union, Optional, SupportsIndex, Sequence
from numpy._typing import _ShapeLike, ArrayLike, DTypeLike

from mytorch.autograd import Tensor

import numpy as np

# Typings

int16 = np.int16
int32 = np.int32
int64 = np.int64
float16 = np.float16
float32 = np.float32
float64 = np.float64

# Factories for creating tensors

def tensor(data: ArrayLike,  dtype: DTypeLike = None, requires_grad = False):
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)

def ones(shape: _ShapeLike, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.ones(shape), dtype=dtype, requires_grad=requires_grad)

def ones_like(tensor: Tensor, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.ones_like(tensor.data), dtype=dtype, requires_grad=requires_grad)

def zeros(shape: _ShapeLike, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad)

def zeros_like(tensor: Tensor, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.zeros_like(tensor.data), dtype=dtype, requires_grad=requires_grad)

def rand(*shape: int, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.random.rand(*shape), dtype=dtype, requires_grad=requires_grad)

def randn(*shape: int, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad)

def uniform(
    low: float, 
    high: float, 
    shape: Optional[_ShapeLike] = None, 
    dtype: DTypeLike = None,
    requires_grad = False
):
    return Tensor(np.random.uniform(low, high, shape), dtype=dtype, requires_grad=requires_grad)

def argmax(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return Tensor(np.argmax(tensor.data, axis=axis, keepdims=keepdims)) # type: ignore

def argmin(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return Tensor(np.argmin(tensor.data, axis=axis, keepdims=keepdims)) # type: ignore

def arange(start = 0, stop = 0, step = 1, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.arange(start, stop, step), dtype=dtype, requires_grad=requires_grad)

def indices(shape: Sequence[int], sparse = False, dtype: DTypeLike = None, requires_grad = False):
    data = np.indices(shape, sparse=sparse)
    
    if sparse:
        return [Tensor(item, dtype=dtype, requires_grad=requires_grad) for item in data]
    else:
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)

# Operations

def abs(tensor: Tensor):
    return tensor.abs()

def sqrt(tensor: Tensor):
    return tensor.sqrt()

def log(tensor: Tensor):
    return tensor.log()

def exp(tensor: Tensor):
    return tensor.exp()

def tanh(tensor: Tensor):
    return tensor.tanh()

def sin(tensor: Tensor):
    return tensor.sin()

def cos(tensor: Tensor):
    return tensor.cos()

def sum(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.sum(axis=axis, keepdims=keepdims)

def mean(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.mean(axis=axis, keepdims=keepdims)

def var(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.var(axis=axis, keepdims=keepdims)

def maximum(tensor: Tensor, other: Union[Tensor, ArrayLike]):
    return tensor.maximum(other)

def minimum(tensor: Tensor, other: Union[Tensor, ArrayLike]):
    return tensor.minimum(other)

def max(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.max(axis=axis, keepdims=keepdims)

def min(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.min(axis=axis, keepdims=keepdims)

def concatenate(*tensors: Union[Tensor, ArrayLike], axis: SupportsIndex = 0):
    tensor = tensors[0] if isinstance(tensors[0], Tensor) else Tensor(tensors[0])
    
    return tensor.concatenate(*tensors[1:], axis=axis)

def reshape(tensor: Tensor, shape: _ShapeLike):
    return tensor.reshape(shape)

def transpose(tensor: Tensor, axes: Optional[_ShapeLike] = None):
    return tensor.transpose(axes=axes)

def swapaxes(tensor: Tensor, axis1: SupportsIndex, axis2: SupportsIndex):
    return tensor.swapaxes(axis1, axis2)

def flip(tensor: Tensor, axis: Optional[_ShapeLike] = None):
    return tensor.flip(axis=axis)

def where(condition: Tensor, x: Union[Tensor, ArrayLike], y: Union[Tensor, ArrayLike]):
    return condition.where(x, y)