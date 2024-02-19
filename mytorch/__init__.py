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

def arange(start = 0, stop = 0, step = 1, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.arange(start, stop, step), dtype=dtype, requires_grad=requires_grad)

def indices(shape: Sequence[int], sparse = False, dtype: DTypeLike = None, requires_grad = False):
    data = np.indices(shape, sparse=sparse)
    
    if sparse:
        return [Tensor(item, dtype=dtype, requires_grad=requires_grad) for item in data]
    else:
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)
