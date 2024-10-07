from typing import Union, Optional, Callable, SupportsIndex, Sequence
from numpy._typing import _ShapeLike, ArrayLike, DTypeLike

import numpy as np
np.seterr(all='ignore')

Tensorable = Union['Tensor', ArrayLike]

def ensure_tensor(input: Tensorable):
    return input if isinstance(input, Tensor) else Tensor(input)

class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike = None,
        requires_grad = False,
        grad_fn: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn

        self.grad: Optional[np.ndarray] = None

    def __repr__(self):
        if self.grad_fn:
            return f'tensor({self.data.round(4)}, dtype={self.dtype}, grad_fn=<{self.grad_fn.__name__}>)'
        elif self.requires_grad:
            return f'tensor({self.data.round(4)}, dtype={self.dtype}, requires_grad=True)'
        else:
            return f'tensor({self.data.round(4)}, dtype={self.dtype})'
    
    # Boolean operations (non gradient)

    def invert(self):
        return Tensor(~self.data)

    def greater(self, other: Tensorable):
        other_t = ensure_tensor(other)    
    
        return Tensor(self.data > other_t.data)
    
    def greater_equal(self, other: Tensorable):
        other_t = ensure_tensor(other)
        
        return Tensor(self.data >= other_t.data) 
    
    def less(self, other: Tensorable):
        other_t = ensure_tensor(other)
        
        return Tensor(self.data < other_t.data)
    
    def less_equal(self, other: Tensorable):
        other_t = ensure_tensor(other)
        
        return Tensor(self.data <= other_t.data)
    
    def equal(self, other: Tensorable):
        other_t = ensure_tensor(other)
        
        return Tensor(self.data == other_t.data)
    
    def not_equal(self, other: Tensorable):
        other_t = ensure_tensor(other)
        
        return Tensor(self.data != other_t.data)

    # Single operations

    def detach(self):
        return Tensor(self.data, self.dtype, requires_grad=False)

    def sign(self):
        data = np.sign(self.data)
        requires_grad = self.requires_grad
        sign_backward = None
        
        if requires_grad:
            def sign_backward(grad: np.ndarray):
                self.backward(grad * np.zeros(self.shape))
                
        return Tensor(data, None, requires_grad, sign_backward)

    def abs(self):
        data = np.abs(self.data)
        requires_grad = self.requires_grad
        abs_backward = None

        if requires_grad:
            def abs_backward(grad: np.ndarray):
                self.backward(grad * np.sign(self.data))

        return Tensor(data, None, requires_grad, abs_backward)

    def positive(self):
        data = self.data
        requires_grad = self.requires_grad
        pos_backward = None

        if requires_grad:
            def pos_backward(grad: np.ndarray):
                self.backward(grad)

        return Tensor(data, None, requires_grad, pos_backward)

    def negative(self):
        data = -self.data
        requires_grad = self.requires_grad
        neg_backward = None  

        if requires_grad:
            def neg_backward(grad: np.ndarray):
                self.backward(-grad)

        return Tensor(data, None, requires_grad, neg_backward)

    def sqrt(self):
        data = np.sqrt(self.data)
        requires_grad = self.requires_grad
        sqrt_backward = None
        
        if requires_grad:
            def sqrt_backward(grad: np.ndarray):
                self.backward(grad / (2 * np.sqrt(self.data)))
            
        return Tensor(data, None, requires_grad, sqrt_backward)

    def log(self, safe: bool = False):
        data = np.log(self.data)
        
        if safe:
            data = data.clip(min=-100)
        
        requires_grad = self.requires_grad
        log_backward = None

        if requires_grad:
            def log_backward(grad: np.ndarray):
                if safe:
                    self.backward(grad / self.data.clip(min=1e-12))
                else:
                    self.backward(grad / self.data)

        return Tensor(data, None, requires_grad, log_backward)

    def exp(self):
        data = np.exp(self.data)
        requires_grad = self.requires_grad
        exp_backward = None

        if requires_grad:            
            def exp_backward(grad: np.ndarray):
                self.backward(grad * data)

        return Tensor(data, None, requires_grad, exp_backward)

    def tanh(self):
        data = np.tanh(self.data)
        requires_grad = self.requires_grad
        tanh_backward = None

        if requires_grad:
            def tanh_backward(grad: np.ndarray):
                self.backward(grad * (1 - data ** 2))

        return Tensor(data, None, requires_grad, tanh_backward)

    def sigmoid(self):
        data = 1 / (1 + np.exp(-self.data))
        requires_grad = self.requires_grad
        sigmoid_backward = None
        
        if requires_grad:
            def sigmoid_backward(grad: np.ndarray):
                self.backward(grad * data * (1 - data))
    
        return Tensor(data, None, requires_grad, sigmoid_backward)
    
    def sin(self):
        data = np.sin(self.data)
        requires_grad = self.requires_grad
        sin_backward = None

        if requires_grad:
            def sin_backward(grad: np.ndarray):
                self.backward(grad * np.cos(self.data))

        return Tensor(data, None, requires_grad, sin_backward)

    def cos(self):
        data = np.cos(self.data)
        requires_grad = self.requires_grad
        cos_backward = None

        if requires_grad:
            def cos_backward(grad: np.ndarray):
                self.backward(grad * -np.sin(self.data))

        return Tensor(data, None, requires_grad, cos_backward)

    # Binary operations

    def add(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = self.data + other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        add_backward = None

        if requires_grad:
            def add_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad)
                
                if other_t.requires_grad:
                    other_t.backward(grad)

        return Tensor(data, None, requires_grad, add_backward)

    def sub(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = self.data - other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        sub_backward = None

        if requires_grad:
            def sub_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad)
                
                if other_t.requires_grad:
                    other_t.backward(-grad)

        return Tensor(data, None, requires_grad, sub_backward)

    def mul(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = self.data * other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        mul_backward = None

        if requires_grad:
            def mul_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * other_t.data)
                    
                if other_t.requires_grad:
                    other_t.backward(grad * self.data)

        return Tensor(data, None, requires_grad, mul_backward)

    def div(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = self.data / other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        div_backward = None

        if requires_grad:
            def div_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad / other_t.data)
                    
                if other_t.requires_grad:
                    other_t.backward(-grad * self.data / other_t.data ** 2)

        return Tensor(data, None, requires_grad, div_backward)

    def matmul(self, other: Tensorable):
        other_t = ensure_tensor(other)
        
        data = self.data @ other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        matmul_backward = None

        if requires_grad:
            def matmul_backward(grad: np.ndarray):
                if self.requires_grad:
                    # Matrix @ Matrix or Vector @ Matrix
                    if (self.ndim > 1 and other_t.ndim > 1 or self.ndim == 1 and other_t.ndim > 1):
                        self.backward(grad @ other_t.data.swapaxes(-1, -2))

                    elif self.ndim == 1 and other_t.ndim == 1: # Vector @ Vector
                        self.backward(grad * other_t.data)

                    elif self.ndim > 1 and other_t.ndim == 1: # Matrix @ Vector
                        self.backward(np.outer(grad, other_t))
                
                if other_t.requires_grad:
                    # Matrix @ Matrix or Matrix @ Vector
                    if (self.ndim > 1 and other_t.ndim > 1 or self.ndim > 1 and other_t.ndim == 1):
                        other_t.backward(self.data.swapaxes(-1, -2) @ grad)

                    elif self.ndim == 1 and other_t.ndim == 1: # Vector @ Vector
                        other_t.backward(grad * self.data)

                    elif self.ndim == 1 and other_t.ndim > 1: # Vector @ Matrix
                        other_t.backward(np.outer(self.data, grad))

        return Tensor(data, None, requires_grad, matmul_backward)

    def outer(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = np.outer(self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        outer_backward = None

        if requires_grad:
            def outer_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward((grad @ other_t.data.reshape(-1)).reshape(self.shape))
                    
                if other_t.requires_grad:
                    other_t.backward((self.data.reshape(-1) @ grad).reshape(other_t.shape))
                
        return Tensor(data, None, requires_grad, outer_backward)

    def pow(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = self.data ** other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        pow_backward = None

        if requires_grad:
            def pow_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * other_t.data * self.data ** (other_t.data - 1))
                    
                if other_t.requires_grad:
                    other_t.backward(grad * np.log(self.data) * data)

        return Tensor(data, None, requires_grad, pow_backward)

    def maximum(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = np.maximum(self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        maximum_backward = None

        if requires_grad:
            def maximum_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * (self.data > other_t.data))
                    self.backward(grad * 0.5 * (self.data == other_t.data))
                    
                if other_t.requires_grad:
                    other_t.backward(grad * (other_t.data > self.data))
                    other_t.backward(grad * 0.5 * (other_t.data == self.data))

        return Tensor(data, None, requires_grad, maximum_backward)

    def minimum(self, other: Tensorable):
        other_t = ensure_tensor(other)

        data = np.minimum(self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        minimum_backward = None

        if requires_grad:
            def minimum_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * (self.data < other_t.data))
                    self.backward(grad * 0.5 * (self.data == other_t.data))
                    
                if other_t.requires_grad:
                    other_t.backward(grad * (other_t.data < self.data))
                    other_t.backward(grad * 0.5 * (other_t.data == self.data))

        return Tensor(data, None, requires_grad, minimum_backward)

    # Batch operations

    def sum(self, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
        data = self.data.sum(axis=dim, keepdims=keepdim)
        requires_grad = self.requires_grad
        sum_backward = None

        if requires_grad:
            def sum_backward(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = np.expand_dims(grad, dim)
            
                self.backward(grad * np.ones(self.shape))

        return Tensor(data, None, requires_grad, sum_backward)

    def mean(self, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
        data = self.data.mean(axis=dim, keepdims=keepdim)
        requires_grad = self.requires_grad
        mean_backward = None

        if requires_grad:
            def mean_backward(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = np.expand_dims(grad, dim)

                # Compute size of the mean
                dim_ = list(dim) if isinstance(dim, tuple) else dim
                size = np.array(self.shape)[dim_].prod() # type: ignore
                
                self.backward(grad * np.ones(self.shape) / size)

        return Tensor(data, None, requires_grad, mean_backward)

    def var(self, dim: Optional[_ShapeLike] = None, correction: int = 1, keepdim: bool = False):
        data = self.data.var(axis=dim, keepdims=keepdim, ddof=correction)
        requires_grad = self.requires_grad
        var_backward = None

        if requires_grad:
            def var_backward(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = np.expand_dims(grad, dim)

                # Compute size of the variance
                dim_ = list(dim) if isinstance(dim, tuple) else dim
                size = np.array(self.shape)[dim_].prod() - correction # type: ignore

                # Compute mean
                mean = self.data.mean(axis=dim, keepdims=True)
 
                self.backward(grad * np.ones(self.shape) * 2 * (self.data - mean) / size)
                
        return Tensor(data, None, requires_grad, var_backward)

    def max(self, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
        data = self.data.max(axis=dim, keepdims=keepdim)
        requires_grad = self.requires_grad
        max_backward = None

        if requires_grad:
            def max_backward(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = np.expand_dims(grad, dim)

                mask = self.data == self.data.max(axis=dim, keepdims=True)
                size = mask.sum(axis=dim, keepdims=True)

                self.backward(grad * mask / size)

        return Tensor(data, None, requires_grad, max_backward)

    def min(self, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
        data = self.data.min(axis=dim, keepdims=keepdim)
        requires_grad = self.requires_grad
        min_backward = None

        if requires_grad:
            def min_backward(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = np.expand_dims(grad, dim)

                mask = self.data == self.data.min(axis=dim, keepdims=True)
                size = mask.sum(axis=dim, keepdims=True)

                self.backward(grad * mask / size)

        return Tensor(data, None, requires_grad, min_backward)

    # Shape operations

    def reshape(self, shape: _ShapeLike):
        data = self.data.reshape(shape)
        requires_grad = self.requires_grad
        reshape_backward = None

        if requires_grad:
            def reshape_backward(grad: np.ndarray):
                self.backward(grad.reshape(self.shape))

        return Tensor(data, None, requires_grad, reshape_backward)

    def transpose(self, axes: Optional[_ShapeLike] = None):
        data = self.data.transpose(axes)
        requires_grad = self.requires_grad
        transpose_backward = None
        
        if requires_grad:
            def transpose_backward(grad: np.ndarray):
                self.backward(grad.transpose(axes))
                
        return Tensor(data, None, requires_grad, transpose_backward)

    def swapaxes(self, axis0: SupportsIndex, axis1: SupportsIndex):
        data = self.data.swapaxes(axis0, axis1)
        requires_grad = self.requires_grad
        swapaxes_backward = None
        
        if requires_grad:
            def swapaxes_backward(grad: np.ndarray):
                self.backward(grad.swapaxes(axis0, axis1))

        return Tensor(data, None, requires_grad, swapaxes_backward)

    def flip(self, dims: Optional[_ShapeLike] = None):
        data = np.flip(self.data, axis=dims)
        requires_grad = self.requires_grad
        flip_backward = None
        
        if requires_grad:
            def flip_backward(grad: np.ndarray):
                self.backward(np.flip(grad, axis=dims))
                
        return Tensor(data, None, requires_grad, flip_backward)
    
    def unsqueeze(self, dim: _ShapeLike):
        data = np.expand_dims(self.data, dim)
        requires_grad = self.requires_grad
        unsqueeze_backward = None
        
        if requires_grad:
            def unsqueeze_backward(grad: np.ndarray):
                self.backward(grad.squeeze(dim)) # type: ignore
                
        return Tensor(data, None, requires_grad, unsqueeze_backward)

    def squeeze(self, dim: Optional[_ShapeLike] = None):
        data = np.squeeze(self.data, dim)
        requires_grad = self.requires_grad
        squeeze_backward = None
        
        if requires_grad:
            def squeeze_backward(grad: np.ndarray):
                if dim is None:
                    self.backward(grad.reshape(self.shape))
                else:
                    self.backward(np.expand_dims(grad, dim))
                
        return Tensor(data, None, requires_grad, squeeze_backward)
    
    # Other operations

    def stack(self, arrays: Sequence[Tensorable], dim: SupportsIndex = 0):
        tensors = [self] + [ensure_tensor(item) for item in arrays]
        
        data = np.stack([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        stack_backward = None
        
        if requires_grad:
            def stack_backward(grad: np.ndarray):
                grads = np.split(grad, len(tensors), axis=dim)
        
                for tensor, grad in zip(tensors, grads):
                    if not tensor.requires_grad:
                        continue
                        
                    tensor.backward(grad.reshape(tensor.shape))

        return Tensor(data, None, requires_grad, stack_backward)

    def cat(self, arrays: Sequence[Tensorable], dim: SupportsIndex = 0):
        tensors = [self] + [ensure_tensor(item) for item in arrays]

        data = np.concatenate([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        cat_backward = None

        if requires_grad:
            def cat_backward(grad: np.ndarray):
                # Get the indices to split the gradient
                indices = np.cumsum([t.shape[dim] for t in tensors[:-1]])

                grads = np.split(grad, indices, axis=dim)
                
                for tensor, grad in zip(tensors, grads):
                    if not tensor.requires_grad:
                        continue
                        
                    tensor.backward(grad)

        return Tensor(data, None, requires_grad, cat_backward)
    
    def where(self, condition: Tensorable, other: Tensorable):
        condition_t = ensure_tensor(condition)
        other_t = ensure_tensor(other)
        
        data = np.where(condition_t.data, self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        where_backward = None
        
        if requires_grad:
            def where_backward(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * condition_t.data)
                    
                if other_t.requires_grad:
                    other_t.backward(grad * ~condition_t.data)
        
        return Tensor(data, None, requires_grad, where_backward)

    def getitem(self, key):
        data = self.data[key]
        requires_grad = self.requires_grad
        select_backward = None
        
        if requires_grad:
            def select_backward(grad: np.ndarray):
                grad_ = np.zeros(self.shape)
                grad_[key] = grad.data
                
                self.backward(grad_)

        return Tensor(data, None, requires_grad, select_backward) 

    def iter(self):
        return iter(self.getitem(i) for i in range(self.shape[0]))
    
    # Magic methods

    def __invert__(self):
        return self.invert()

    def __gt__(self, other: Tensorable):
        return self.greater(other)
    
    def __ge__(self, other: Tensorable):
        return self.greater_equal(other)
    
    def __lt__(self, other: Tensorable):
        return self.less(other)
    
    def __le__(self, other: Tensorable):
        return self.less_equal(other)
    
    def __eq__(self, other):
        return self.equal(other)
    
    def __ne__(self, other):
        return self.not_equal(other)

    def __abs__(self):
        return self.abs()
    
    def __pos__(self):
        return self.positive()

    def __neg__(self):
        return self.negative()

    def __add__(self, other: Tensorable):
        return self.add(other)

    def __radd__(self, other: Tensorable):
        return ensure_tensor(other).add(self)

    def __sub__(self, other: Tensorable):
        return self.sub(other)

    def __rsub__(self, other: Tensorable):
        return ensure_tensor(other).sub(self)

    def __mul__(self, other: Tensorable):
        return self.mul(other)

    def __rmul__(self, other: Tensorable):
        return ensure_tensor(other).mul(self)

    def __truediv__(self, other: Tensorable):
        return self.div(other)

    def __rtruediv__(self, other: Tensorable):
        return ensure_tensor(other).div(self)

    def __matmul__(self, other: Tensorable):
        return self.matmul(other)

    def __rmatmul__(self, other: Tensorable):
        return ensure_tensor(other).matmul(self)

    def __pow__(self, other: Tensorable):
        return self.pow(other)

    def __rpow__(self, other: Tensorable):
        return ensure_tensor(other).pow(self)

    def __getitem__(self, key):
        return self.getitem(key)
        
    def __iter__(self):
        return self.iter()

    def __array__(self):
        return self.data

    # Properties

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def ndim(self): 
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def T(self):
        return self.transpose()
    
    # Backward

    def backward(self, grad: Optional[np.ndarray] = None):
        ''' Backpropagates the gradient through the computation graph '''
        
        if not self.requires_grad:
            raise RuntimeError('Cannot compute gradient on a non-required-grad tensor')

        # Initialize gradient if not provided
        if grad is None:
            grad = np.ones(self.shape, dtype=self.dtype)
        else:
            grad = np.array(grad, dtype=self.dtype)

        # Sum gradient to match data shape
        if self.shape != grad.shape:
            keepdims = self.ndim == grad.ndim

            if keepdims:
                self_shape = np.array(self.shape)
            else:
                self_shape = np.array((1,) * (grad.ndim - self.ndim) + self.shape)

            grad_shape = np.array(grad.shape)

            dim = tuple(np.where(self_shape != grad_shape)[0])

            grad = grad.sum(axis=dim, keepdims=keepdims).reshape(self.shape)

        if self.grad is None:
            # Initialize gradient
            self.grad = grad
        else:
            # Accumulate gradient
            self.grad += grad

        if grad is not None and self.grad_fn is not None:
            self.grad_fn(grad)

# Data Types

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
float16 = np.float16
float32 = np.float32
float64 = np.float64

# Factory methods

def tensor(data: ArrayLike, dtype: DTypeLike = None, requires_grad = False):
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

def binomial(n: int, p: float, shape: Optional[_ShapeLike] = None, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.random.binomial(n, p, shape), dtype=dtype, requires_grad=requires_grad)

def uniform(low: float, high: float, shape: Optional[_ShapeLike] = None, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.random.uniform(low, high, shape), dtype=dtype, requires_grad=requires_grad)

def arange(start = 0, stop = 0, step = 1, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.arange(start, stop, step), dtype=dtype, requires_grad=requires_grad)

def indices(shape: Sequence[int], sparse = False, dtype: DTypeLike = None, requires_grad = False):
    data = np.indices(shape, sparse=sparse)
    
    if sparse:
        return [Tensor(item, dtype=dtype, requires_grad=requires_grad) for item in data]
    else:
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)

# Non gradient operations

def invert(input: Tensor):
    return input.invert()

def greater(input: Tensor, other: Tensor):
    return input.greater(other)

def greater_equal(input: Tensor, other: Tensor):
    return input.greater_equal(other)

def less(input: Tensor, other: Tensor):
    return input.less(other)

def less_equal(input: Tensor, other: Tensor):
    return input.less_equal(other)

def equal(input: Tensor, other: Tensor):
    return input.equal(other)

def not_equal(input: Tensor, other: Tensor):
    return input.not_equal(other)

# Single operations

def detach(input: Tensor):
    return input.detach()

def sign(input: Tensor):
    return input.sign()

def abs(input: Tensor):
    return input.abs()

def positive(input: Tensor):
    return input.positive()

def negative(input: Tensor):
    return input.negative()

def sqrt(input: Tensor):
    return input.sqrt()

def log(input: Tensor, safe: bool = False):
    return input.log(safe)

def exp(input: Tensor):
    return input.exp()

def tanh(input: Tensor):
    return input.tanh()

def sin(input: Tensor):
    return input.sin()

def cos(input: Tensor):
    return input.cos()

# Binary operations

def add(input: Tensor, other: Tensor):
    return input.add(other)

def sub(input: Tensor, other: Tensor):
    return input.sub(other)

def mul(input: Tensor, other: Tensor):
    return input.mul(other)

def div(input: Tensor, other: Tensor):
    return input.div(other)

def matmul(input: Tensor, other: Tensor):
    return input.matmul(other)

def outer(input: Tensor, other: Tensor):
    return input.outer(other)

def pow(input: Tensor, other: Tensor):
    return input.pow(other)

def maximum(input: Tensor, other: Tensor):
    return input.maximum(other)

def minimum(input: Tensor, other: Tensor):
    return input.minimum(other)

# Batch operations

def sum(input: Tensor, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
    return input.sum(dim=dim, keepdim=keepdim)

def mean(input: Tensor, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
    return input.mean(dim=dim, keepdim=keepdim)

def var(input: Tensor, dim: Optional[_ShapeLike] = None, correction: int = 1, keepdim: bool = False):
    return input.var(dim=dim, correction=correction, keepdim=keepdim)

def max(input: Tensor, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
    return input.max(dim=dim, keepdim=keepdim)

def min(input: Tensor, dim: Optional[_ShapeLike] = None, keepdim: bool = False):
    return input.min(dim=dim, keepdim=keepdim)

# Shape operations

def reshape(input: Tensor, shape: _ShapeLike):
    return input.reshape(shape)

def transpose(input: Tensor, axes: Optional[_ShapeLike] = None):
    return input.transpose(axes)

def swapaxes(input: Tensor, axis1: SupportsIndex, axis2: SupportsIndex):
    return input.swapaxes(axis1, axis2)

def flip(input: Tensor, dims: Optional[_ShapeLike] = None):
    return input.flip(dims=dims)

def unsqueeze(input: Tensor, dim: _ShapeLike):
    return input.unsqueeze(dim)

def squeeze(input: Tensor, dim: Optional[_ShapeLike] = None):
    return input.squeeze(dim=dim)

# Other operations

def stack(arrays: Sequence[Tensor], dim: SupportsIndex = 0):
    return arrays[0].stack(arrays[1:], dim=dim)

def cat(arrays: Sequence[Tensor], dim: SupportsIndex = 0):
    return arrays[0].cat(arrays[1:], dim=dim)

def where(condition: Tensor, input: Tensor, other: Tensor):
    return input.where(condition, other)
