from typing import Union, Optional, Callable, SupportsIndex
from numpy._typing import _ShapeLike, ArrayLike, DTypeLike

import numpy as np

class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike = None,
        requires_grad = False,
        grad_fn: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn

        self.grad: Optional[np.ndarray] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self):
        ''' Returns a string representation of the tensor '''

        return f'Tensor({self.data}, dtype={self.dtype}, requires_grad={self.requires_grad})'

    def detach(self):
        ''' Detaches the tensor from the computation graph '''
        
        return Tensor(self.data, self.dtype, requires_grad=False)

    def zero_grad(self):
        ''' Zeros the gradient of the tensor '''

        self.grad = np.zeros(self.data.shape, dtype=self.dtype)

    def pos(self):
        ''' Gets called when using +t '''

        data = self.data
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __pos__(self):
        ''' Gets called when using +t '''

        return self.pos()

    def neg(self):
        ''' Gets called when using -t '''

        data = -self.data
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(-grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __neg__(self):
        ''' Gets called when using -t '''

        return self.neg()

    def add(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t + other '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data + tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad)
                
                if tensor.requires_grad:
                    tensor.backward(grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __add__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t + other '''

        return self.add(other)

    def __radd__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using other + t '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        return tensor.add(self)

    def sub(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t - other '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data - tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad)
                
                if tensor.requires_grad:
                    tensor.backward(-grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __sub__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t - other '''

        return self.sub(other)

    def __rsub__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using other - t '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        return tensor.sub(self)

    def mul(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t * other '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data * tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * tensor.data)
                    
                if tensor.requires_grad:
                    tensor.backward(grad * self.data)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __mul__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t * other '''

        return self.mul(other)

    def __rmul__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using other * t '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        return tensor.mul(self)

    def div(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t / other '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data / tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad / tensor.data)
                    
                if tensor.requires_grad:
                    tensor.backward(-grad * self.data / tensor.data ** 2)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __truediv__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t / other '''

        return self.div(other)

    def __rtruediv__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using other / t '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        return tensor.div(self)

    def power(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t ** other '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data ** tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * tensor.data * self.data ** (tensor.data - 1))
                    
                if tensor.requires_grad:
                    tensor.backward(grad * np.log(self.data) * data)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __pow__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t ** other '''

        return self.power(other)

    def __rpow__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using other ** t '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        return tensor.power(self)

    def matmul(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t @ other '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)
        
        data = self.data @ tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    # Matrix @ Matrix or Vector @ Matrix
                    if (
                        self.data.ndim > 1 and tensor.data.ndim > 1 or
                        self.data.ndim == 1 and tensor.data.ndim > 1
                    ):
                        self.backward(grad @ tensor.data.swapaxes(-1, -2))

                    # Vector @ Vector
                    elif self.data.ndim == 1 and tensor.data.ndim == 1:
                        self.backward(grad * tensor.data)

                    # Matrix @ Vector
                    elif self.data.ndim > 1 and tensor.data.ndim == 1:
                        self.backward(np.outer(grad, tensor.data))
                
                if tensor.requires_grad:
                    # Matrix @ Matrix or Matrix @ Vector
                    if (
                        self.data.ndim > 1 and tensor.data.ndim > 1 or
                        self.data.ndim > 1 and tensor.data.ndim == 1
                    ):
                        tensor.backward(self.data.swapaxes(-1, -2) @ grad)

                    # Vector @ Vector
                    elif self.data.ndim == 1 and tensor.data.ndim == 1:
                        tensor.backward(grad * self.data)

                    # Vector @ Matrix
                    elif self.data.ndim == 1 and tensor.data.ndim > 1:
                        tensor.backward(np.outer(self.data, grad))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __matmul__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using t @ other '''

        return self.matmul(other)

    def __rmatmul__(self, other: Union['Tensor', ArrayLike]):
        ''' Gets called when using other @ t '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        return tensor.matmul(self)

    def abs(self):
        ''' Returns the absolute value of the tensor '''

        data = np.abs(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * np.sign(self.data))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __abs__(self):
        ''' Returns the absolute value of the tensor '''
        
        return self.abs()

    def sum(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the sum of the tensor '''

        data = self.data.sum(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)

                self.backward(grad * np.ones(self.data.shape))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def mean(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the mean of the tensor '''

        data = self.data.mean(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)

                # Compute size of the mean
                axis_ = list(axis) if isinstance(axis, tuple) else axis
                size = np.prod(np.array(self.data.shape)[axis_]) # type: ignore

                self.backward(grad * np.ones(self.data.shape) / size)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def var(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the variance of the tensor '''

        data = self.data.var(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)

                # Compute size of the variance
                axis_ = list(axis) if isinstance(axis, tuple) else axis
                size = np.prod(np.array(self.data.shape)[axis_]) # type: ignore

                # Compute mean
                mean = self.data.mean(axis=axis, keepdims=True)

                self.backward(grad * np.ones(self.data.shape) * 2 * (self.data - mean) / size)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def sqrt(self):
        ''' Returns the square root of the tensor '''

        data = np.sqrt(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad / (2 * data))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def log(self):
        ''' Returns the log of the tensor '''

        data = np.log(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad / self.data)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def exp(self):
        ''' Returns the exponential of the tensor '''

        data = np.exp(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * data)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def tanh(self):
        ''' Returns the hyperbolic tangent of the tensor '''

        data = np.tanh(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * (1 - data ** 2))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def sin(self):
        ''' Returns the sine of the tensor '''

        data = np.sin(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * np.cos(self.data))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def cos(self):
        ''' Returns the cosine of the tensor '''

        data = np.cos(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * -np.sin(self.data))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def maximum(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the max values of the tensor '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        data = np.maximum(self.data, tensor.data)
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * (self.data >= tensor.data))
                    
                if tensor.requires_grad:
                    tensor.backward(grad * (tensor.data >= self.data))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def minimum(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the min values of the tensor '''

        tensor = other if isinstance(other, Tensor) else Tensor(other)

        data = np.minimum(self.data, tensor.data)
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                if self.requires_grad:
                    self.backward(grad * (self.data <= tensor.data))
                    
                if tensor.requires_grad:
                    tensor.backward(grad * (tensor.data <= self.data))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def max(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the biggest value of the tensor '''

        data = self.data.max(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)

                self.backward(grad * (self.data == self.data.max(axis=axis, keepdims=True)))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def min(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the small value of the tensor '''

        data = self.data.min(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)

                self.backward(grad * (self.data == self.data.min(axis=axis, keepdims=True)))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def concatenate(self, *arrays: Union['Tensor', ArrayLike], axis: SupportsIndex = 0):
        ''' Concatenates the tensors '''
        
        tensors = [self] + [t if isinstance(t, Tensor) else Tensor(t) for t in arrays]

        data = np.concatenate([t.data for t in tensors], axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Get the indices to split the gradient
                indices = np.cumsum([t.data.shape[axis] for t in tensors[:-1]])

                grads = np.split(grad, indices, axis=axis)

                for tensor, grad in zip(tensors, grads):
                    if tensor.requires_grad:
                        tensor.backward(grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def reshape(self, shape: _ShapeLike):
        ''' Reshapes the tensor '''

        data = self.data.reshape(shape)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad.reshape(self.data.shape))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def transpose(self, axes: Optional[_ShapeLike] = None):
        ''' Transposes the tensor '''
        
        if axes is None:
            axes = tuple(reversed(range(self.data.ndim)))
        
        data = self.data.transpose(axes)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad.transpose(axes))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def swapaxes(self, axis1: SupportsIndex, axis2: SupportsIndex):
        ''' Swaps the axes of the tensor '''
        
        data = self.data.swapaxes(axis1, axis2)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad.swapaxes(axis1, axis2))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def flip(self, axis: Optional[_ShapeLike] = None):
        ''' Flips the tensor '''
        
        if axis is None:
            axis = tuple(range(self.data.ndim))
        
        data = np.flip(self.data, axis=axis)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(np.flip(grad, axis=axis))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
    def where(self, x: Union['Tensor', ArrayLike], y: Union['Tensor', ArrayLike]):
        ''' Returns elements chosen from x or y depending on condition '''
        
        tensor_x = x if isinstance(x, Tensor) else Tensor(x)
        tensor_y = y if isinstance(y, Tensor) else Tensor(y)
        
        data = np.where(self.data, tensor_x.data, tensor_y.data)
        requires_grad = tensor_x.requires_grad or tensor_y.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                tensor_x.backward(grad * self.data)
                tensor_y.backward(grad * ~self.data)
        
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
        
    def iter(self):
        ''' Returns an iterator over the tensor '''
        
        for i in range(self.data.shape[0]):
            yield self[i]
        
    def __iter__(self):
        ''' Returns an iterator over the tensor '''
        
        return self.iter()

    def getitem(self, key):
        ''' Gets called when using t[key] '''
        
        if isinstance(key, Tensor):
            key = key.data
        
        data = self.data[key]
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                grad_ = np.zeros(self.data.shape)
                grad_[key] = grad
                
                self.backward(grad_)
        
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __getitem__(self, key):
        ''' Gets called when using t[key] '''
        
        return self.getitem(key)
    
    # No grad
    
    def gt(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self > other '''
        
        tensor = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data > tensor.data)
    
    def __gt__(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self > other '''
        
        return self.gt(other)
    
    def ge(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self >= other '''
        
        tensor = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data >= tensor.data)
    
    def __ge__(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self >= other '''
        
        return self.ge(other)
    
    def lt(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self < other '''
        
        tensor = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data < tensor.data)
    
    def __lt__(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self < other '''
        
        return self.lt(other)
    
    def le(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self <= other '''
        
        tensor = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data <= tensor.data)
    
    def __le__(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self <= other '''
        
        return self.le(other)
    
    def eq(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self == other '''
        
        tensor = other if isinstance(other, Tensor) else Tensor(other)
                
        return Tensor(self.data == tensor.data)
    
    def __eq__(self, other):
        ''' Returns the truth value of self == other '''
        
        return self.eq(other)
    
    def ne(self, other: Union['Tensor', ArrayLike]):
        ''' Returns the truth value of self != other '''
        
        tensor = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data != tensor.data)
    
    def __ne__(self, other):
        ''' Returns the truth value of self != other '''
        
        return self.ne(other)

    # Properties

    @property
    def shape(self):
        ''' Returns the shape of the tensor '''

        return self.data.shape
    
    @property
    def size(self):
        ''' Returns the size of the tensor '''
        
        return self.data.size
    
    @property
    def ndim(self): 
        ''' Returns the number of dimensions of the tensor '''
        
        return self.data.ndim
    
    @property
    def T(self):
        ''' Returns the transpose of the tensor '''
        
        return self.transpose()

    @property
    def dtype(self):
        ''' Returns the data type of the tensor '''
        
        return self.data.dtype

    # End properties

    def backward(self, grad: Optional[ArrayLike] = None):
        if not self.requires_grad or self.grad is None:
            return

        # Initialize gradient if not provided
        if grad is None:
            grad = np.ones(self.data.shape, dtype=self.dtype)
        else:
            grad = np.array(grad, dtype=self.dtype)

        # Sum gradient to match data shape
        if self.data.shape != grad.shape:
            keepdims = self.data.ndim == grad.ndim

            if keepdims:
                self_shape = np.array(self.data.shape)
            else:
                self_shape = np.array((1,) * (grad.ndim - self.data.ndim) + self.data.shape)

            axis = tuple(np.where(self_shape != np.array(grad.shape))[0])

            grad = grad.sum(axis=axis, keepdims=keepdims).reshape(self.data.shape)

        # Accumulate gradient
        self.grad += grad

        if self.grad_fn is not None:
            self.grad_fn(grad) # type: ignore

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
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

def ones_like(tensor: Tensor, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.ones_like(tensor.data, dtype=dtype), requires_grad=requires_grad)

def zeros(shape: _ShapeLike, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

def zeros_like(tensor: Tensor, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.zeros_like(tensor.data, dtype=dtype), requires_grad=requires_grad)

def rand(*shape: int, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)

def randn(*shape: int, dtype: DTypeLike = None, requires_grad = False):
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

def argmin(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return Tensor(np.argmin(tensor.data, axis=axis, keepdims=keepdims)) # type: ignore

def argmax(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return Tensor(np.argmax(tensor.data, axis=axis, keepdims=keepdims)) # type: ignore

def arange(
    start = 0, 
    stop = 0, 
    step = 1, 
    dtype: DTypeLike = None, 
    requires_grad = False
):
    return Tensor(np.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad)

# Operations

def sum(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.sum(axis=axis, keepdims=keepdims)

def mean(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.mean(axis=axis, keepdims=keepdims)

def var(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.var(axis=axis, keepdims=keepdims)

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

def abs(tensor: Tensor):
    return tensor.abs()

def maximum(tensor: Tensor, other: Union[ArrayLike, 'Tensor']):
    return tensor.maximum(other)

def minimum(tensor: Tensor, other: Union[ArrayLike, 'Tensor']):
    return tensor.minimum(other)

def max(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.max(axis=axis, keepdims=keepdims)

def min(tensor: Tensor, axis: Optional[_ShapeLike] = None, keepdims = False):
    return tensor.min(axis=axis, keepdims=keepdims)

def concatenate(*tensors: Union[ArrayLike, 'Tensor'], axis: SupportsIndex = 0):
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