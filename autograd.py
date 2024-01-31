from typing import Union, Optional, Callable, SupportsIndex
from numpy._typing import _ShapeLike, ArrayLike

import numpy as np

Arrayable = Union[int, float, tuple, list, np.ndarray]

def ensure_array(arrayable: ArrayLike) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable

    return np.array(arrayable)

Tensorable = Union['Tensor', Arrayable]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable

    return Tensor(tensorable)

class Tensor:
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        grad_fn: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn

        self.grad: Optional[np.ndarray] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self):
        ''' Returns a string representation of the tensor '''

        return f'Tensor({self.data}, requires_grad={self.requires_grad})'

    def zero_grad(self):
        ''' Zeros the gradient of the tensor '''

        self.grad = np.zeros(self.data.shape)

    def pos(self):
        ''' Gets called when using +t '''

        data = self.data
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad)

        return Tensor(data, requires_grad, grad_fn)

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

        return Tensor(data, requires_grad, grad_fn)

    def __neg__(self):
        ''' Gets called when using -t '''

        return self.neg()

    def add(self, other: Tensorable):
        ''' Gets called when using t + other '''

        tensor = ensure_tensor(other)

        data = self.data + tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad)
                tensor.backward(grad)

        return Tensor(data, requires_grad, grad_fn)

    def __add__(self, other: Tensorable):
        ''' Gets called when using t + other '''

        return self.add(other)

    def __radd__(self, other: Tensorable):
        ''' Gets called when using other + t '''

        return ensure_tensor(other).add(self)

    def sub(self, other: Tensorable):
        ''' Gets called when using t - other '''

        tensor = ensure_tensor(other)

        data = self.data - tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad)
                tensor.backward(-grad)

        return Tensor(data, requires_grad, grad_fn)

    def __sub__(self, other: Tensorable):
        ''' Gets called when using t - other '''

        return self.sub(other)

    def __rsub__(self, other: Tensorable):
        ''' Gets called when using other - t '''

        return ensure_tensor(other).sub(self)

    def mul(self, other: Tensorable):
        ''' Gets called when using t * other '''

        tensor = ensure_tensor(other)

        data = self.data * tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * tensor.data)
                tensor.backward(grad * self.data)

        return Tensor(data, requires_grad, grad_fn)

    def __mul__(self, other: Tensorable):
        ''' Gets called when using t * other '''

        return self.mul(other)

    def __rmul__(self, other: Tensorable):
        ''' Gets called when using other * t '''

        return ensure_tensor(other).mul(self)

    def div(self, other: Tensorable):
        ''' Gets called when using t / other '''

        tensor = ensure_tensor(other)

        data = self.data / tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad / tensor.data)
                tensor.backward(-grad * self.data / tensor.data ** 2)

        return Tensor(data, requires_grad, grad_fn)

    def __truediv__(self, other: Tensorable):
        ''' Gets called when using t / other '''

        return self.div(other)

    def __rtruediv__(self, other: Tensorable):
        ''' Gets called when using other / t '''

        return ensure_tensor(other).div(self)

    def power(self, other: Tensorable):
        ''' Gets called when using t ** other '''

        tensor = ensure_tensor(other)

        data = self.data ** tensor.data
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * tensor.data * self.data ** (tensor.data - 1))
                tensor.backward(grad * np.log(self.data) * data)

        return Tensor(data, requires_grad, grad_fn)

    def __pow__(self, other: Tensorable):
        ''' Gets called when using t ** other '''

        return self.power(other)

    def __rpow__(self, other: Tensorable):
        ''' Gets called when using other ** t '''

        return ensure_tensor(other).power(self)

    def matmul(self, other: Tensorable):
        ''' Gets called when using t @ other '''

        tensor = ensure_tensor(other)
        data = self.data @ tensor.data
        grad_fn = None

        requires_grad = self.requires_grad or tensor.requires_grad

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Matrix @ Matrix
                if self.data.ndim > 1 and tensor.data.ndim > 1:
                    self.backward(grad @ tensor.data.swapaxes(-1, -2))
                    tensor.backward(self.data.swapaxes(-1, -2) @ grad)

                # Vector @ Vector
                elif self.data.ndim == 1 and tensor.data.ndim == 1:
                    self.backward(grad * tensor.data)
                    tensor.backward(grad * self.data)

                # Matrix @ Vector
                elif self.data.ndim > 1 and tensor.data.ndim == 1:
                    self.backward(np.outer(grad, tensor.data))
                    tensor.backward(self.data.swapaxes(-1, -2) @ grad)

                # Vector @ Matrix
                elif self.data.ndim == 1 and tensor.data.ndim > 1:
                    self.backward(grad @ tensor.data.swapaxes(-1, -2))
                    tensor.backward(np.outer(self.data, grad))

        return Tensor(data, requires_grad, grad_fn)

    def __matmul__(self, other: Tensorable):
        ''' Gets called when using t @ other '''

        return self.matmul(other)

    def __rmatmul__(self, other: Tensorable):
        ''' Gets called when using other @ t '''

        return ensure_tensor(other).matmul(self)

    def abs(self):
        ''' Returns the absolute value of the tensor '''

        data = np.abs(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * np.sign(self.data))

        return Tensor(data, requires_grad, grad_fn)

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

        return Tensor(data, requires_grad, grad_fn)

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

        return Tensor(data, requires_grad, grad_fn)

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

        return Tensor(data, requires_grad, grad_fn)

    def sqrt(self):
        ''' Returns the square root of the tensor '''

        data = np.sqrt(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad / (2 * data))

        return Tensor(data, requires_grad, grad_fn)

    def log(self):
        ''' Returns the log of the tensor '''

        data = np.log(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad / self.data)

        return Tensor(data, requires_grad, grad_fn)

    def exp(self):
        ''' Returns the exponential of the tensor '''

        data = np.exp(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * data)

        return Tensor(data, requires_grad, grad_fn)

    def tanh(self):
        ''' Returns the hyperbolic tangent of the tensor '''

        data = np.tanh(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * (1 - data ** 2))

        return Tensor(data, requires_grad, grad_fn)

    def sin(self):
        ''' Returns the sine of the tensor '''

        data = np.sin(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * np.cos(self.data))

        return Tensor(data, requires_grad, grad_fn)

    def cos(self):
        ''' Returns the cosine of the tensor '''

        data = np.cos(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * -np.sin(self.data))

        return Tensor(data, requires_grad, grad_fn)

    def maximum(self, other: Tensorable):
        ''' Returns the max values of the tensor '''

        tensor = ensure_tensor(other)

        data = np.maximum(self.data, tensor.data)
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * (self.data >= tensor.data))
                tensor.backward(grad * (tensor.data >= self.data))

        return Tensor(data, requires_grad, grad_fn)

    def minimum(self, other: Tensorable):
        ''' Returns the min values of the tensor '''

        tensor = ensure_tensor(other)

        data = np.minimum(self.data, tensor.data)
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * (self.data <= tensor.data))
                tensor.backward(grad * (tensor.data <= self.data))

        return Tensor(data, requires_grad, grad_fn)

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

        return Tensor(data, requires_grad, grad_fn)

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

        return Tensor(data, requires_grad, grad_fn)

    def concatenate(self, *arrays: Tensorable, axis: SupportsIndex = 0):
        ''' Concatenates the tensors '''
        
        tensors = [self] + [ensure_tensor(t) for t in arrays]

        data = np.concatenate([t.data for t in tensors], axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Get the indices to split the gradient
                indices = np.cumsum([t.data.shape[axis] for t in tensors[:-1]])

                grads = np.split(grad, indices, axis=axis)

                for tensor, grad in zip(tensors, grads):
                    tensor.backward(grad)

        return Tensor(data, requires_grad, grad_fn)

    def reshape(self, shape: _ShapeLike):
        ''' Reshapes the tensor '''

        data = self.data.reshape(shape)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad.reshape(self.data.shape))

        return Tensor(data, requires_grad, grad_fn)

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
                
        return Tensor(data, requires_grad, grad_fn)

    def swapaxes(self, axis1: SupportsIndex, axis2: SupportsIndex):
        ''' Swaps the axes of the tensor '''
        
        data = self.data.swapaxes(axis1, axis2)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad.swapaxes(axis1, axis2))

        return Tensor(data, requires_grad, grad_fn)

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
                
        return Tensor(data, requires_grad, grad_fn)

    def __iter__(self):
        ''' Returns an iterator over the tensor '''
        
        for i in range(self.data.shape[0]):
            yield self[i]

    def __getitem__(self, key):
        ''' Gets called when using t[key] '''
        
        data = self.data[key]
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                grad_ = np.zeros(self.data.shape)
                grad_[key] = grad
                
                self.backward(grad_)
        
        return Tensor(data, requires_grad, grad_fn)

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

    def backward(self, grad: Optional[Arrayable] = None):
        if not self.requires_grad or self.grad is None:
            return

        # Initialize gradient if not provided
        if grad is None:
            grad = np.ones(self.data.shape)
        else:
            grad = ensure_array(grad)

        # Sum gradient to match data shape
        if self.data.shape != grad.shape:
            keepdims = self.data.ndim == grad.ndim

            if keepdims:
                self_shape = np.array(self.data.shape)
            else:
                self_shape = np.array((1,) * (grad.ndim - self.data.ndim) + self.data.shape)

            grad_shape = np.array(grad.shape)

            axis = tuple(np.where(self_shape != grad_shape)[0])

            grad = ensure_array(grad.sum(axis=axis, keepdims=keepdims))
            grad = grad.reshape(self.data.shape)

        # Accumulate gradient
        self.grad += grad

        if self.grad_fn is not None:
            self.grad_fn(grad)
