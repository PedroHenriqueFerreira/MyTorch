from typing import Union, Optional, Callable, SupportsIndex
from numpy._typing import _ShapeLike, ArrayLike, DTypeLike

import numpy as np
np.seterr(all='ignore')

Tensorable = Union['Tensor', ArrayLike]

def ensure_tensor(tensor: Tensorable) -> 'Tensor':
    ''' Ensures the input is a tensor '''
    
    return tensor if isinstance(tensor, Tensor) else Tensor(tensor)

class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike = None,
        requires_grad = False,
        grad_fn: Optional[Callable[['Tensor'], None]] = None
    ):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn

        self.grad: Optional[Tensor] = None

    def __repr__(self):
        ''' Returns a string representation of the tensor '''
        
        if self.grad_fn:
            return f'tensor({self.data}, grad_fn=<{self.grad_fn.__name__}>)'
        elif self.requires_grad:
            return f'tensor({self.data}, requires_grad=True)'
        else:
            return f'tensor({self.data})'
    
    def detach(self):
        ''' Detaches the tensor from the computation graph '''
        
        return Tensor(self.data, self.dtype, requires_grad=False)

    def invert(self):
        ''' Gets the inverse value of the tensor '''

        return Tensor(~self.data)

    def __invert__(self):
        ''' Gets called when using ~t '''

        return self.invert()

    def pos(self):
        ''' Gets the positive value of the tensor '''

        data = self.data
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __pos__(self):
        ''' Gets called when using +t '''

        return self.pos()

    def neg(self):
        ''' Gets the negative value of the tensor '''

        data = -self.data
        requires_grad = self.requires_grad
        grad_fn = None  

        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(-grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __neg__(self):
        ''' Gets called when using -t '''

        return self.neg()

    def add(self, other):
        ''' Gets the sum of the tensor and another tensor '''

        other = ensure_tensor(other)

        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad)
                
                if other.requires_grad:
                    other.backward(grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __add__(self, other):
        ''' Gets called when using t + other '''

        return self.add(other)

    def __radd__(self, other):
        ''' Gets called when using other + t '''

        return ensure_tensor(other).add(self)

    def sub(self, other):
        ''' Gets the difference of the tensor and another tensor '''

        other = ensure_tensor(other)

        data = self.data - other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad)
                
                if other.requires_grad:
                    other.backward(-grad)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __sub__(self, other):
        ''' Gets called when using t - other '''

        return self.sub(other)

    def __rsub__(self, other):
        ''' Gets called when using other - t '''

        return ensure_tensor(other).sub(self)

    def mul(self, other):
        ''' Gets the product of the tensor and another tensor '''

        other = ensure_tensor(other)

        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad * other)
                    
                if other.requires_grad:
                    other.backward(grad * self)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __mul__(self, other):
        ''' Gets called when using t * other '''

        return self.mul(other)

    def __rmul__(self, other):
        ''' Gets called when using other * t '''

        return ensure_tensor(other).mul(self)

    def div(self, other):
        ''' Gets the division of the tensor and another tensor '''

        other = ensure_tensor(other)

        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad / other)
                    
                if other.requires_grad:
                    other.backward(-grad * self / other ** 2)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __truediv__(self, other):
        ''' Gets called when using t / other '''

        return self.div(other)

    def __rtruediv__(self, other):
        ''' Gets called when using other / t '''
    
        return ensure_tensor(other).div(self)

    def matmul(self, other):
        ''' Gets the matrix product of the tensor and another tensor '''

        other = ensure_tensor(other)
        
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    # Matrix @ Matrix or Vector @ Matrix
                    if (self.ndim > 1 and other.ndim > 1 or self.ndim == 1 and other.ndim > 1):
                        self.backward(grad @ other.swapaxes(-1, -2))

                    # Vector @ Vector
                    elif self.ndim == 1 and other.ndim == 1:
                        self.backward(grad * other)

                    # Matrix @ Vector
                    elif self.ndim > 1 and other.ndim == 1:
                        self.backward(grad.outer(other))
                
                if other.requires_grad:
                    # Matrix @ Matrix or Matrix @ Vector
                    if (self.ndim > 1 and other.ndim > 1 or self.ndim > 1 and other.ndim == 1):
                        other.backward(self.swapaxes(-1, -2) @ grad)

                    # Vector @ Vector
                    elif self.ndim == 1 and other.ndim == 1:
                        other.backward(grad * self)

                    # Vector @ Matrix
                    elif self.ndim == 1 and other.ndim > 1:
                        other.backward(self.outer(grad))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __matmul__(self, other):
        ''' Gets called when using t @ other '''

        return self.matmul(other)

    def __rmatmul__(self, other):
        ''' Gets called when using other @ t '''

        return ensure_tensor(other).matmul(self)

    def power(self, other):
        ''' Gets the power of the tensor and another tensor '''

        other = ensure_tensor(other)

        data = self.data ** other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad * other * self ** (other - 1))
                    
                if other.requires_grad:
                    other.backward(grad * self.log() * data)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __pow__(self, other):
        ''' Gets called when using t ** other '''

        return self.power(other)

    def __rpow__(self, other):
        ''' Gets called when using other ** t '''

        return ensure_tensor(other).power(self)

    def abs(self):
        ''' Returns the absolute value of the tensor '''

        data = np.abs(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * self.sign())

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __abs__(self):
        ''' Gets called when using abs(t) '''
        
        return self.abs()

    def outer(self, other):
        ''' Returns the outer product of the tensor and another tensor '''

        other = ensure_tensor(other)

        data = np.outer(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward((grad @ other).reshape(self.shape))
                    
                if other.requires_grad:
                    other.backward((self @ grad).reshape(other.shape))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def sqrt(self):
        ''' Returns the square root of the tensor '''

        data = np.sqrt(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad / (2 * data))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def log(self):
        ''' Returns the log of the tensor '''

        data = np.log(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad / self)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def exp(self):
        ''' Returns the exponential of the tensor '''

        data = np.exp(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:            
            def grad_fn(grad: Tensor):
                self.backward(grad * data)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def tanh(self):
        ''' Returns the hyperbolic tangent of the tensor '''

        data = np.tanh(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad * (1 - data ** 2))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def sin(self):
        ''' Returns the sine of the tensor '''

        data = np.sin(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad * self.cos())

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def cos(self):
        ''' Returns the cosine of the tensor '''

        data = np.cos(self.data)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad * -self.sin())

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def expand_dims(self, axis: _ShapeLike):
        ''' Expands the dimensions of the tensor '''
        
        data = np.expand_dims(self.data, axis)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad.squeeze(axis))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def squeeze(self, axis: Optional[_ShapeLike] = None):
        ''' Squeezes the tensor '''
        
        data = np.squeeze(self.data, axis)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: Tensor):
                if axis is None:
                    self.backward(grad.reshape(self.shape))
                else:
                    self.backward(grad.expand_dims(axis))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def sum(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the sum of the tensor '''

        data = self.data.sum(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.expand_dims(axis)

                self.backward(grad * np.ones(self.shape))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def mean(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the mean of the tensor '''

        data = self.data.mean(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.expand_dims(axis)

                # Compute size of the mean
                axis_ = list(axis) if isinstance(axis, tuple) else axis
                size = np.array(self.shape)[axis_].prod() # type: ignore
                
                self.backward(grad * np.ones(self.data.shape) / size)

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def var(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the variance of the tensor '''

        data = self.data.var(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.expand_dims(axis)

                # Compute size of the variance
                axis_ = list(axis) if isinstance(axis, tuple) else axis
                size = np.array(self.shape)[axis_].prod() # type: ignore

                # Compute mean
                mean = self.mean(axis=axis, keepdims=True)

                self.backward(grad * np.ones(self.data.shape) * 2 * (self - mean) / size)
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def maximum(self, other: Tensorable):
        ''' Returns the max values of the tensor '''

        tensor = ensure_tensor(other)

        data = np.maximum(self.data, tensor.data)
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad * (self > tensor))
                    self.backward(grad * 0.5 * (self == tensor))
                    
                if tensor.requires_grad:
                    tensor.backward(grad * (tensor > self))
                    tensor.backward(grad * 0.5 * (tensor == self))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def minimum(self, other: Tensorable):
        ''' Returns the min values of the tensor '''

        tensor = ensure_tensor(other)

        data = np.minimum(self.data, tensor.data)
        requires_grad = self.requires_grad or tensor.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad * (self < tensor))
                    self.backward(grad * 0.5 * (self == tensor))
                    
                if tensor.requires_grad:
                    tensor.backward(grad * (tensor < self))
                    tensor.backward(grad * 0.5 * (tensor == self))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def max(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the biggest value of the tensor '''

        data = self.data.max(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.expand_dims(axis)

                self.backward(grad * (self == self.max(axis=axis, keepdims=True)))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def min(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the small value of the tensor '''

        data = self.data.min(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.expand_dims(axis)

                self.backward(grad * (self == self.min(axis=axis, keepdims=True)))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def split(self, indices_or_sections: _ShapeLike, axis: SupportsIndex = 0):
        ''' Splits the tensor '''
        
        data = np.split(self.data, indices_or_sections, axis=axis)
        requires_grad = self.requires_grad
        
        tensors: list[Tensor] = []
        
        for i, item in enumerate(data):
            grad_fn = None
            
            if requires_grad:
                def grad_fn(grad: Tensor, i=i):
                    previous_ = data[:i]
                    next_ = data[i+1:]
                    
                    if len(previous_) != 0:
                        previous_ = np.zeros_like(np.concatenate(previous_, axis=axis))
                        
                    if len(next_) != 0:
                        next_ = np.zeros_like(np.concatenate(next_, axis=axis))
                    
                    previous_ = ensure_tensor(previous_) # type: ignore
                    next_ = ensure_tensor(next_) # type: ignore
                    
                    self.backward(previous_.concatenate([grad, next_], axis=axis)) # type: ignore
            
            tensors.append(Tensor(item, requires_grad=requires_grad, grad_fn=grad_fn))

        return tensors

    def concatenate(self, arrays: list[Tensorable], axis: SupportsIndex = 0):
        ''' Concatenates the tensors '''
        
        tensors = [self] + [ensure_tensor(tensor) for tensor in arrays]

        data = np.concatenate([tensor.data for tensor in tensors], axis=axis)
        requires_grad = any(tensor.requires_grad for tensor in tensors)
        grad_fn = None

        if requires_grad:
            def grad_fn(grad: Tensor):
                # Get the indices to split the gradient
                indices = np.cumsum([t.shape[axis] for t in tensors[:-1]])

                grads = grad.split(indices, axis=axis)

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
            def grad_fn(grad: Tensor):
                self.backward(grad.reshape(self.shape))

        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def transpose(self, axes: Optional[_ShapeLike] = None):
        ''' Transposes the tensor '''
        
        if axes is None:
            axes = tuple(reversed(range(self.data.ndim)))
        
        data = self.data.transpose(axes)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad.transpose(axes))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def swapaxes(self, axis1: SupportsIndex, axis2: SupportsIndex):
        ''' Swaps the axes of the tensor '''
        
        data = self.data.swapaxes(axis1, axis2)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: Tensor):
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
            def grad_fn(grad: Tensor):
                self.backward(grad.flip(axis=axis))
                
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
    def where(self, condition, other):
        ''' Returns elements chosen from self or other depending on condition '''
        
        condition = ensure_tensor(condition)
        other = ensure_tensor(other)
        
        data = np.where(condition.data, self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: Tensor):
                if self.requires_grad:
                    self.backward(grad * condition)
                    
                if other.requires_grad:
                    other.backward(grad * ~condition)
        
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
      
    def argmax(self, axis: Optional[SupportsIndex] = None, keepdims: bool = False):
        ''' Returns the indices of the maximum values along an axis '''
    
        return Tensor(self.data.argmax(axis=axis, keepdims=keepdims))
      
    def argmin(self, axis: Optional[SupportsIndex] = None, keepdims: bool = False):
        ''' Returns the indices of the minimum values along an axis '''
      
        return Tensor(self.data.argmin(axis=axis, keepdims=keepdims))
      
    def getitem(self, key):
        ''' Gets called when using t[key] '''
        
        if isinstance(key, Tensor):
            key = key.data
        
        elif isinstance(key, (list, tuple)):
            key_ = []
            
            for item in key:
                if isinstance(item, Tensor):
                    key_.append(item.data)
                else:
                    key_.append(item)
                   
            key = tuple(key_) if isinstance(key, tuple) else key_
            
        data = self.data[key]
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: Tensor):
                grad_ = np.zeros(self.shape)
                grad_[key] = grad
                
                self.backward(ensure_tensor(grad_))
        
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)

    def __getitem__(self, key):
        ''' Gets called when using t[key] '''
        
        return self.getitem(key)  
         
    def iter(self):
        ''' Returns an iterator over the tensor '''
        
        for i in range(self.data.shape[0]):
            yield self.getitem(i)
        
    def __iter__(self):
        ''' Gets called when using iter(t) '''
        
        return self.iter()
    
    def sign(self):
        ''' Returns the sign of the tensor '''
        
        data = np.sign(self.data)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: Tensor):
                self.backward(grad * np.zeros(self.shape))
        
        return Tensor(data, requires_grad=requires_grad, grad_fn=grad_fn)
    
    def gt(self, other: Tensorable):
        ''' Gets the truth value of self > other '''
        
        return Tensor(self.data > ensure_tensor(other).data)
    
    def __gt__(self, other: Tensorable):
        ''' Gets called when using t > other '''
        
        return self.gt(other)
    
    def ge(self, other: Tensorable):
        ''' Gets the truth value of self >= other '''
        
        return Tensor(self.data >= ensure_tensor(other).data)
    
    def __ge__(self, other: Tensorable):
        ''' Gets called when using t >= other '''
        
        return self.ge(other)
    
    def lt(self, other: Tensorable):
        ''' Gets the truth value of self < other '''
        
        return Tensor(self.data < ensure_tensor(other).data)
    
    def __lt__(self, other: Tensorable):
        ''' Gets called when using t < other '''
        
        return self.lt(other)
    
    def le(self, other: Tensorable):
        ''' Gets the truth value of self <= other '''
        
        return Tensor(self.data <= ensure_tensor(other).data)
    
    def __le__(self, other: Tensorable):
        ''' Gets called when using t <= other '''
        
        return self.le(other)
    
    def eq(self, other: Tensorable):
        ''' Returns the truth value of self == other '''
        
        return Tensor(self.data == ensure_tensor(other).data)
    
    def __eq__(self, other):
        ''' Gets called when using t == other '''
        
        return self.eq(other)
    
    def ne(self, other: Tensorable):
        ''' Gets the truth value of self != other '''
        
        return Tensor(self.data != ensure_tensor(other).data)
    
    def __ne__(self, other):
        ''' Gets called when using t != other '''
        
        return self.ne(other)

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
    def dtype(self):
        ''' Returns the data type of the tensor '''
        
        return self.data.dtype
    
    @property
    def T(self):
        ''' Returns the transpose of the tensor '''
        
        return self.transpose()

    def backward(self, grad: Optional['Tensor'] = None):
        ''' Backpropagates the gradient through the computation graph '''
        
        if not self.requires_grad:
            raise RuntimeError('Cannot compute gradient on tensor that does not require grad')

        # Initialize gradient if not provided
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0, dtype=self.dtype)
            else:
                raise RuntimeError('Gradient must be provided for non-scalar tensor')

        # Sum gradient to match data shape
        if self.shape != grad.shape:
            keepdims = self.ndim == grad.ndim

            if keepdims:
                self_shape = np.array(self.shape)
            else:
                self_shape = np.array((1,) * (grad.ndim - self.ndim) + self.shape)

            grad_shape = np.array(grad.shape)

            axis = tuple(np.where(self_shape != grad_shape)[0])

            grad = grad.sum(axis=axis, keepdims=keepdims).reshape(self.shape)

        if self.grad is None:
            # Initialize gradient
            self.grad = grad # type: ignore
        else:
            # Accumulate gradient
            self.grad += grad

        if self.grad_fn is not None:
            self.grad_fn(grad) # type: ignore
