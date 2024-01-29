from typing import Union, Optional, Callable

import numpy as np

Arrayable = Union[int, float, tuple, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
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
    
    def __pos__(self):
        ''' Gets called when using +t '''
        
        data = self.data
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad)
        
        return Tensor(data, requires_grad, grad_fn)
    
    def __neg__(self):
        ''' Gets called when using -t '''
        
        data = -self.data
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(-grad)
        
        return Tensor(data, requires_grad, grad_fn)
    
    def __add__(self, other):
        ''' Gets called when using t + other '''
        
        other = ensure_tensor(other)
        
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad)
                other.backward(grad)
        
        return Tensor(data, requires_grad, grad_fn)

    def __radd__(self, other):
        ''' Gets called when using other + t '''
        
        return ensure_tensor(other).__add__(self)

    def __sub__(self, other):
        ''' Gets called when using t - other '''
        
        other = ensure_tensor(other)
        
        data = self.data - other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad)
                other.backward(-grad)
        
        return Tensor(data, requires_grad, grad_fn)

    def __rsub__(self, other):
        ''' Gets called when using other - t '''
        
        return ensure_tensor(other).__sub__(self)

    def __mul__(self, other):
        ''' Gets called when using t * other '''
        
        other = ensure_tensor(other)

        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * other.data)
                other.backward(grad * self.data)
            
        return Tensor(data, requires_grad, grad_fn)

    def __rmul__(self, other):
        ''' Gets called when using other * t '''
        
        return ensure_tensor(other).__mul__(self)

    def __div__(self, other):
        ''' Gets called when using t / other '''
        
        other = ensure_tensor(other)
        
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None
    
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad / other.data)
                other.backward(-grad * self.data / other.data ** 2)
        
        return Tensor(data, requires_grad, grad_fn)
    
    def __rdiv__(self, other):
        ''' Gets called when using other / t '''
        
        return ensure_tensor(other).__div__(self)  

    def __pow__(self, other):
        ''' Gets called when using t ** other '''
        
        other = ensure_tensor(other)
        
        data = self.data ** other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                self.backward(grad * other.data * self.data ** (other.data - 1))
                other.backward(grad * np.log(self.data) * self.data ** other.data)
            
        return Tensor(data, requires_grad, grad_fn)

    def __rpow__(self, other):
        ''' Gets called when using other ** t '''
        
        return ensure_tensor(other).__pow__(self)

    def __matmul__(self, other):
        ''' Gets called when using t @ other '''
        
        other = ensure_tensor(other)
        data = self.data @ other.data
        grad_fn = None
        
        requires_grad = self.requires_grad or other.requires_grad

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Matrix @ Matrix
                if self.data.ndim > 1 and other.data.ndim > 1:
                    self.backward(grad @ other.data.T)
                    other.backward(self.data.T @ grad)
            
                # Vector @ Vector
                elif self.data.ndim == 1 and other.data.ndim == 1:
                    self.backward(grad * other.data)
                    other.backward(grad * self.data)

                # Matrix @ Vector 
                elif self.data.ndim > 1 and other.data.ndim == 1:
                    self.backward(np.outer(grad, other.data))
                    other.backward(self.data.T @ grad)
                    
                # Vector @ Matrix
                elif self.data.ndim == 1 and other.data.ndim > 1:
                    self.backward(grad @ other.data.T)
                    other.backward(np.outer(self.data, grad))

        return Tensor(data, requires_grad, grad_fn)

    def __rmatmul__(self, other):
        ''' Gets called when using other @ t '''
        
        return ensure_tensor(other).__matmul__(self)

    def sum(self, *args, **kwargs):
        ''' Returns the sum of the tensor '''
        
        axis = args[0] if args else kwargs.get('axis', None)
        
        data = self.data.sum(*args, **kwargs)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)
                
                self.backward(grad * np.ones(self.data.shape))
                
        return Tensor(data, requires_grad, grad_fn)

    def mean(self, *args, **kwargs):
        ''' Returns the mean of the tensor '''
        
        axis = args[0] if args else kwargs.get('axis', None)
        
        data = self.data.mean(*args, **kwargs)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)
                
                # Compute size of the mean
                axis_list = list(axis) if isinstance(axis, tuple) else axis
                size = np.prod(np.array(self.data.shape)[axis_list])
                
                self.backward(grad * np.ones(self.data.shape) / size)

        return Tensor(data, requires_grad, grad_fn)

    def var(self, *args, **kwargs):
        ''' Returns the variance of the tensor '''
        
        axis = args[0] if args else kwargs.get('axis', None)
        
        data = self.data.var(*args, **kwargs)
        requires_grad = self.requires_grad
        grad_fn = None
        
        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Expand gradient to match data shape
                if self.data.ndim != grad.ndim and axis is not None:
                    grad = np.expand_dims(grad, axis)
                
                # Compute size of the var
                axis_list = list(axis) if isinstance(axis, tuple) else axis
                size = np.prod(np.array(self.data.shape)[axis_list])
                
                # Compute the mean
                mean = self.data.mean(axis=axis, keepdims=True)
                
                # Compute the centered data
                xmu = self.data - mean
                
                self.backward(grad * np.ones(self.data.shape) * 2 * xmu / size)
        
        return Tensor(data, requires_grad, grad_fn)

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
            
y = Tensor([3, 4], requires_grad=True)
p = Tensor([2, 3.5], requires_grad=True)

mse = (0.5 * (y - p) ** 2).mean()

mse.backward()

print('MSE', p.grad)