from typing import Union, Optional, Callable, SupportsIndex
from numpy._typing import _ShapeLike, ArrayLike, DTypeLike

import numpy as np
np.seterr(all='ignore')

class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        dtype: DTypeLike = None,
        requires_grad = False,
        grad_fn: Optional[Callable[['Tensor'], list[tuple['Tensor', Optional['Tensor']]]]] = None
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

    # Non gradient operations

    def invert(self):
        ''' Gets the inverse value of the tensor '''

        return Tensor(~self.data)

    def greater(self, other):
        ''' Gets the truth value of self > other '''
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data > other.data)
    
    def greater_equal(self, other):
        ''' Gets the truth value of self >= other '''
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data >= other.data)
    
    def less(self, other):
        ''' Gets the truth value of self < other '''
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data < other.data)
    
    def less_equal(self, other):
        ''' Gets the truth value of self <= other '''
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data <= other.data)
    
    def equal(self, other):
        ''' Returns the truth value of self == other '''
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data == other.data)
    
    def not_equal(self, other):
        ''' Gets the truth value of self != other '''
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        return Tensor(self.data != other.data)

    # End of non gradient operations

    def sign(self):
        ''' Returns the sign of the tensor '''
        
        data = np.sign(self.data)
        requires_grad = self.requires_grad
        sign_backward = None
        
        if requires_grad:
            def sign_backward(grad: Tensor):
                self_grad = grad * np.zeros(self.shape)
                
                return [ (self, self_grad) ]
        
        return Tensor(data, None, requires_grad, sign_backward)

    def abs(self):
        ''' Returns the absolute value of the tensor '''

        data = np.abs(self.data)
        requires_grad = self.requires_grad
        abs_backward = None

        if requires_grad:
            def abs_backward(grad: Tensor):
                self_grad = grad * self.sign()
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, abs_backward)

    def positive(self):
        ''' Gets the positive value of the tensor '''

        data = self.data
        requires_grad = self.requires_grad
        pos_backward = None

        if requires_grad:
            def pos_backward(grad: Tensor):
                self_grad = grad
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, pos_backward)

    def negative(self):
        ''' Gets the negative value of the tensor '''

        data = -self.data
        requires_grad = self.requires_grad
        neg_backward = None  

        if requires_grad:
            def neg_backward(grad: Tensor):
                self_grad = -grad
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, neg_backward)

    def add(self, other):
        ''' Gets the sum of the tensor and another tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        add_backward = None

        if requires_grad:
            def add_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad
                
                if other.requires_grad:
                    other_grad = grad
                
                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, add_backward)

    def sub(self, other):
        ''' Gets the difference of the tensor and another tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data - other.data
        requires_grad = self.requires_grad or other.requires_grad
        sub_backward = None

        if requires_grad:
            def sub_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad
                
                if other.requires_grad:
                    other_grad = -grad
                    
                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, sub_backward)

    def mul(self, other):
        ''' Gets the product of the tensor and another tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        mul_backward = None

        if requires_grad:
            def mul_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad * other
                    
                if other.requires_grad:
                    other_grad = grad * self
                    
                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, mul_backward)

    def div(self, other):
        ''' Gets the division of the tensor and another tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        div_backward = None

        if requires_grad:
            def div_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad / other
                    
                if other.requires_grad:
                    other_grad = -grad * self / other ** 2

                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, div_backward)

    def matmul(self, other):
        ''' Gets the matrix product of the tensor and another tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        matmul_backward = None

        if requires_grad:
            def matmul_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    # Matrix @ Matrix or Vector @ Matrix
                    if (self.ndim > 1 and other.ndim > 1 or self.ndim == 1 and other.ndim > 1):
                        self_grad = grad @ other.swapaxes(-1, -2)

                    elif self.ndim == 1 and other.ndim == 1: # Vector @ Vector
                        self_grad = grad * other

                    elif self.ndim > 1 and other.ndim == 1: # Matrix @ Vector
                        self_grad = grad.outer(other)
                
                if other.requires_grad:
                    # Matrix @ Matrix or Matrix @ Vector
                    if (self.ndim > 1 and other.ndim > 1 or self.ndim > 1 and other.ndim == 1):
                        other_grad = self.swapaxes(-1, -2) @ grad

                    elif self.ndim == 1 and other.ndim == 1: # Vector @ Vector
                        other_grad = grad * self

                    elif self.ndim == 1 and other.ndim > 1: # Vector @ Matrix
                        other_grad = self.outer(grad)

                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, matmul_backward)

    def outer(self, other):
        ''' Returns the outer product of the tensor and another tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = np.outer(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        outer_backward = None

        if requires_grad:
            def outer_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = (grad @ other.reshape(-1)).reshape(self.shape)
                    
                if other.requires_grad:
                    other_grad = (self @ grad.reshape(-1)).reshape(other.shape)
                
                return [ (self, self_grad), (other, other_grad) ]
                
        return Tensor(data, None, requires_grad, outer_backward)

    def power(self, other):
        ''' Gets the power of the tensor and another tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = self.data ** other.data
        requires_grad = self.requires_grad or other.requires_grad
        pow_backward = None

        if requires_grad:
            def pow_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad * other * self ** (other - 1)
                    
                if other.requires_grad:
                    other_grad = grad * self.log() * data

                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, pow_backward)

    def maximum(self, other):
        ''' Returns the max values of the tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = np.maximum(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        maximum_backward = None

        if requires_grad:
            def maximum_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad * (self > other)
                    self_grad = grad * 0.5 * (self == other)
                    
                if other.requires_grad:
                    other_grad = grad * (other > self)
                    other_grad = grad * 0.5 * (other == self)
                    
                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, maximum_backward)

    def minimum(self, other):
        ''' Returns the min values of the tensor '''

        other = other if isinstance(other, Tensor) else Tensor(other)

        data = np.minimum(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        minimum_backward = None

        if requires_grad:
            def minimum_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad * (self < other)
                    self_grad = grad * 0.5 * (self == other)
                    
                if other.requires_grad:
                    other_grad = grad * (other < self)
                    other_grad = grad * 0.5 * (other == self)

                return [ (self, self_grad), (other, other_grad) ]

        return Tensor(data, None, requires_grad, minimum_backward)

    def sqrt(self):
        ''' Returns the square root of the tensor '''

        data = np.sqrt(self.data)
        requires_grad = self.requires_grad
        sqrt_backward = None
        
        if requires_grad:
            def sqrt_backward(grad: Tensor):
                self_grad = grad / (2 * self.sqrt())
                
                return [ (self, self_grad) ]
            
        return Tensor(data, None, requires_grad, sqrt_backward)

    def log(self):
        ''' Returns the log of the tensor '''

        data = np.log(self.data)
        requires_grad = self.requires_grad
        log_backward = None

        if requires_grad:
            def log_backward(grad: Tensor):
                self_grad = grad / self
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, log_backward)

    def exp(self):
        ''' Returns the exponential of the tensor '''

        data = np.exp(self.data)
        requires_grad = self.requires_grad
        exp_backward = None

        if requires_grad:            
            def exp_backward(grad: Tensor):
                self_grad = grad * self.exp()
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, exp_backward)

    def tanh(self):
        ''' Returns the hyperbolic tangent of the tensor '''

        data = np.tanh(self.data)
        requires_grad = self.requires_grad
        tanh_backward = None

        if requires_grad:
            def tanh_backward(grad: Tensor):
                self_grad = grad * (1 - self.tanh() ** 2)
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, tanh_backward)

    def sin(self):
        ''' Returns the sine of the tensor '''

        data = np.sin(self.data)
        requires_grad = self.requires_grad
        sin_backward = None

        if requires_grad:
            def sin_backward(grad: Tensor):
                self_grad = grad * self.cos()
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, sin_backward)

    def cos(self):
        ''' Returns the cosine of the tensor '''

        data = np.cos(self.data)
        requires_grad = self.requires_grad
        cos_backward = None

        if requires_grad:
            def cos_backward(grad: Tensor):
                self_grad = grad * -self.sin()
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, cos_backward)

    # Batch operations

    def sum(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the sum of the tensor '''

        data = self.data.sum(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        sum_backward = None

        if requires_grad:
            def sum_backward(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.unsqueeze(axis)

                self_grad = grad * np.ones(self.shape)

                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, sum_backward)

    def mean(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the mean of the tensor '''

        data = self.data.mean(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        mean_backward = None

        if requires_grad:
            def mean_backward(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.unsqueeze(axis)

                # Compute size of the mean
                axis_ = list(axis) if isinstance(axis, tuple) else axis
                size = np.array(self.shape)[axis_].prod() # type: ignore
                
                self_grad = grad * np.ones(self.data.shape) / size
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, mean_backward)

    def var(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the variance of the tensor '''

        data = self.data.var(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        var_backward = None

        if requires_grad:
            def var_backward(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.unsqueeze(axis)

                # Compute size of the variance
                axis_ = list(axis) if isinstance(axis, tuple) else axis
                size = np.array(self.shape)[axis_].prod() # type: ignore

                # Compute mean
                mean = self.mean(axis=axis, keepdims=True)

                self_grad = grad * np.ones(self.data.shape) * 2 * (self - mean) / size
                
                return [ (self, self_grad) ]
                
        return Tensor(data, None, requires_grad, var_backward)

    def max(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the biggest value of the tensor '''

        data = self.data.max(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        max_backward = None

        if requires_grad:
            def max_backward(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.unsqueeze(axis)

                self_grad = grad * (self == self.max(axis=axis, keepdims=True))
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, max_backward)

    def min(self, axis: Optional[_ShapeLike] = None, keepdims: bool = False):
        ''' Returns the small value of the tensor '''

        data = self.data.min(axis=axis, keepdims=keepdims)
        requires_grad = self.requires_grad
        min_backward = None

        if requires_grad:
            def min_backward(grad: Tensor):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and axis is not None:
                    grad = grad.unsqueeze(axis)

                self_grad = grad * (self == self.min(axis=axis, keepdims=True))

                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, min_backward)

    # End of batch operations

    # Other operations

    def split(self, indices_or_sections: _ShapeLike, axis: SupportsIndex = 0):
        ''' Splits the tensor '''
        
        data = np.split(self.data, indices_or_sections, axis=axis)
        requires_grad = self.requires_grad
        
        tensors: list[Tensor] = []
        
        for i, item in enumerate(data):
            split_backward = None
            
            if requires_grad:
                def split_backward(grad: Tensor, i=i):
                    previous_ = data[:i]
                    next_ = data[i+1:]
                    
                    if len(previous_) != 0:
                        previous_ = np.zeros_like(np.concatenate(previous_, axis=axis))
                        
                    if len(next_) != 0:
                        next_ = np.zeros_like(np.concatenate(next_, axis=axis))
                    
                    self_grad = Tensor(previous_).concatenate([grad, next_], axis=axis)
                    
                    return [ (self, self_grad) ]
            
            tensors.append(Tensor(item, None, requires_grad, split_backward))

        return tensors

    @staticmethod
    def concatenate(arrays: list[Union['Tensor', ArrayLike]], axis: SupportsIndex = 0):
        ''' Concatenates the tensors '''
        
        tensors: list[Tensor] = []
        
        for item in arrays:
            tensor = item if isinstance(item, Tensor) else Tensor(item)
            
            tensors.append(tensor)

        data = np.concatenate([t.data for t in tensors], axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        concat_backward = None

        if requires_grad:
            def concat_backward(grad: Tensor):
                # Get the indices to split the gradient
                indices = np.cumsum([tensor.shape[axis] for tensor in tensors[:-1]])

                grads = grad.split(indices, axis=axis)
                
                all_grads: list[tuple[Tensor, Optional[Tensor]]] = []

                for tensor, grad in zip(tensors, grads):
                    if tensor.requires_grad:
                        all_grads.append((tensor, grad))
                    else:
                        all_grads.append((tensor, None))

                return all_grads

        return Tensor(data, None, requires_grad, concat_backward)
    
    def where(self, condition, other):
        ''' Returns elements chosen from self or other depending on condition '''
        
        condition = condition if isinstance(condition, Tensor) else Tensor(condition)
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        data = np.where(condition.data, self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        where_backward = None
        
        if requires_grad:
            def where_backward(grad: Tensor):
                self_grad = None
                other_grad = None
                
                if self.requires_grad:
                    self_grad = grad * condition
                    
                if other.requires_grad:
                    other_grad = grad * ~condition
        
                return [ (self, self_grad), (other, other_grad) ]
        
        return Tensor(data, None, requires_grad, where_backward)

    def getitem(self, key):
        ''' Gets the item of the tensor at the specified key '''
    
        data = self.data[key]
        requires_grad = self.requires_grad
        select_backward = None
        
        if requires_grad:
            def select_backward(grad: Tensor):
                mask = np.zeros(self.shape)
                mask[key] = grad.data

                self_grad = Tensor(mask)
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, select_backward)  
         
    def iter(self):
        ''' Returns an iterator over the tensor '''
        
        return iter(self[i] for i in range(self.shape[0]))

    # End of other operations

    # Shape operations

    def reshape(self, shape: _ShapeLike):
        ''' Reshapes the tensor '''

        data = self.data.reshape(shape)
        requires_grad = self.requires_grad
        reshape_backward = None

        if requires_grad:
            def reshape_backward(grad: Tensor):
                self_grad = grad.reshape(self.shape)
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, reshape_backward)

    def transpose(self, axes: Optional[_ShapeLike] = None):
        ''' Transposes the tensor '''
        
        if axes is None:
            axes = tuple(reversed(range(self.data.ndim)))
        
        data = self.data.transpose(axes)
        requires_grad = self.requires_grad
        transpose_backward = None
        
        if requires_grad:
            def transpose_backward(grad: Tensor):
                self_grad = grad.transpose(axes)
                
                return [ (self, self_grad) ]
                
        return Tensor(data, None, requires_grad, transpose_backward)

    def swapaxes(self, axis1: SupportsIndex, axis2: SupportsIndex):
        ''' Swaps the axes of the tensor '''
        
        data = self.data.swapaxes(axis1, axis2)
        requires_grad = self.requires_grad
        swapaxes_backward = None
        
        if requires_grad:
            def swapaxes_backward(grad: Tensor):
                self_grad = grad.swapaxes(axis1, axis2)
                
                return [ (self, self_grad) ]

        return Tensor(data, None, requires_grad, swapaxes_backward)

    def flip(self, axis: Optional[_ShapeLike] = None):
        ''' Flips the tensor '''
        
        if axis is None:
            axis = tuple(range(self.data.ndim))
        
        data = np.flip(self.data, axis=axis)
        requires_grad = self.requires_grad
        flip_backward = None
        
        if requires_grad:
            def flip_backward(grad: Tensor):
                self_grad = grad.flip(axis=axis)
                
                return [ (self, self_grad) ]
                
        return Tensor(data, None, requires_grad, flip_backward)
    
    def unsqueeze(self, axis: _ShapeLike):
        ''' Expands the dimensions of the tensor '''
        
        data = np.expand_dims(self.data, axis)
        requires_grad = self.requires_grad
        unsqueeze_backward = None
        
        if requires_grad:
            def unsqueeze_backward(grad: Tensor):
                self_grad = grad.squeeze(axis)
                
                return [ (self, self_grad) ]
                
        return Tensor(data, None, requires_grad, unsqueeze_backward)

    def squeeze(self, axis: Optional[_ShapeLike] = None):
        ''' Squeezes the tensor '''
        
        data = np.squeeze(self.data, axis)
        requires_grad = self.requires_grad
        squeeze_backward = None
        
        if requires_grad:
            def squeeze_backward(grad: Tensor):
                self_grad = grad
                
                if axis is None:
                    self_grad = grad.reshape(self.shape)
                else:
                    self_grad = grad.unsqueeze(axis)
                
                return [ (self, self_grad) ]
                
        return Tensor(data, None, requires_grad, squeeze_backward)
    
    # End of shape operations
      
    # Magic methods

    def __invert__(self):
        return self.invert()

    def __gt__(self, other):
        return self.greater(other)
    
    def __ge__(self, other):
        return self.greater_equal(other)
    
    def __lt__(self, other):
        return self.less(other)
    
    def __le__(self, other):
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

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other.add(self)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other.sub(self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other.mul(self)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other.div(self)

    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other.matmul(self)

    def __pow__(self, other):
        return self.power(other)

    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other.power(self)

    def __getitem__(self, key):
        return self.getitem(key)
        
    def __iter__(self):
        return self.iter()

    def __array__(self):
        ''' Returns the tensor as a numpy array '''
        
        return self.data

    # End of magic methods

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
    def dtype(self):
        ''' Returns the data type of the tensor '''
        
        return self.data.dtype
    
    @property
    def T(self):
        ''' Returns the transpose of the tensor '''
        
        return self.transpose()

    # End of properties

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
            self.grad = grad
        else:
            # Accumulate gradient
            self.grad += grad

        if self.grad_fn is not None:
            grads = self.grad_fn(grad) # type: ignore
            
            for tensor, grad in grads:
                if grad is None:
                    continue
                
                tensor.backward(grad)