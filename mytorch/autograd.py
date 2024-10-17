from typing import Union, Optional, Callable, SupportsIndex, Sequence, Literal, Any
from numpy._typing import _ShapeLike, _ArrayLikeInt, ArrayLike, DTypeLike

import numpy as np
import cupy as cp

np.seterr(all='ignore')

ShapeLike = _ShapeLike
ArrayLikeInt = _ArrayLikeInt

NDArray = Union[np.ndarray, cp.ndarray]
TensorLikeType = Union['Tensor', ArrayLike]
DeviceLikeType = Literal['cpu', 'cuda']

class Tensor:
    def __init__(
        self,
        data: TensorLikeType,
        dtype: DTypeLike = None,
        requires_grad = False,
        grad_fn: Optional[Callable[[NDArray], None]] = None,
        device: DeviceLikeType  = 'cpu'
    ):
        match device:
            case 'cpu':
                self.lib = np
            case 'cuda':
                self.lib = cp
            case _:
                raise ValueError('Invalid device')
        
        if isinstance(data, Tensor):
            self.data: NDArray = self.lib.array(data.data, dtype=dtype)
        else:
            self.data: NDArray = self.lib.array(data, dtype=dtype)
        
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.device = device

        self.grad: Optional[NDArray] = None

    def __repr__(self):
        if self.grad_fn:
            return f'tensor({self.data.round(4)}, dtype={self.dtype}, grad_fn=<{self.grad_fn.__name__}>, device="{self.device}")'
        elif self.requires_grad:
            return f'tensor({self.data.round(4)}, dtype={self.dtype}, requires_grad=True, device="{self.device}")'
        else:
            return f'tensor({self.data.round(4)}, dtype={self.dtype}, device="{self.device}")'
    
    def ensure_tensor(self, other: TensorLikeType):
        if isinstance(other, Tensor):
            if self.device != other.device:
                raise ValueError('Tensors must be in the same device')
            
            return other
        
        return Tensor(other, dtype=self.dtype, device=self.device)
    
    # Utils methods

    def detach(self):
        return Tensor(self.data, dtype=self.dtype, device=self.device)

    def numpy(self):
        if self.device == 'cuda':
            raise ValueError('Tensor must be in CPU to convert to numpy')
        
        if self.requires_grad:
            raise RuntimeError('Tensor must not require grad to convert to numpy')
        
        return self.data
    
    def to(self, device: DeviceLikeType):
        if device not in ('cpu', 'cuda'):
            raise ValueError('Invalid device')
        
        data = self.data if self.device == 'cpu' else self.data.get()
            
        return Tensor(data, dtype=self.dtype, requires_grad=self.requires_grad, device=device)
    
    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('cuda')
    
    def item(self):
        return self.data.item()
    
    # Boolean operations (non gradient)

    def invert(self):
        return Tensor(~self.data, dtype=self.dtype, device=self.device)

    def greater(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)    
    
        return Tensor(self.data > other_t.data, dtype=self.dtype, device=self.device)
    
    def greater_equal(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)
        
        return Tensor(self.data >= other_t.data, dtype=self.dtype, device=self.device)
    
    def less(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)
        
        return Tensor(self.data < other_t.data, dtype=self.dtype, device=self.device)
    
    def less_equal(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)
        
        return Tensor(self.data <= other_t.data, dtype=self.dtype, device=self.device)
    
    def equal(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)
        
        return Tensor(self.data == other_t.data, dtype=self.dtype, device=self.device)
    
    def not_equal(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)
        
        return Tensor(self.data != other_t.data, dtype=self.dtype, device=self.device)

    # Indices operations (non gradient)
    
    def argmin(self, dim: SupportsIndex = 0, keepdim: bool = False):
        data = self.data.argmin(axis=dim, keepdims=keepdim)
        
        return Tensor(data, dtype=self.lib.int32, device=self.device)

    def argmax(self, dim: SupportsIndex = 0, keepdim: bool = False):
        data = self.data.argmax(axis=dim, keepdims=keepdim)
        
        return Tensor(data, dtype=self.lib.int32, device=self.device)
    
    # Single operations
    
    def sign(self):
        data = self.lib.sign(self.data)
        sign_backward = None
        
        if self.requires_grad:
            def sign_backward(grad: NDArray):
                self.backward(grad * self.lib.zeros(self.shape))
                
        return Tensor(data, self.dtype, self.requires_grad, sign_backward, self.device)

    def abs(self):
        data = self.lib.abs(self.data)
        abs_backward = None

        if self.requires_grad:
            def abs_backward(grad: NDArray):
                self.backward(grad * self.lib.sign(self.data))

        return Tensor(data, self.dtype, self.requires_grad, abs_backward, self.device)

    def positive(self):
        data = self.data
        pos_backward = None

        if self.requires_grad:
            def pos_backward(grad: NDArray):
                self.backward(grad)

        return Tensor(data, self.dtype, self.requires_grad, pos_backward, self.device)

    def negative(self):
        data = -self.data
        neg_backward = None  

        if self.requires_grad:
            def neg_backward(grad: NDArray):
                self.backward(-grad)

        return Tensor(data, self.dtype, self.requires_grad, neg_backward, self.device)

    def sqrt(self):
        data = self.lib.sqrt(self.data)
        sqrt_backward = None
        
        if self.requires_grad:
            def sqrt_backward(grad: NDArray):
                self.backward(grad / (2 * data))
            
        return Tensor(data, self.dtype, self.requires_grad, sqrt_backward, self.device)

    def log(self, clip: bool = False):
        data = self.lib.log(self.data)
        
        if clip:
            data = data.clip(min=-100)
        
        log_backward = None

        if self.requires_grad:
            def log_backward(grad: NDArray):
                self_data = self.data
                
                if clip:
                    self_data = self_data.clip(min=1e-12)
                
                self.backward(grad / self_data)

        return Tensor(data, self.dtype, self.requires_grad, log_backward, self.device)

    def exp(self):
        data = self.lib.exp(self.data)
        exp_backward = None

        if self.requires_grad:            
            def exp_backward(grad: NDArray):
                self.backward(grad * data)

        return Tensor(data, self.dtype, self.requires_grad, exp_backward, self.device)

    def tanh(self):
        data = self.lib.tanh(self.data)
        tanh_backward = None

        if self.requires_grad:
            def tanh_backward(grad: NDArray):
                self.backward(grad * (1 - data ** 2))

        return Tensor(data, self.dtype, self.requires_grad, tanh_backward, self.device)

    def sigmoid(self):
        data = 1 / (1 + self.lib.exp(-self.data))
        sigmoid_backward = None
        
        if self.requires_grad:
            def sigmoid_backward(grad: NDArray):
                self.backward(grad * data * (1 - data))
    
        return Tensor(data, self.dtype, self.requires_grad, sigmoid_backward, self.device)
    
    def sin(self):
        data = self.lib.sin(self.data)
        sin_backward = None

        if self.requires_grad:
            def sin_backward(grad: NDArray):
                self.backward(grad * self.lib.cos(self.data))

        return Tensor(data, self.dtype, self.requires_grad, sin_backward, self.device)

    def cos(self):
        data = self.lib.cos(self.data)
        cos_backward = None

        if self.requires_grad:
            def cos_backward(grad: NDArray):
                self.backward(grad * -self.lib.sin(self.data))

        return Tensor(data, self.dtype, self.requires_grad, cos_backward, self.device)

    # Binary operations

    def add(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.data + other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        add_backward = None

        if requires_grad:
            def add_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad)
                
                if other_t.requires_grad:
                    other_t.backward(grad)

        return Tensor(data, self.dtype, requires_grad, add_backward, self.device)

    def sub(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.data - other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        sub_backward = None

        if requires_grad:
            def sub_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad)
                
                if other_t.requires_grad:
                    other_t.backward(-grad)

        return Tensor(data, self.dtype, requires_grad, sub_backward, self.device)

    def mul(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.data * other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        mul_backward = None

        if requires_grad:
            def mul_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad * other_t.data)
                    
                if other_t.requires_grad:
                    other_t.backward(grad * self.data)

        return Tensor(data, self.dtype, requires_grad, mul_backward, self.device)

    def div(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.data / other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        div_backward = None

        if requires_grad:
            def div_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad / other_t.data)
                    
                if other_t.requires_grad:
                    other_t.backward(-grad * self.data / other_t.data ** 2)

        return Tensor(data, self.dtype, requires_grad, div_backward, self.device)

    def matmul(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)
        
        data = self.data @ other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        matmul_backward = None

        if requires_grad:
            def matmul_backward(grad: NDArray):
                if self.requires_grad:
                    # Matrix @ Matrix or Vector @ Matrix
                    if (self.ndim > 1 and other_t.ndim > 1 or self.ndim == 1 and other_t.ndim > 1):
                        self.backward(grad @ other_t.data.swapaxes(-1, -2))

                    elif self.ndim == 1 and other_t.ndim == 1: # Vector @ Vector
                        self.backward(grad * other_t.data)

                    elif self.ndim > 1 and other_t.ndim == 1: # Matrix @ Vector
                        self.backward(self.lib.outer(grad, other_t))
                
                if other_t.requires_grad:
                    # Matrix @ Matrix or Matrix @ Vector
                    if (self.ndim > 1 and other_t.ndim > 1 or self.ndim > 1 and other_t.ndim == 1):
                        other_t.backward(self.data.swapaxes(-1, -2) @ grad)

                    elif self.ndim == 1 and other_t.ndim == 1: # Vector @ Vector
                        other_t.backward(grad * self.data)

                    elif self.ndim == 1 and other_t.ndim > 1: # Vector @ Matrix
                        other_t.backward(self.lib.outer(self.data, grad))

        return Tensor(data, self.dtype, requires_grad, matmul_backward, self.device)

    def outer(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.lib.outer(self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        outer_backward = None

        if requires_grad:
            def outer_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward((grad @ other_t.data.reshape(-1)).reshape(self.shape))
                    
                if other_t.requires_grad:
                    other_t.backward((self.data.reshape(-1) @ grad).reshape(other_t.shape))
                
        return Tensor(data, self.dtype, requires_grad, outer_backward, self.device)

    def pow(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.data ** other_t.data
        requires_grad = self.requires_grad or other_t.requires_grad
        pow_backward = None

        if requires_grad:
            def pow_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad * other_t.data * self.data ** (other_t.data - 1))
                    
                if other_t.requires_grad:
                    other_t.backward(grad * self.lib.log(self.data) * data)

        return Tensor(data, self.dtype, requires_grad, pow_backward, self.device)

    def maximum(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.lib.maximum(self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        maximum_backward = None

        if requires_grad:
            def maximum_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad * (self.data > other_t.data))
                    self.backward(grad * 0.5 * (self.data == other_t.data))
                    
                if other_t.requires_grad:
                    other_t.backward(grad * (other_t.data > self.data))
                    other_t.backward(grad * 0.5 * (other_t.data == self.data))

        return Tensor(data, self.dtype, requires_grad, maximum_backward, self.device)

    def minimum(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)

        data = self.lib.minimum(self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        minimum_backward = None

        if requires_grad:
            def minimum_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad * (self.data < other_t.data))
                    self.backward(grad * 0.5 * (self.data == other_t.data))
                    
                if other_t.requires_grad:
                    other_t.backward(grad * (other_t.data < self.data))
                    other_t.backward(grad * 0.5 * (other_t.data == self.data))

        return Tensor(data, self.dtype, requires_grad, minimum_backward, self.device)

    def embedding(self, other: TensorLikeType):
        other_t = self.ensure_tensor(other)
        
        data = self.data[other_t.data]
        embedding_backward = None
        
        if self.requires_grad:
            def embedding_backward(grad: NDArray):
                grad_ = self.lib.zeros(self.shape)
                grad_[other_t.data] = grad
                
                for i, count in enumerate(self.lib.bincount(other_t.data.reshape(-1))):
                    grad_[i] *= count
                
                self.backward(grad_)
                
        return Tensor(data, self.dtype, self.requires_grad, embedding_backward, self.device)

    # Batch operations

    def sum(self, dim: Optional[ShapeLike] = None, keepdim: bool = False):
        data = self.data.sum(axis=dim, keepdims=keepdim)
        sum_backward = None

        if self.requires_grad:
            def sum_backward(grad: NDArray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = self.lib.expand_dims(grad, dim)
            
                self.backward(grad * self.lib.ones(self.shape))

        return Tensor(data, self.dtype, self.requires_grad, sum_backward, self.device)

    def mean(self, dim: Optional[ShapeLike] = None, keepdim: bool = False):
        data = self.data.mean(axis=dim, keepdims=keepdim)
        mean_backward = None

        if self.requires_grad:
            def mean_backward(grad: NDArray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = self.lib.expand_dims(grad, dim)

                # Compute size of the mean
                dim_ = list(dim) if isinstance(dim, tuple) else dim
                size = self.lib.array(self.shape)[dim_].prod() # type: ignore
                
                self.backward(grad * self.lib.ones(self.shape) / size)

        return Tensor(data, self.dtype, self.requires_grad, mean_backward, self.device)

    def var(self, dim: Optional[ShapeLike] = None, correction: int = 1, keepdim: bool = False):
        data = self.data.var(axis=dim, keepdims=keepdim, ddof=correction)
        var_backward = None

        if self.requires_grad:
            def var_backward(grad: NDArray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = self.lib.expand_dims(grad, dim)

                # Compute size of the variance
                dim_ = list(dim) if isinstance(dim, tuple) else dim
                size = self.lib.array(self.shape)[dim_].prod() - correction # type: ignore

                # Compute mean
                mean = self.data.mean(axis=dim, keepdims=True)
 
                self.backward(grad * self.lib.ones(self.shape) * 2 * (self.data - mean) / size)
                
        return Tensor(data, self.dtype, self.requires_grad, var_backward, self.device)

    def max(self, dim: Optional[ShapeLike] = None, keepdim: bool = False):
        data = self.data.max(axis=dim, keepdims=keepdim)
        max_backward = None

        if self.requires_grad:
            def max_backward(grad: NDArray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = self.lib.expand_dims(grad, dim)

                mask = self.data == self.data.max(axis=dim, keepdims=True)
                size = mask.sum(axis=dim, keepdims=True)

                self.backward(grad * mask / size)

        return Tensor(data, self.dtype, self.requires_grad, max_backward, self.device)

    def min(self, dim: Optional[ShapeLike] = None, keepdim: bool = False):
        data = self.data.min(axis=dim, keepdims=keepdim)
        min_backward = None

        if self.requires_grad:
            def min_backward(grad: NDArray):
                # Expand gradient to match data shape
                if self.ndim != grad.ndim and dim is not None:
                    grad = self.lib.expand_dims(grad, dim)

                mask = self.data == self.data.min(axis=dim, keepdims=True)
                size = mask.sum(axis=dim, keepdims=True)

                self.backward(grad * mask / size)

        return Tensor(data, self.dtype, self.requires_grad, min_backward, self.device)

    # Shape operations

    def reshape(self, shape: ShapeLike):
        data = self.data.reshape(shape)
        reshape_backward = None

        if self.requires_grad:
            def reshape_backward(grad: NDArray):
                self.backward(grad.reshape(self.shape))

        return Tensor(data, self.dtype, self.requires_grad, reshape_backward, self.device)

    def transpose(self, axes: Optional[ShapeLike] = None):
        data = self.data.transpose(axes)
        transpose_backward = None
        
        if self.requires_grad:
            def transpose_backward(grad: NDArray):
                self.backward(grad.transpose(axes))
                
        return Tensor(data, self.dtype, self.requires_grad, transpose_backward, self.device)

    def swapaxes(self, axis0: SupportsIndex, axis1: SupportsIndex):
        data = self.data.swapaxes(axis0, axis1)
        swapaxes_backward = None
        
        if self.requires_grad:
            def swapaxes_backward(grad: NDArray):
                self.backward(grad.swapaxes(axis0, axis1))

        return Tensor(data, self.dtype, self.requires_grad, swapaxes_backward, self.device)

    def flip(self, dims: Optional[ShapeLike] = None):
        data = self.lib.flip(self.data, axis=dims)
        flip_backward = None
        
        if self.requires_grad:
            def flip_backward(grad: NDArray):
                self.backward(self.lib.flip(grad, axis=dims))
                
        return Tensor(data, self.dtype, self.requires_grad, flip_backward, self.device)
    
    def unsqueeze(self, dim: ShapeLike):
        data = self.lib.expand_dims(self.data, dim)
        unsqueeze_backward = None
        
        if self.requires_grad:
            def unsqueeze_backward(grad: NDArray):
                self.backward(grad.squeeze(dim)) # type: ignore
                
        return Tensor(data, self.dtype, self.requires_grad, unsqueeze_backward, self.device)

    def squeeze(self, dim: Optional[ShapeLike] = None):
        data = self.lib.squeeze(self.data, dim)
        squeeze_backward = None
        
        if self.requires_grad:
            def squeeze_backward(grad: NDArray):
                if dim is None:
                    self.backward(grad.reshape(self.shape))
                else:
                    self.backward(self.lib.expand_dims(grad, dim))
                
        return Tensor(data, self.dtype, self.requires_grad, squeeze_backward, self.device)
    
    # Other operations

    # TODO: Implement multiple repeats and dim parameter
    def repeat(self, repeat: int):
        data = self.data.repeat(repeat)
        repeat_backward = None
        
        if self.requires_grad:
            def repeat_backward(grad: NDArray):
                self.backward(grad.reshape((-1, repeat)).sum(axis=1).reshape(self.shape))

        return Tensor(data, self.dtype, self.requires_grad, repeat_backward, self.device)

    # TODO: Implement multiple tiles
    def tile(self, dim: int):
        data = self.lib.tile(self.data, dim)
        tile_backward = None
        
        if self.requires_grad:
            def tile_backward(grad: NDArray):
                self.backward(grad.reshape((dim, -1)).sum(axis=0).reshape(self.shape))

        return Tensor(data, self.dtype, self.requires_grad, tile_backward, self.device)

    # TODO: Implement mode 
    def pad(self, pad_width: ArrayLikeInt, constant_values: float = 0.0):
        data = self.lib.pad(self.data, pad_width, constant_values=constant_values)
        pad_backward = None
        
        if self.requires_grad:
            def pad_backward(grad: NDArray):
                self.backward(grad[tuple([slice(left, -right if right != 0 else None) for left, right in pad_width])])
                
        return Tensor(data, self.dtype, self.requires_grad, pad_backward, self.device)

    def stack(self, arrays: Sequence[TensorLikeType], dim: SupportsIndex = 0):
        tensors = [self] + [self.ensure_tensor(item) for item in arrays]
        
        data = self.lib.stack([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        stack_backward = None
        
        if requires_grad:
            def stack_backward(grad: NDArray):
                grads = self.lib.split(grad, len(tensors), axis=dim)
        
                for tensor, grad in zip(tensors, grads):
                    if not tensor.requires_grad:
                        continue
                        
                    tensor.backward(grad.reshape(tensor.shape))

        return Tensor(data, self.dtype, requires_grad, stack_backward, self.device)

    def cat(self, arrays: Sequence[TensorLikeType], dim: SupportsIndex = 0):
        tensors = [self] + [self.ensure_tensor(item) for item in arrays]

        data = self.lib.concatenate([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        cat_backward = None

        if requires_grad:
            def cat_backward(grad: NDArray):
                # Get the indices to split the gradient
                indices = self.lib.cumsum([t.shape[dim] for t in tensors[:-1]])

                grads = self.lib.split(grad, indices, axis=dim)
                
                for tensor, grad in zip(tensors, grads):
                    if not tensor.requires_grad:
                        continue
                        
                    tensor.backward(grad)

        return Tensor(data, self.dtype, requires_grad, cat_backward, self.device)
    
    def where(self, condition: TensorLikeType, other: TensorLikeType):
        condition_t = self.ensure_tensor(condition)
        other_t = self.ensure_tensor(other)
        
        data = self.lib.where(condition_t.data, self.data, other_t.data)
        requires_grad = self.requires_grad or other_t.requires_grad
        where_backward = None
        
        if requires_grad:
            def where_backward(grad: NDArray):
                if self.requires_grad:
                    self.backward(grad * condition_t.data)
                    
                if other_t.requires_grad:
                    other_t.backward(grad * ~condition_t.data)
        
        return Tensor(data, self.dtype, requires_grad, where_backward, self.device)

    def getitem(self, key):
        data = self.data[key]
        getitem_backward = None

        if self.requires_grad:
            def getitem_backward(grad: NDArray):
                grad_ = self.lib.zeros(self.shape)
                
                grad_[key] = grad
                
                self.backward(grad_)

        return Tensor(data, self.dtype, self.requires_grad, getitem_backward, self.device) 

    def setitem(self, key, value: TensorLikeType):
        if not hasattr(self, 'setitem_tensors'):
            self.setitem_tensors: list[tuple[Any, Tensor]] = []
        
        value_t = self.ensure_tensor(value)
        self.setitem_tensors.append((key, value_t))
        
        self.data[key] = value_t.data
        
        if self.requires_grad:
            def setitem_backward(grad: NDArray):
                for key, value_t in self.setitem_tensors:
                    if value_t.requires_grad:
                        value_t.backward(grad[key])
        
            self.grad_fn = setitem_backward

    def iter(self):
        return iter(self.getitem(i) for i in range(self.shape[0]))
    
    # Magic methods

    def __invert__(self):
        return self.invert()

    def __gt__(self, other: TensorLikeType):
        return self.greater(other)
    
    def __ge__(self, other: TensorLikeType):
        return self.greater_equal(other)
    
    def __lt__(self, other: TensorLikeType):
        return self.less(other)
    
    def __le__(self, other: TensorLikeType):
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

    def __add__(self, other: TensorLikeType):
        return self.add(other)

    def __radd__(self, other: TensorLikeType):
        return self.ensure_tensor(other).add(self)

    def __sub__(self, other: TensorLikeType):
        return self.sub(other)

    def __rsub__(self, other: TensorLikeType):
        return self.ensure_tensor(other).sub(self)

    def __mul__(self, other: TensorLikeType):
        return self.mul(other)

    def __rmul__(self, other: TensorLikeType):
        return self.ensure_tensor(other).mul(self)

    def __truediv__(self, other: TensorLikeType):
        return self.div(other)

    def __rtruediv__(self, other: TensorLikeType):
        return self.ensure_tensor(other).div(self)

    def __matmul__(self, other: TensorLikeType):
        return self.matmul(other)

    def __rmatmul__(self, other: TensorLikeType):
        return self.ensure_tensor(other).matmul(self)

    def __pow__(self, other: TensorLikeType):
        return self.pow(other)

    def __rpow__(self, other: TensorLikeType):
        return self.ensure_tensor(other).pow(self)

    def __getitem__(self, key):
        return self.getitem(key)
        
    def __setitem__(self, key, value: TensorLikeType):
        self.setitem(key, value)
        
    def __iter__(self):
        return self.iter()

    def __array__(self):
        return self.data if self.device == 'cpu' else self.data.get()

    # Properties

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self) -> int:
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
    
    # Backward method
    def backward(self, grad: Optional[NDArray] = None):
        ''' Backpropagates the gradient through the computation graph '''
        
        if not self.requires_grad:
            raise RuntimeError('Cannot compute gradient on a non-required-grad tensor')

        # Initialize gradient if not provided
        if grad is None:
            grad = self.lib.ones(self.shape, dtype=self.dtype)
        else:
            grad = self.lib.array(grad, dtype=self.dtype)
                
        # Sum gradient to match data shape
        if self.shape != grad.shape:
            keepdims = self.ndim == grad.ndim

            if keepdims:
                self_shape = self.lib.array(self.shape)
            else:
                self_shape = self.lib.array((1,) * (grad.ndim - self.ndim) + self.shape)

            grad_shape = self.lib.array(grad.shape)

            dim = tuple([value.item() for value in self.lib.where(self_shape != grad_shape)[0]])

            grad = grad.sum(axis=dim, keepdims=keepdims).reshape(self.shape)

        if self.grad is None:
            # Initialize gradient
            self.grad = grad
        else:
            # Accumulate gradient
            self.grad += grad

        # Backpropagate gradient
        if grad is not None and self.grad_fn is not None:
            self.grad_fn(grad)
