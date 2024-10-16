from pickle import load, dump

from mytorch.autograd import *

# Data Types

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
float16 = np.float16
float32 = np.float32
float64 = np.float64

# File operations

def save(obj: object, out_file: str, protocol: int = 2):
    with open(out_file, 'wb') as file:
        dump(obj, file, protocol=protocol)

def load(in_file: str):
    with open(in_file, 'rb') as file:
        return load(file)

# Factory methods

def tensor(
    data: TensorLikeType, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, device=device)

def ones(
    shape: ShapeLike, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.ones(shape), dtype=dtype, requires_grad=requires_grad, device=device)

def ones_like(
    tensor: Tensor, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.ones_like(tensor.data), dtype=dtype, requires_grad=requires_grad, device=device)

def zeros(
    shape: ShapeLike, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad, device=device)

def zeros_like(
    tensor: Tensor, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.zeros_like(tensor.data), dtype=dtype, requires_grad=requires_grad, device=device)

def rand(
    *shape: int, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.random.rand(*shape), dtype=dtype, requires_grad=requires_grad, device=device)

def randn(
    *shape: int, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad, device=device)

def binomial(
    n: int, 
    p: float, 
    shape: Optional[ShapeLike] = None, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.random.binomial(n, p, shape), dtype=dtype, requires_grad=requires_grad, device=device)

def uniform(
    low: float, 
    high: float, 
    shape: Optional[ShapeLike] = None, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.random.uniform(low, high, shape), dtype=dtype, requires_grad=requires_grad, device=device)

def arange(
    start = 0, 
    stop = 0, 
    step = 1, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    return Tensor(np.arange(start, stop, step), dtype=dtype, requires_grad=requires_grad, device=device)

def indices(
    shape: Sequence[int], 
    sparse = False, 
    dtype: DTypeLike = None, 
    requires_grad = False, 
    device: DeviceLikeType = 'cpu'
):
    data = np.indices(shape, sparse=sparse)
    
    if sparse:
        return [Tensor(item, dtype=dtype, requires_grad=requires_grad, device=device) for item in data]
    else:
        return Tensor(data, dtype=dtype, requires_grad=requires_grad, device=device)

# Boolean operations (non gradient)

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

# Indices operations (non gradient)

def argmin(input: Tensor, dim: Optional[ShapeLike] = None, keepdim: bool = False):
    return input.argmin(dim=dim, keepdim=keepdim)

def argmax(input: Tensor, dim: Optional[ShapeLike] = None, keepdim: bool = False):
    return input.argmax(dim=dim, keepdim=keepdim)

# Single operations

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

def sigmoid(input: Tensor):
    return input.sigmoid()

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

def embedding(input: Tensor, other: Tensor):
    return input.embedding(other)

# Batch operations

def sum(input: Tensor, dim: Optional[ShapeLike] = None, keepdim: bool = False):
    return input.sum(dim=dim, keepdim=keepdim)

def mean(input: Tensor, dim: Optional[ShapeLike] = None, keepdim: bool = False):
    return input.mean(dim=dim, keepdim=keepdim)

def var(input: Tensor, dim: Optional[ShapeLike] = None, correction: int = 1, keepdim: bool = False):
    return input.var(dim=dim, correction=correction, keepdim=keepdim)

def max(input: Tensor, dim: Optional[ShapeLike] = None, keepdim: bool = False):
    return input.max(dim=dim, keepdim=keepdim)

def min(input: Tensor, dim: Optional[ShapeLike] = None, keepdim: bool = False):
    return input.min(dim=dim, keepdim=keepdim)

# Shape operations

def reshape(input: Tensor, shape: ShapeLike):
    return input.reshape(shape)

def transpose(input: Tensor, axes: Optional[ShapeLike] = None):
    return input.transpose(axes)

def swapaxes(input: Tensor, axis1: SupportsIndex, axis2: SupportsIndex):
    return input.swapaxes(axis1, axis2)

def flip(input: Tensor, dims: Optional[ShapeLike] = None):
    return input.flip(dims=dims)

def unsqueeze(input: Tensor, dim: ShapeLike):
    return input.unsqueeze(dim)

def squeeze(input: Tensor, dim: Optional[ShapeLike] = None):
    return input.squeeze(dim=dim)

# Other operations

def stack(arrays: Sequence[Tensor], dim: SupportsIndex = 0):
    return arrays[0].stack(arrays[1:], dim=dim)

def cat(arrays: Sequence[Tensor], dim: SupportsIndex = 0):
    return arrays[0].cat(arrays[1:], dim=dim)

def where(condition: Tensor, input: Tensor, other: Tensor):
    return input.where(condition, other)
