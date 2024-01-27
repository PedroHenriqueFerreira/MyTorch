import numpy as np

class Tensor:
    def __init__(self, value, requires_grad=False):
        if isinstance(value, Tensor):
            self.data = np.array(value.data)
            self.grad = value.grad
        else:
            self.data = np.array(value)
            self.grad = None

        self.requires_grad = requires_grad

    def tensor(self, value, requires_grad=False):
        if isinstance(value, Tensor):
            return value
        
        return Tensor(value, requires_grad=requires_grad)

    def __repr__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones(self.data.shape)
        else:
            grad = np.array(grad)

        # Broadcast gradient if needed
        if self.data.shape != grad.shape:
            # Sum gradient to match shape of data
            if self.data.size == 1:
                
                grad = grad.sum()
                
            # Sum gradient in the axis that doesn't match the shape of data
            elif self.data.ndim == grad.ndim:
                
                self_shape = np.array(self.data.shape)
                grad_shape = np.array(grad.shape)

                axis = tuple(np.where(self_shape != grad_shape)[0])
                
                grad = grad.sum(axis=axis, keepdims=True)

            else:
                
                self_shape = np.array((1,) * (grad.ndim - self.data.ndim) + self.data.shape)
                grad_shape = np.array(grad.shape)
                
                axis = tuple(np.where(self_shape != grad_shape)[0])
                
                grad = grad.sum(axis=axis)
                
            grad = grad.reshape(self.data.shape)
            
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad