from abc import ABC, abstractmethod
from typing import Optional, Literal
from numpy._typing import ArrayLike

from mytorch.autograd import Tensor

Reduction = Literal['mean', 'sum', 'none']

class Loss(ABC):
    ''' Base class for all loss functions. '''
    
    @abstractmethod
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, p: Tensor, y: Tensor) -> Tensor:
        ''' When the object is called, it calls the forward method. '''
        
        return self.forward(p, y)    
    
class MSELoss(Loss):
    ''' Mean Squared Error Loss. '''
    
    def __init__(self, reduction: Reduction = 'mean'):
        self.reduction = reduction
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        loss = (p - y) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class L1Loss(Loss):
    ''' Mean Absolute Error Loss. '''
    
    def __init__(self, reduction: Reduction = 'mean'):
        self.reduction = reduction
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        loss = (p - y).abs()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class BCELoss(Loss):
    ''' Binary Cross Entropy Loss. '''
    
    def __init__(self, weight: Optional[ArrayLike] = None, reduction: Reduction = 'mean'):
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        if self.weight is None:
            self.weight = 1
        
        loss = self.weight * - (y * p.log() + (1 - y) * (1 - p).log())
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss