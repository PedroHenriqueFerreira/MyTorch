from abc import ABC, abstractmethod
from typing import Optional, Literal
from numpy._typing import ArrayLike

from mytorch.autograd import Tensor

import numpy as np

class Loss(ABC):
    ''' Base class for all loss functions. '''
    
    @abstractmethod
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, p: Tensor, y: Tensor) -> Tensor:
        ''' When the object is called, it calls the forward method. '''
        
        return self.forward(p, y)    
    
class L1Loss(Loss):
    ''' Mean Absolute Error Loss. '''
    
    def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        self.reduction = reduction
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        loss = (p - y).abs()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
class MSELoss(Loss):
    ''' Mean Squared Error Loss. '''
    
    def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        self.reduction = reduction
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        loss = (p - y) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class BCELoss(Loss):
    ''' Binary Cross Entropy Loss. '''
    
    def __init__(
        self, 
        weight: Optional[Tensor] = None,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        if self.weight is None:
            self.weight = Tensor(1)
        
        loss = -self.weight * (y * p.log() + (1 - y) * (1 - p).log())
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class HuberLoss(Loss):
    ''' Huber Loss. '''
    
    def __init__(
        self, 
        delta: float = 1, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        loss = ((p - y).abs() < self.delta).where(
            0.5 * (p - y) ** 2, 
            self.delta * ((p - y).abs() - 0.5 * self.delta)
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class KLDivLoss(Loss):
    ''' Kullback-Leibler Divergence Loss. '''
    
    def __init__(
        self, 
        log_target: bool = False,
        reduction: Literal['mean', 'batchmean', 'sum', 'none'] = 'mean'
    ):
        self.log_target = log_target
        self.reduction = reduction
        
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        if not self.log_target:
            loss = y * (y.log() - p)
        else:
            loss = y.exp() * (y - p)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'batchmean':
            return loss.sum() / p.shape[0]
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class NLLLoss(Loss):
    ''' Negative Log Likelihood Loss. '''
    
    def __init__(
        self,
        weight: Optional[Tensor] = None, 
        ignore_index: int = -100, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, p: Tensor, y: Tensor):
        if self.weight is None:
            self.weight = Tensor(np.ones(p.shape[1])) # TODO: create tensor ones
            
        if p.ndim == 2:
            p = p[..., None]

        if y.ndim == 1:
            y = y[..., None]
        
        indices = np.indices(y.shape, sparse=True)
        criterion = (indices[0], y.data, *indices[1:])
        
        weight = self.weight[y] * (y != self.ignore_index)
        
        loss = -weight * p[criterion]
        
        if self.reduction == 'mean':
            return (loss / weight.sum()).sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss