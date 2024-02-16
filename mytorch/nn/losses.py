import mytorch as mt
from mytorch.nn import Softmax

from abc import ABC, abstractmethod
from typing import Optional, Literal

class Loss(ABC):
    ''' Base class for all loss functions. '''
    
    @abstractmethod
    def forward(self, p: mt.Tensor, y: mt.Tensor) -> mt.Tensor:
        ''' Forward pass. '''
        
        pass
    
    def __call__(self, p: mt.Tensor, y: mt.Tensor) -> mt.Tensor:
        ''' When the object is called, it calls the forward method. '''
        
        return self.forward(p, y)    
    
class L1Loss(Loss):
    ''' Mean Absolute Error Loss. '''
    
    def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        self.reduction = reduction
    
    def forward(self, p: mt.Tensor, y: mt.Tensor):
        loss = mt.abs(p - y)
        
        if self.reduction == 'mean':
            return mt.mean(loss)
        elif self.reduction == 'sum':
            return mt.sum(loss)
        else:
            return loss
    
class MSELoss(Loss):
    ''' Mean Squared Error Loss. '''
    
    def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        self.reduction = reduction
    
    def forward(self, p: mt.Tensor, y: mt.Tensor):
        loss = (p - y) ** 2
        
        if self.reduction == 'mean':
            return mt.mean(loss)
        elif self.reduction == 'sum':
            return mt.sum(loss)
        else:
            return loss
        
class BCELoss(Loss):
    ''' Binary Cross Entropy Loss. '''
    
    def __init__(
        self, 
        weight: Optional[mt.Tensor] = None,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, p: mt.Tensor, y: mt.Tensor):
        if self.weight is None:
            self.weight = mt.ones(p.shape[0])
        
        loss = -self.weight * (y * mt.log(p) + (1 - y) * mt.log(1 - p))
        
        if self.reduction == 'mean':
            return mt.mean(loss)
        elif self.reduction == 'sum':
            return mt.sum(loss)
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
    
    def forward(self, p: mt.Tensor, y: mt.Tensor):
        loss = mt.where(
            mt.abs(p - y) < self.delta,
            0.5 * (p - y) ** 2, 
            self.delta * (mt.abs(p - y) - 0.5 * self.delta)
        )
        
        if self.reduction == 'mean':
            return mt.mean(loss)
        elif self.reduction == 'sum':
            return mt.sum(loss)
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
        
    def forward(self, p: mt.Tensor, y: mt.Tensor):
        if not self.log_target:
            loss = y * (mt.log(y) - p)
        else:
            loss = mt.exp(y) * (y - p)
        
        if self.reduction == 'mean':
            return mt.mean(loss)
        elif self.reduction == 'batchmean':
            return mt.sum(loss) / p.shape[0]
        elif self.reduction == 'sum':
            return mt.sum(loss)
        else:
            return loss
        
class NLLLoss(Loss):
    ''' Negative Log Likelihood Loss. '''
    
    def __init__(
        self,
        weight: Optional[mt.Tensor] = None, 
        ignore_index: int = -100, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, p: mt.Tensor, y: mt.Tensor):
        if self.weight is None:
            self.weight = mt.ones(p.shape[1])
        
        if p.ndim == 2:
            p = p[..., None]

        if y.ndim == 1:
            y = y[..., None]
        
        indices = mt.indices(y.shape, sparse=True)
        criterion = (indices[0], y, *indices[1:])
        
        weight = self.weight[y] * (y != self.ignore_index)
        
        loss = -weight * p[criterion]
        
        if self.reduction == 'mean':
            return mt.sum(loss / mt.sum(weight))
        elif self.reduction == 'sum':
            return mt.sum(loss)
        else:
            return loss
        
class CrossEntroyLoss(Loss):
    ''' Cross Entropy Loss. '''
    
    def __init__(
        self,
        weight: Optional[mt.Tensor] = None, 
        ignore_index: int = -100, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        self.softmax = Softmax(axis=1)
        self.nll_loss = NLLLoss(weight, ignore_index, reduction)
        
    def forward(self, p: mt.Tensor, y: mt.Tensor):
        return self.nll_loss(mt.log(self.softmax(p)), y)