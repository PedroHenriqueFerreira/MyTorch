from abc import ABC, abstractmethod

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
    
class MSELoss(Loss):
    ''' Mean Squared Error Loss. '''
    
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        return ((p - y) ** 2).mean()