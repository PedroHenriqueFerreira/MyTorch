from abc import ABC, abstractmethod

from mytorch.autograd import Tensor

import numpy as np

class Loss(ABC):
    ''' Base class for all loss functions. '''
    
    @abstractmethod
    def forward(self, p: Tensor, y: Tensor) -> Tensor: