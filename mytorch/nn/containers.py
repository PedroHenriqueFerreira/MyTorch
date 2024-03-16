from abc import ABC, abstractmethod
from typing import Union

from mytorch import Tensor

from mytorch.nn import Layer, Activation

class Container(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        ''' Returns the parameters of the model. '''
        
        pass
    
    @abstractmethod
    def eval(self):
        ''' Sets the module to evaluation mode. '''
        
        pass
    
    @abstractmethod
    def train(self):
        ''' Sets the module to training mode. '''
        
        pass
      
class Module(Container):
    ''' Base class for all neural network modules. '''
    
    def __init__(self):
        self.training = True
    
    def parameters(self):
        params = []
        
        for name, item in self.__dict__.items():
            if hasattr(item, 'parameters'):
                params.extend(item.parameters())
            
            if hasattr(item, 'named_parameters'):
                for name, param in item.named_parameters():
                    if hasattr(param, 'requires_grad'):
                        if param.requires_grad:
                            params.append(param)

        return params
    
    def eval(self):
        self.training = False
        
        for name, item in self.__dict__.items():
            if hasattr(item, 'eval'):
                item.eval()
                
    def train(self):
        self.training = True
        
        for name, item in self.__dict__.items():
            if hasattr(item, 'train'):
                item.train()
        
class Sequential(Container):
    ''' Sequential container. '''
    
    def __init__(self, *layers: Union[Layer, Activation, Module, 'Sequential']):
        self.layers = layers
        
        self.training = True
        
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    def parameters(self):
        ''' Returns the parameters of the model. '''
        
        params = []
        
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
            
            if hasattr(layer, 'named_parameters'):
                for name, param in layer.named_parameters():
                    if hasattr(param, 'requires_grad'):
                        if param.requires_grad:
                            params.append(param)
            
        return params
    
    def eval(self):
        self.training = False
        
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
                
    def train(self):
        self.training = True
        
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()