import mytorch as mt

from mytorch.nn import Layer, Activation

from abc import ABC, abstractmethod

from typing import Union

class Container(ABC):
    @abstractmethod
    def forward(self, x: mt.Tensor) -> mt.Tensor:
        pass
    
    def __call__(self, x: mt.Tensor) -> mt.Tensor:
        return self.forward(x)

    @abstractmethod
    def parameters(self) -> list[mt.Tensor]:
        ''' Returns the parameters of the model. '''
        
        pass
      
class Module(Container):
    ''' Base class for all neural network modules. '''
    
    def parameters(self):
        params = []
        
        for name, item in self.__dict__.items():
            if hasattr(item, 'parameters'):
                params.extend(item.parameters())
                
            if hasattr(item, 'weight'):
                if isinstance(item.weight, mt.Tensor):
                    if item.weight.requires_grad:
                        params.append(item.weight)
                
            if hasattr(item, 'bias'):
                if isinstance(item.bias, mt.Tensor):
                    if item.bias.requires_grad:
                        params.append(item.bias)
            
            if hasattr(item, 'named_parameters'):
                for name, param in item.named_parameters():
                    if isinstance(param, mt.Tensor):
                        if param.requires_grad:
                            params.append(param)

        return params
        
class Sequential(Container):
    ''' Sequential container. '''
    
    def __init__(self, *layers: Union[Layer, Activation, Module, 'Sequential']):
        self.layers = layers
        
    def forward(self, x: mt.Tensor) -> mt.Tensor:
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
                
            if hasattr(layer, 'weight'):
                if isinstance(layer.weight, mt.Tensor):
                    if layer.weight.requires_grad:
                        params.append(layer.weight)
                
            if hasattr(layer, 'bias'):
                if isinstance(layer.bias, mt.Tensor):
                    if layer.bias.requires_grad:
                        params.append(layer.bias)
            
            if hasattr(layer, 'named_parameters'):
                for name, param in layer.named_parameters():
                    if isinstance(param, mt.Tensor):
                        if param.requires_grad:
                            params.append(param)
            
        return params