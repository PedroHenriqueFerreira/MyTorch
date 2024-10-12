from mytorch import Tensor, DeviceLikeType
      
class Module:
    ''' Base class for all neural network modules. '''
    
    def __init__(self):
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def parameters(self):
        params: list[Tensor] = []
        
        for name, item in self.__dict__.items():
            if isinstance(item, Tensor):
                if item.requires_grad:
                    params.append(item)
            
            if hasattr(item, 'parameters'):
                params.extend(item.parameters())

        return params
    
    def eval(self):
        self.training = False
        
        for name, item in self.__dict__.items():
            if hasattr(item, 'eval'):
                item.eval()
                
    def train(self, mode: bool = True):
        self.training = mode
        
        for name, item in self.__dict__.items():
            if hasattr(item, 'train'):
                item.train(mode)
                
    def to(self, device: DeviceLikeType):
        self.device = device
        
        for name, item in self.__dict__.items():
            if hasattr(item, 'to'):
                self.__dict__[name] = item.to(device)
                
        return self
                
    def cpu(self):
        self.to('cpu')
        
    def cuda(self):
        self.to('cuda')
        
class Sequential(Module):
    ''' Sequential container. '''
    
    def __init__(self, *modules: Module):
        self.modules = list(modules)
        self.training = True
        
    def forward(self, x: Tensor) -> Tensor:
        ''' Forward pass. '''
        
        for module in self.modules:
            x = module(x)
            
        return x
    
    def parameters(self):
        ''' Returns the parameters of the model. '''
        
        params = []
        
        for module in self.modules:
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
            
        return params
    
    def eval(self):
        self.training = False
        
        for module in self.modules:
            if hasattr(module, 'eval'):
                module.eval()
                
    def train(self, mode: bool = True):
        self.training = mode
        
        for module in self.modules:
            if hasattr(module, 'train'):
                module.train(mode)
    
    def to(self, device: DeviceLikeType):
        for i, module in enumerate(self.modules):
            if hasattr(module, 'to'):
                self.modules[i] = module.to(device)
                
        return self
                
class ModuleList(Module):
    def __init__(self, modules: list[Module]):
        self.modules = modules
        self.training = True
        
    def __getitem__(self, idx: int):
        return self.modules[idx]
    
    def __setitem__(self, idx: int, module: Module):
        self.modules[idx] = module
        
    def __delitem__(self, idx: int):
        del self.modules[idx]
    
    def __len__(self):
        return len(self.modules)
    
    def append(self, module: Module):
        self.modules.append(module)
        
    def extend(self, modules: list[Module]):
        self.modules.extend(modules)
        
    def insert(self, idx: int, module: Module):
        self.modules.insert(idx, module)
        
    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module.forward(x)
            
        return x
    
    def parameters(self):
        params = []
        
        for module in self.modules:
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
                
        return params
    
    def eval(self):
        self.training = False
        
        for module in self.modules:
            if hasattr(module, 'eval'):
                module.eval()
                
    def train(self, mode: bool = True):
        self.training = mode
        
        for module in self.modules:
            if hasattr(module, 'train'):
                module.train(mode)
                
    def to(self, device: DeviceLikeType):
        for i, module in enumerate(self.modules):
            if hasattr(module, 'to'):
                self.modules[i] = module.to(device)
                
        return self