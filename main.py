import mytorch

from mytorch import nn
from mytorch import optim

# Load the dataset

from numpy import array, transpose, float32
from time import time
from os import listdir
from PIL import Image

device = 'cpu'

images = []

for file in listdir('in/'):
    image = array(Image.open(f'in/{file}').convert('RGB')).astype(float32).transpose((2, 0, 1))
    
    images.append((image - 127.5) / 127.5)
    
images = array(images)

# Generator Code

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 128 * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128 * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(128 * 16, 128 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 8),
            nn.ReLU(),
            nn.ConvTranspose2d( 128 * 8, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU(),
            nn.ConvTranspose2d( 128 * 4, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(),
            nn.ConvTranspose2d( 128 * 2, 128 * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 1),
            nn.ReLU(),
            nn.ConvTranspose2d( 128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
    def __call__(self, input: mytorch.Tensor) -> mytorch.Tensor:
        return self.forward(input)
    
netG = Generator().to(device)
    
# Discriminator Code
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
    def __call__(self, input: mytorch.Tensor) -> mytorch.Tensor:
        return self.forward(input)
 
netD = Discriminator().to(device)
    
# Loss and Optimizers

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training

fixed_noise = mytorch.randn(10, 100, 1, 1, device=device)

epochs = 5
batchs = ((images.shape[0] - 1) // 10) + 1

n = 0

for epoch in range(epochs):
    G_losses, D_losses = [], []
    
    for batch in range(batchs):
        # Train Discriminator
        print('Training Discriminator')
        
        print('Training real images')
        
        print('Forward pass')
        
        optimizerD.zero_grad()
        
        real_images = mytorch.tensor(images[batch * 10 : (batch + 1) * 10], mytorch.float32, device=device)
        real_label = mytorch.ones((real_images.shape[0], ), mytorch.float32, device=device)
        real_output = netD(real_images).reshape(-1)
        
        print('REAL OUTPUT:', real_output.shape)
        print('REAL LABEL:', real_label.shape)
        
        real_errorD = criterion(real_output, real_label)
        
        print('Backward pass')
        
        real_errorD.backward()
        
        print('Training fake images') 
        
        print('Forward pass')
        
        noise = mytorch.randn(64, 100, 1, 1, mytorch.float32, device=device)
        
        fake_images = netG(noise)
        fake_label = mytorch.zeros((real_images.shape[0], ), mytorch.float32, device=device)
        fake_output = netD(fake_images.detach()).reshape(-1)
        
        fake_errorD = criterion(fake_output, fake_label)
        
        print('Backward pass')
          
        fake_errorD.backward()
        
        errorD = real_errorD + fake_errorD
        
        print('Optimizing')
        
        optimizerD.step()
        
        # Train Generator
        print('Training Generator')
        
        print('Forward pass')
        
        optimizerG.zero_grad()
        
        output = netD(fake_images).reshape(-1)
        
        errorG = criterion(output, real_label)
        
        print('Backward pass')
        
        errorG.backward()
        
        print('Optimizing')
        
        optimizerG.step()
        
        print('Saving losses')
        
        G_losses.append(errorG.item())
        D_losses.append(errorD.item())
        
        if batch % 1 == 0:
            print(f'[{epoch + 1}/{epochs}][{batch + 1}/{batchs}]\tLoss D: {G_losses[-1]}\tLoss G: {D_losses[-1]}')
            
        if n % 10 == 0:
            netG.eval()
            generated_images = netG(fixed_noise).detach().cpu().numpy()
            netG.train()
            
            for i in range(generated_images.shape[0]):
                image = (generated_images[i] * 127.5 + 127.5).astype(mytorch.int8).transpose((1, 2, 0))

                Image.fromarray(image, mode='RGB').save(f'out/{int(time())}_{i}.png')
            
        n += 1