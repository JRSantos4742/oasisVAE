import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 20
learning_rate = 5e-3

#--------------
#Data
height = 128

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1317, 0.1317, 0.1317), (0.1864, 0.1864, 0.1864)),
                                      transforms.CenterCrop(height)])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1343, 0.1343, 0.1343), (0.1879, 0.1879, 0.1879)),
                                     transforms.CenterCrop(height)])


trainset = CustomDataset('keras_png_slices_train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
total_step = len(train_loader)

testset = CustomDataset('keras_png_slices_test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)

#--------------
# Model
hDimension1 = 256
hDimension2 = 128

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dimension):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hDimension1, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hDimension1)
        self.conv2 = nn.Conv2d(hDimension1, hDimension2, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hDimension2)
        self.linear = nn.Linear(hDimension2*height*height, latent_dimension)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out.view(-1, hDimension2*height*height)
        out = self.linear(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dimension):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dimension, hDimension2*height*height)
        self.bn1 = nn.BatchNorm2d(hDimension2)
        self.conv1 = nn.ConvTranspose2d(hDimension2, hDimension1, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hDimension1)
        self.conv2 = nn.ConvTranspose2d(hDimension1, in_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(-1, hDimension2, height, height)
        out = F.relu(self.bn1(out))
        out = F.relu(self.bn2(self.conv1(out)))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(out)
        return out

class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dimension):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hDimension1, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hDimension1)
        self.conv2 = nn.Conv2d(hDimension1, hDimension2, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hDimension2)
        self.linear1 = nn.Linear(hDimension2*height*height, latent_dimension)
        self.linear2 = nn.Linear(hDimension2*height*height, latent_dimension)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out.view(-1, hDimension2*height*height)
        mu =  self.linear1(out)
        sigma = torch.exp(self.linear2(out))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dimension):
        super().__init__()
         
        #self.encoder = Encoder(in_channels, latent_dimension)
        self.encoder = VariationalEncoder(in_channels, latent_dimension)
        self.decoder = Decoder(in_channels, latent_dimension)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
