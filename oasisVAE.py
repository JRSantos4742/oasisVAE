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

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dimension):
        super().__init__()
         
        self.encoder = Encoder(in_channels, latent_dimension)
        self.decoder = Decoder(in_channels, latent_dimension)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
