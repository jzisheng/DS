from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels =1,out_channels=32,kernel_size=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=2)
        self.fc1 = nn.Linear(64, 128)        
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x).flatten()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
"""
autoencoder model
"""

from torch.utils.data import Dataset
class Autoencoder(torch.nn.Module):
    def __init__(self, num_features):
        num_hidden_1 = 4
        super(Autoencoder, self).__init__()        
        # Encoding layers
        self.linear_1 = torch.nn.Linear(num_features, 64)
        self.linear_2 = torch.nn.Linear(64,4)
        
        # Decoding layers
        self.linear_3 = torch.nn.Linear(4,64)                        
        self.linear_4 = torch.nn.Linear(64, num_features)
        pass
    
    def encoder(self,x):
        # Encoding
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        encoded = self.linear_2(encoded)
        encoded = F.leaky_relu(encoded)
        return encoded 
    
    def decoder(self,x):
        # Decoding
        logits = self.linear_3(x)
        logits = F.leaky_relu(logits)                
        logits = self.linear_4(logits)
        decoded = torch.sigmoid(logits)
        #decoded = F.relu(logits)
        return decoded

    def forward(self, x):
        x_hat = self.encoder(x)
        decoded_hat = self.decoder(x_hat)
        return decoded_hat
    
        

