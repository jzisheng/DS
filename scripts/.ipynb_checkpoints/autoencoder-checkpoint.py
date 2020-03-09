import os
import torch
import torch.nn.functional as F
from torch import nn
"""
Building autoencoder network
"""

from torch.utils.data import Dataset
class Autoencoder(torch.nn.Module):
    def __init__(self, num_features):
        num_hidden_1 = 4
        super(Autoencoder, self).__init__()        
        # Encoding layers
        self.linear_1 = torch.nn.Linear(num_features, 64)
        self.linear_2 = torch.nn.Linear(64,2)
        
        # Decoding layers
        self.linear_3 = torch.nn.Linear(2,64)                        
        self.linear_4 = torch.nn.Linear(64, num_features)
        pass
    
    def encoder(self,x):
        # Encoding
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        encoded = self.linear_2(encoded)
        encoded = F.leaky_relu(encoded) 
    def decoder(self,x):
        # Decoding
        logits = self.linear_3(encoded)
        logits = F.leaky_relu(logits)                
        logits = self.linear_4(logits)        
        #decoded = torch.sigmoid(logits)
        decoded = F.relu(logits)                

    def forward(self, x):
        x_hat = self.encoder(x)
        decoded_hat = self.decoder(x)

        return decoded
    
    def encode(self,x):
        encoded = self.linear_1(x)
        return F.leaky_relu(encoded)
        
