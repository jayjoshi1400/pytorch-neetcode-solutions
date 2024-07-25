# https://neetcode.io/problems/handwritten-digit-classifier

import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        self.first_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.second_layer = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        first_out = self.first_layer(images)
        relu_out = self.relu(first_out)
        drop_out = self.dropout(relu_out)
        sec_out = self.second_layer(drop_out)
        sig_out = self.sigmoid(sec_out)
        return torch.round(sig_out, decimals=4)
        # Return the model's prediction to 4 decimal places
