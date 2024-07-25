# https://neetcode.io/problems/sentiment-analysis

import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.first_l = nn.Embedding(vocabulary_size, 16)
        # print('emb:')
        # print(self.first_l.weight)
        self.sec_l = nn.Linear(16, 1)
        self.out_l = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        # print(x.shape)
        first_out = self.first_l(x)
        # print(first_out)
        # print(first_out.shape)
        avg_emb = torch.mean(first_out, axis=1)
        # print(avg_emb.shape)
        sec_out = self.sec_l(avg_emb)
        sig_out = self.out_l(sec_out)
        return torch.round(sig_out, decimals=4)

        # Return a B, 1 tensor and round to 4 decimal places

