# https://neetcode.io/problems/gpt-dataset

import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        idx = torch.randint(low=0, high=len(tokens)-context_length, size=(batch_size,))
        x = []
        y = []
        for i in range(batch_size):
            x_temp = []
            y_temp = []
            ini_idx = idx[i]
            for j in range(context_length):
                x_temp.append(tokens[ini_idx])
                y_temp.append(tokens[ini_idx+1])
                ini_idx+=1
            x.append(x_temp)
            y.append(y_temp)
        return x, y
        
