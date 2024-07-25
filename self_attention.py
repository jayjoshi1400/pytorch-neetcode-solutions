# https://neetcode.io/problems/self-attention

import torch
import torch.nn as nn
from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Query is what the token wants
        # Key is what info token has
        # Value is what info token wants to share
        self.key_l = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_l = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.val_l = nn.Linear(embedding_dim, attention_dim, bias=False)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Return your answer to 4 decimal places
        key_out = self.key_l(embedded)
        query_out = self.query_l(embedded)
        val_out = self.val_l(embedded)

        atten_score = torch.matmul(query_out, torch.transpose(key_out, 1, 2))
        context_len, atten_dim = key_out.shape[1], key_out.shape[2]
        atten_score = atten_score/(atten_dim**0.5)

        pre_mask = torch.tril(torch.ones(context_len, context_len))
        mask = pre_mask == 0
        atten_score = atten_score.masked_fill(mask, float('-inf'))
        # e^-inf = 1/e^inf = 1/inf = 0
        atten_score = nn.functional.softmax(atten_score, dim=2)
        
        res = torch.matmul(atten_score, val_out)

        return torch.round(res, decimals=4)
