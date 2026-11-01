# Last edited: 2026-10-29 18:50 UTC
import math

import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2 / (in_features + out_features))
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f'Hello the shape of self.W is {self.W.shape}')
        print(f'Hello the shape of x is {x.shape}')
        return torch.einsum("o i, b n i -> b n o", self.W, x)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
            super().__init__()
            self.num_embeddings = num_embeddings # Number of embeddings in the vocabulary
            self.embedding_dim = embedding_dim # d_model, the dimension of each embedding vector
    
            self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
            nn.init.trunc_normal_(self.W, mean=0.0, std=1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Logic for matching each token id to an embedding
        # Initialize an empty tensor of dimensions (d_model, token_ids)

        batch_size = token_ids.shape[0]
        n_queries = token_ids.shape[1]
        d_model = self.embedding_dim
        output_tensor = torch.empty(batch_size, n_queries, d_model)

        # For each token id, you just change that column of the output matrix to the corresponding column in the embedding matrix
        for i in range(batch_size):
            for j in range(n_queries):
                output_tensor[i][j] = torch.tensor(self.W[token_ids[i][j]])
        return output_tensor


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        rms_weights: torch.Tensor,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.rms_weights = nn.Parameter(rms_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reference loop-based implementation:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        batches, tokens, d_model = x.shape
        for b in range(batches):
            for tok in range(tokens):
                summation_value = 0
                for i in range(d_model):
                    summation_value += torch.pow(x[b][tok][i], 2)
                rms = torch.sqrt(self.eps + summation_value / self.d_model)
                for i in range(d_model):
                    x[b][tok][i] = (x[b][tok][i] / rms) * self.rms_weights[i]
        return x.to(in_dtype)

        
