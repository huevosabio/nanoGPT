"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class SVDLinear(nn.Module):
    """
    SVD-decomposed linear layer.
    Only the eigenvalues are set as parameters.
    """
    def __init__(self, in_features, out_features, weight_matrix, bias=None):
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Perform SVD decomposition on the provided weight matrix
        U, S, V = torch.svd(weight_matrix)
        self.U = nn.Parameter(U, requires_grad=False)
        self.S = nn.Parameter(S)
        self.V = nn.Parameter(V, requires_grad=False)
        
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        US = self.U * self.S
        weight_matrix = torch.matmul(US, self.V.T)
        return F.linear(x, weight_matrix, self.bias)

def replace_linear_with_svd(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            weight_matrix = child.weight.data
            bias = child.bias.data if child.bias is not None else None
            in_features, out_features = weight_matrix.shape[1], weight_matrix.shape[0]
            svd_linear_layer = SVDLinear(in_features, out_features, weight_matrix, bias)
            setattr(module, name, svd_linear_layer)
        else:
            replace_linear_with_svd(child)