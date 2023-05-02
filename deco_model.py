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
from model import LayerNorm

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

        # to keep track of zeroed indices
        self.active_mask = nn.Parameter(
            torch.Tensor([True for _ in range(len(self.S))]).bool(),
            requires_grad=False
        )
        
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        #  As the rank of the weight matrix goes down this should be faster
        US = self.U[:, self.active_mask] * self.S[self.active_mask]
        left_side = F.linear(x, (self.V[:, self.active_mask]).T)
        return F.linear(left_side, US, self.bias)
    
    @torch.no_grad()
    def update_mask(self):
        # update the prune mask
        # momentum from AdamW may have pushed some values beyond zero
        self.active_mask = (self.S != 0.0) & self.active_mask
        return
    
    @property
    def weight(self):
        # WARNING this is actually _slower_ than the default linear layer
        # you should use the forward pass instead
        return torch.mm(self.U * self.S, self.V.T)
    
    @torch.no_grad()
    def _remove_zeroed_columns(self):
        # removes the values of S that are zero and the corresponding columns in U,V
        self.U = nn.Parameter(self.U[:, self.active_mask], requires_grad=False)
        self.V = nn.Parameter(self.V[:, self.active_mask], requires_grad=False)
        self.S = nn.Parameter(self.S[self.active_mask])
        return
    
def replace_linear_with_svd(module, exceptions=()):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if name in exceptions:
                continue
            weight_matrix = child.weight.data
            bias = child.bias.data if child.bias is not None else None
            in_features, out_features = weight_matrix.shape[1], weight_matrix.shape[0]
            svd_linear_layer = SVDLinear(in_features, out_features, weight_matrix, bias)
            setattr(module, name, svd_linear_layer)
        else:
            replace_linear_with_svd(child)

def drop_zeroed_columns(module):
    for name, child in module.named_children():
        if isinstance(child, nn.SVDLinear):
            child._remove_zeroed_columns()
        else:
            drop_zeroed_columns(child)

def update_masks(module):
    for mn, m in module.named_modules():
        if isinstance(m, nn.SVDLinear):
            m.update_mask()

def configure_optimizers(module, weight_decay, learning_rate, betas, device_type):
    """
    This is an adaptation of the original code in nanoGPT.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    frozen = set()
    whitelist_weight_modules = (SVDLinear,)
    blacklist_weight_modules = ()
    frozen_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding, torch.nn.Linear)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will be frozen
                frozen.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif fpn.endswith('lm_head.S') and isinstance(m, whitelist_weight_modules):
                # we are going to ignore the last layer since it is tied to the embedding
                p.requires_grad = False
                frozen.add(fpn)
            elif pn.endswith('S') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('U') or pn.endswith('V') or pn.endswith('active_mask'):
                # these SVDLinear params are always frozen
                p.requires_grad = False
                frozen.add(fpn)
            elif isinstance(m, frozen_weight_modules):
                # weights of frozen modules will NOT be updated
                p.requires_grad = False
                frozen.add(fpn)

    # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
    # will appear in the no_decay and decay sets respectively after the above.
    # In addition, because named_parameters() doesn't return duplicates, it
    # will only return the first occurence, key'd by 'transformer.wte.weight', below.
    # so let's manually remove 'lm_head.weight' from decay set. This will include
    # this tensor into optimization via transformer.wte.weight only, and not decayed.
    # decay.remove('lm_head.weight')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay | frozen
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay/frozen set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    return optimizer