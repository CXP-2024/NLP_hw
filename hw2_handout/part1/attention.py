import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        ## TODO: Implement causal self attention
        # obtain the key, query, and value projections to multiple heads: B, T, n_head, C // n_head
        k = self.key(x).view(B, T, self.n_head, C // self.n_head)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head)

        # transpose the q to B, n_head, T, C // n_head, then compute q @ k^T: B, n_head, T, T
        q = q.transpose(1, 2)  # B, n_head, T, C // n_head
        k = k.transpose(1, 2)  # B, n_head, T, C // n_head
        v = v.transpose(1, 2)  # B, n_head, T, C // n_head

        # compute attention scores: B, n_head, T, T
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # norm by the dimension of the key vectors 
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  
        att = self.attn_drop(att)

        # apply attention to values
        y = att @ v # B, n_head, T, C // n_head
        y = y.transpose(1, 2).contiguous().view(B, T, C) # B, T, C

        #  output projection
        y = self.resid_drop(self.proj(y)) 
        ### TODO END
        
        return y
