import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return x * cdf
