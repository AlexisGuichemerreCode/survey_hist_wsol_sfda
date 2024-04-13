import torch
import torch.nn as nn
from functools import partial
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.model_zoo import load_url

import sys
from os.path import dirname, abspath, join

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.shared import count_params

# Implement: "Unsupervised Domain Adaptation by Backpropagation",
# https://arxiv.org/pdf/1409.7495.pdf
# Only: the domain discriminator. It can be used with standard feature
# extractor.


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, features_dim: int):
        super(DomainDiscriminator, self).__init__()

        assert isinstance(features_dim, int), type(features_dim)
        assert features_dim > 0, features_dim
        self.features_dim = features_dim

        self.model = nn.Sequential(
            nn.Linear(features_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=False),
            nn.Linear(100, 2)
        )

    def forward(self, features: torch.Tensor, alpha: float) -> torch.Tensor:
        assert features.ndim == 2, features.ndim  # bsz, m
        assert alpha >= 0, alpha
        reverse_feature = ReverseLayerF.apply(features, alpha)

        d_out = self.model(reverse_feature)
        return d_out


def run():
    from dlib.utils.reproducibility import set_seed
    from dlib.utils.shared import count_params

    set_seed(0)
    cuda = "0"
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else
                          "cpu")

    features_dim = 512
    bsz = 32
    alpha = 0.5
    d = DomainDiscriminator(features_dim=features_dim).to(device)

    print(f"Domain discriminator: {count_params(d)} params.")


    x = torch.randn((bsz, features_dim), device=device)
    with torch.no_grad():
        logits = d(x, alpha)

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")


if __name__ == "__main__":
    run()
