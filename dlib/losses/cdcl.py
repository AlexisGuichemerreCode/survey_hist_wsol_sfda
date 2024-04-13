from torch import nn
#from utils.utils import to_cuda
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def to_cuda(x):
    return x.cuda()

class Cdcl(torch.nn.Module):
    def __init__(self, tau, **kwargs):

        self.tau = tau

    def forward(self, zis, pseudo_labels,weights):

        self.weights = weights
        batch_size = len(pseudo_labels)
        
        features = F.normalize(zis, p=2, dim=1)
        weights_normalized = F.normalize(self.weights.to(features.device), p=2, dim=1)

        logits = torch.mm(features, weights_normalized.t()) / self.tau
        log_probs = F.log_softmax(logits, dim=1)

        loss = -log_probs[torch.arange(batch_size), pseudo_labels]

        return loss