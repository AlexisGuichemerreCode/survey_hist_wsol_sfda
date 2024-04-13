from torch import nn
#from utils.utils import to_cuda
import torch
import torch.nn.functional as F

class NA(torch.nn.Module):
    def forward(self, output_re, score_near_kk, weight_kk):
        const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1)*weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        neig_aff = torch.mean(const)
        return neig_aff
    
class ENA(torch.nn.Module):
    def forward(self, softmax_out_un, score_near,weight):
        ext_neig_aff = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)*weight.cuda()).sum(1))
        return ext_neig_aff

class KL(torch.nn.Module):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def forward(self, msoftmax,weight_kk):
        gentropy = torch.sum(msoftmax*torch.log(msoftmax + self.epsilon))
        return gentropy
    
