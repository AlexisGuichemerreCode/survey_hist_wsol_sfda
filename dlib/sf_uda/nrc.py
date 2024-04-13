import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from math import ceil
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from typing import Dict, Iterable, Callable

from dlib.datasets.wsol_loader import get_data_loader
from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg



__all__ = ['Nrc']

class FeatureExtractor_for_source_code(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


class Nrc(object):
    def __init__(self,
                 model_trg,
                 train_loader_trg,
                 k_nearest_neighbors=2,
                 k_neighbors=2,
                 r_nrc=0.1,
                 ):
        

        self.model = model_trg
        self.train_loader_trg = train_loader_trg
        self.num_sample=len(train_loader_trg.dataset)
        self.fea_bank=torch.randn(self.num_sample,2048)
        self.score_bank = torch.randn(self.num_sample, 2).cuda()
        self.name=[]

        self.neighbor_k = k_neighbors
        self.neighbor_kk = k_nearest_neighbors
        self.r_nrc = r_nrc

        # Instantiate the feature extractor
        #layer_to_extract_feat = 'classification_head.avgpool'  # replace with the layers you want to extract features from
        #self.feature_extractor = FeatureExtractor_for_source_code(model=self.model, layers=[layer_to_extract_feat])
        # Initialize a dictionary with empty lists for each key
        self.dictionary = {'name': [], 'fea_bank': [], 'score_bank': []}
        
        with torch.no_grad():
            

            for id, (data, label, plabel, index, _, _, _, _) in enumerate(iter(self.train_loader_trg)):
                #feat =  self.feature_extractor(data.cuda())[layer_to_extract_feat].squeeze(2).squeeze(2)
                out = self.model(data.cuda())
                feat = self.model.lin_ft
                feat_normalized = F.normalize(feat, p=2, dim=1)
                model_output = self.model(data.cuda())
                softmax_output = F.softmax(model_output, dim=1)

                # Calculate start and end indices for the current batch
                start_index = id * data.size(0)
                end_index = start_index + data.size(0)

                # Update fea_bank and score_bank for the corresponding images
                self.fea_bank[start_index:end_index] = feat_normalized
                self.score_bank[start_index:end_index] = softmax_output
                self.name.extend(index)

        self.dictionary['name'] = self.name
        self.dictionary['fea_bank'] = self.fea_bank
        self.dictionary['score_bank'] = self.score_bank

        print('NRC: Feature bank and score bank are initialized')

    def update(self, model, images, index):
        self.model = model
        nb_imgs=len(index)
        # fea_bank= self.dictionary['fea_bank']
        # score_bank = self.dictionary['score_bank']


        tar_idx = torch.tensor([self.dictionary['name'].index(name) for name in index])
        
        # Instantiate the feature extractor
        #layer_to_extract_feat = 'classification_head.avgpool'
        #self.feature_extractor = FeatureExtractor_for_source_code(model=self.model, layers=[layer_to_extract_feat])

        model_output = self.model(images.cuda())
        softmax_out = F.softmax(model_output, dim=1)

        with torch.no_grad():
            out = self.model(images.cuda())
            feat = self.model.lin_ft
            output_f_norm = F.normalize(feat, p=2, dim=1)
            output_f_ = output_f_norm.cpu().detach().clone()

            self.dictionary['fea_bank'][tar_idx] = output_f_.detach().clone().cpu()
            self.dictionary['score_bank'][tar_idx] = softmax_out.detach().clone()

            fea_bank = self.dictionary['fea_bank']
            score_bank = self.dictionary['score_bank']

            
            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=self.neighbor_k+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]    #batch x K x C

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
            _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=self.neighbor_kk+1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:] # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(self.r_nrc))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    self.neighbor_kk)  # batch x K x M
            weight_kk = weight_kk.fill_(self.r_nrc)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                2)  # batch x KM x C

            score_self = score_bank[tar_idx]

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, self.neighbor_k * self.neighbor_kk ,-1)  # batch x C x 1
        
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, self.neighbor_k,-1)  # batch x K x C

        msoftmax = softmax_out.mean(dim=0)

        keys_args = {'output_re': output_re, 'score_near': score_near, 'score_near_kk': score_near_kk, 'weight_kk': weight_kk, 'weight': weight, 'score_self': score_self, 'softmax_out_un': softmax_out_un, 'msoftmax': msoftmax}

        return keys_args