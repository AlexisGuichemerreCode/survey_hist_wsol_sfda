import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple
from itertools import cycle

import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg

# Paper FAUST: "Feature Alignment by Uncertainty and Self-Training for
# Source-Free Unsupervised Domain Adaptation", Neural networks, 2023.
# https://arxiv.org/abs/2208.14888.


__all__ = ['Faust']


class Faust(object):
    def __init__(self, args, model_trg):

        self.model_trg = model_trg
        self.args = args

        assert args.task in [constants.STD_CL, constants.NEGEV], args.task

        self.c_epoch = 0

    def set_c_epoch(self, epoch: int):
        assert isinstance(epoch, int), type(epoch)
        self.c_epoch = epoch

    def if_on_views_ft_consist(self) -> bool:
        st = self.args.views_ft_consist
        st &= (self.c_epoch >= self.args.views_ft_consist_start_ep)

        if self.args.views_ft_consist_end_ep != -1:
            st &= (self.c_epoch <= self.args.views_ft_consist_end_ep)


        if self.args.views_ft_consist:
            assert self.args.faust
            assert self.args.faust_n_views > 0, self.args.faust_n_views

        return st

    def if_on_ce_views_soft_pl(self) -> bool:
        st = self.args.ce_views_soft_pl
        st &= (self.c_epoch >= self.args.ce_views_soft_pl_start_ep)

        if self.args.ce_views_soft_pl_end_ep != -1:
            st &= (self.c_epoch <= self.args.ce_views_soft_pl_end_ep)

        if self.args.ce_views_soft_pl:
            assert self.args.faust
            assert self.args.faust_n_views > 0, self.args.faust_n_views
            assert self.args.ce_views_soft_pl_t > 0, self.args.ce_views_soft_pl_t
            assert isinstance(self.args.ce_views_soft_pl_t, float), type(
                self.args.ce_views_soft_pl_t)

        return st

    def if_on_mc_var_prob(self) -> bool:
        st = self.args.mc_var_prob
        st &= (self.c_epoch >= self.args.mc_var_prob_start_ep)

        if self.args.mc_var_prob_end_ep != -1:
            st &= (self.c_epoch <= self.args.mc_var_prob_end_ep)

        if self.args.mc_var_prob:
            assert self.args.faust
            assert self.args.mc_var_prob_n_dout > 1, \
                self.args.mc_var_prob_n_dout
            assert self.args.model['spatial_dropout'] > 0, self.args.model[
                'spatial_dropout']

        return st

    def if_on_min_prob_entropy(self):
        st = self.args.min_prob_entropy
        st &= (self.c_epoch >= self.args.min_prob_entropy_start_ep)

        if self.args.min_prob_entropy_end_ep != -1:
            st &= (self.c_epoch <= self.args.min_prob_entropy_end_ep)

        if self.args.min_prob_entropy:
            assert self.args.faust

        return st

    def split_views(self, views: torch.Tensor) -> list:
        assert self.args.faust

        n = self.args.faust_n_views
        assert views.ndim == 5, views.ndim  # b, n+1 views, c, h, w; n>=0
        b, z, c, h, w = views.shape

        assert (z == (n + 1)) or (z == 1), f"{z} | {n + 1}"  # 1 non-augmented
        # + [optional: n  views]
        assert z >= 1, z  # non-ag, [optional: at least 1 random view].

        x = views.swapdims(0, 1)  # z, b, c, h, w
        l_views = [x[i] for i in range(z)]  # [non-aug, v0, v1, ...]

        assert len(l_views) >= 1, len(l_views)

        return l_views

    def forward_views(self, l_views: list) -> Tuple[list, list]:

        assert (self.if_on_views_ft_consist() or self.if_on_ce_views_soft_pl())

        l_cl_logits = []
        l_lin_ft = []

        for v in l_views:

            out = self.model_trg(v)
            if self.args.task == constants.STD_CL:
                cl_logits = out

            elif self.args.task in [constants.NEGEV]:
                cl_logits, fcams, im_recon = out

            else:
                raise NotImplementedError

            l_cl_logits.append(cl_logits)
            l_lin_ft.append(self.model_trg.lin_ft)

        return l_cl_logits, l_lin_ft

    def perform_mc_dropout(self, clean_img: torch.Tensor) -> torch.Tensor:

        assert self.if_on_mc_var_prob()

        assert clean_img.ndim == 4, clean_img.ndim  # b, c, h, w

        n = self.args.mc_var_prob_n_dout
        assert n > 1, n

        n_cls = self.args.num_classes

        tmp = []

        for i in range(n):
            out = self.model_trg(clean_img)
            if self.args.task == constants.STD_CL:
                cl_logits = out

            elif self.args.task in [constants.NEGEV]:
                cl_logits, fcams, im_recon = out

            else:
                raise NotImplementedError

            assert cl_logits.ndim == 2, cl_logits.ndim  # b, cls
            assert cl_logits.shape[1] == n_cls, f"{cl_logits.shape[1]} | " \
                                                f"{n_cls}"
            prob = torch.softmax(cl_logits, dim=1)

            tmp.append(prob)

        holder = torch.stack(tmp, dim=0)  # n, b, ncls
        holder = holder.swapdims(0, 1)  # b, n, cls
        std = torch.std(holder, dim=1, correction=0, keepdim=False)  # b, cls
        assert std.ndim == 2, std.ndim  # b, cls

        return std

    def perform_clean_forward(self, clean_img: torch.Tensor) -> torch.Tensor:

        assert clean_img.ndim == 4, clean_img.ndim  # b, c, h, w

        out = self.model_trg(clean_img)

        if self.args.task == constants.STD_CL:
            cl_logits = out

        elif self.args.task in [constants.NEGEV]:
            cl_logits, fcams, im_recon = out

        else:
            raise NotImplementedError

        return cl_logits

    def cosine_sim(self,
                   embeds: torch.Tensor,
                   protos: torch.Tensor) -> torch.Tensor:

        assert embeds.ndim == 2, embeds.ndim  # m, d
        assert protos.ndim == 2, protos.ndim  # n, d
        assert embeds.shape[1] == protos.shape[1], f"{embeds.shape[1]} | " \
                                                   f"{protos.shape[1]}"

        c_sim = F.cosine_similarity(embeds[:, :, None],
                                    protos.t()[None, :, :], dim=1)  # m, n

        assert c_sim.ndim == 2, c_sim.ndim
        assert c_sim.shape[0] == embeds.shape[0], f"{c_sim.shape[0]} | " \
                                                  f"{embeds.shape[0]}"
        assert c_sim.shape[1] == protos.shape[0], f"{c_sim.shape[1]} | " \
                                                  f"{protos.shape[0]}"

        return c_sim

    def build_cl_soft_labels(self,
                             clean_ft: torch.Tensor,
                             clean_cl_logits: torch.Tensor
                             ) -> torch.Tensor:

        assert self.if_on_ce_views_soft_pl()

        assert clean_ft.ndim == 2, clean_ft.ndim  # b, ft
        assert clean_cl_logits.ndim == 2, clean_cl_logits.ndim  # b, nlcs
        
        assert clean_ft.shape[0] == clean_cl_logits.shape[0]

        ncls = clean_cl_logits.shape[1]
        b = clean_cl_logits.shape[0]

        assert ncls == self.args.num_classes, f"{ncls} |" \
                                              f" {self.args.num_classes}"


        probs = torch.softmax(clean_cl_logits, dim=1)  # b, ncls

        cl_prototypes = probs.transpose(0, 1).matmul(clean_ft)  # cl, ft
        msg = f"{cl_prototypes.shape} | {(ncls, clean_ft.shape[1])}"
        assert cl_prototypes.shape == (ncls, clean_ft.shape[1]), msg

        c_sim = self.cosine_sim(clean_ft, cl_prototypes)  # b, cls
        assert c_sim.shape == (b, ncls), f"{c_sim.shape} | {(b, ncls)}"

        # heatup
        t = self.args.ce_views_soft_pl_t
        assert isinstance(t, float), type(t)
        assert t > 0, t

        logits_soft_label: torch.Tensor = c_sim / t

        return logits_soft_label

    def forward_data(self, views: torch.Tensor) -> dict:
        # model must be in train mode.

        assert self.args.faust

        results = {
            'faust_views_cl_logits': None,
            'faust_views_lin_ft': None,
            'faust_views_soft_pl': None,
            'mc_var_prob_std': None,
            'min_prob_entropy_logits': None
        }

        l_views = self.split_views(views)  # [non-aug, v0, v1], vi: optional

        # 1. views_ft_consist / ce_views_soft_pl:
        if self.if_on_views_ft_consist() or self.if_on_ce_views_soft_pl():
            l_cl_logits, l_lin_ft = self.forward_views(l_views)

            results['faust_views_cl_logits'] = l_cl_logits  # list of tensors
            # of shape: b, ncls
            results['faust_views_lin_ft'] = l_lin_ft  # list of tensors of
            # sahape: b, ft

            if self.if_on_ce_views_soft_pl():
                logits_soft_label = self.build_cl_soft_labels(
                    clean_ft=l_lin_ft[0], clean_cl_logits=l_cl_logits[0])  #
                # b, ncls

                results['faust_views_soft_pl'] = logits_soft_label

        # 2. mc_var_prob:
        if self.if_on_mc_var_prob():
            std = self.perform_mc_dropout(clean_img=l_views[0])  # b, ncls

            results['mc_var_prob_std'] = std

        # 3- min_prob_entropy
        if self.if_on_min_prob_entropy():
            cl_logits = self.perform_clean_forward(clean_img=l_views[0])  #
            # b, ncls

            results['min_prob_entropy_logits'] = cl_logits

        return results
