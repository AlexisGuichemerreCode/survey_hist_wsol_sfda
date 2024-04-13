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

# Paper: 'Do We Really Need to Access the Source Data? Source Hypothesis
# Transfer for Unsupervised Domain Adaptation', 2021,
# https://arxiv.org/abs/2002.08546


__all__ = ['Shot']


class Shot(object):
    def __init__(self,
                 model_trg,
                 train_loader_trg,
                 task: str,
                 n_cls: int,
                 shot_freq_epoch: int = 1,
                 shot_dist: str = constants.SHOT_COSINE
                 ):

        self.model_trg = model_trg
        self.train_loader_trg = train_loader_trg  # eval mode.

        assert isinstance(shot_freq_epoch, int), type(shot_freq_epoch)
        assert shot_freq_epoch > 0, shot_freq_epoch
        self.shot_freq_epoch = shot_freq_epoch

        assert task in [constants.STD_CL, constants.NEGEV], task
        self.task = task
        assert shot_dist in constants.SHOT_DISTS, shot_dist
        self.shot_dist = shot_dist

        assert isinstance(n_cls, int), type(n_cls)
        assert n_cls > 0, n_cls
        self.n_cls = n_cls

    def pairwise_dist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        assert x1.ndim == 2, x1.ndim  # m, d
        assert x2.ndim == 2, x2.ndim  # n, d
        assert x1.shape[1] == x2.shape[1], f"{x1.shape[1]} | {x2.shape[1]}"

        if self.shot_dist == constants.SHOT_EUCL:
            dist = torch.cdist(x1, x2, p=2.)  # m, n

        elif self.shot_dist == constants.SHOT_COSINE:
            dist = 1. - F.cosine_similarity(x1[:, :, None], x2.t()[None, :,
                                                            :], dim=1)  # m, n

        else:
            raise NotImplementedError(self.shot_dist)

        assert dist.ndim == 2, dist.ndim
        assert dist.shape[0] == x1.shape[0], f"{dist.shape[0]} | {x1.shape[0]}"
        assert dist.shape[1] == x2.shape[0], f"{dist.shape[1]} | {x2.shape[0]}"

        return dist

    def _update_img_cls_pseudo_lbs(self) -> dict:

        self.model_trg.eval()

        loader = self.train_loader_trg
        lin_features = None
        out_cl_logits = None
        all_image_ids = []
        all_img_lbs = None

        for i, (images, targets, _, image_ids, _, _, _, _) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()  # todo: track targets to MEASURE acc of
            # pseudo-labels.
            out = self.model_trg(images)
            if self.task == constants.STD_CL:
                cl_logits = out

            elif self.task in [constants.NEGEV]:
                cl_logits, fcams, im_recon = out

            else:
                raise NotImplementedError

            lin_ft = self.model_trg.lin_ft

            assert lin_ft is not None
            assert lin_ft.ndim == 2, lin_ft.ndim  # bsize, ft_dim

            if lin_features is None:
                lin_features = lin_ft
                out_cl_logits = cl_logits

                # todo: for tracking only.
                all_img_lbs = targets

            else:
                lin_features = torch.cat((lin_features, lin_ft), dim=0)
                out_cl_logits = torch.cat((out_cl_logits, cl_logits), dim=0)

                # todo: for tracking only.
                all_img_lbs = torch.cat((all_img_lbs, targets), dim=0)

            all_image_ids.extend(image_ids)

        probs = torch.softmax(out_cl_logits, dim=1)  # data_size, n_cls
        k = probs.shape[1]  # number cls.
        assert k == self.n_cls, f"{k} | {self.n_cls}"
        assign = probs
        predict = torch.argmax(probs, dim=1, keepdim=False)

        # update centers/pseudo-labels several times (twice seems to work for
        # the paper SHOT.
        # https://github.com/tim-learn/SHOT/blob/07d0c713e4882e83fded1aff2a447dff77856d64/object/image_target.py#L277
        for i in range(2):
            # update centers.
            initc = torch.matmul(assign.transpose(0, 1), lin_features)  #
            # n_cls, ft_dim
            z = assign.sum(axis=0)  # 1, n_cls
            z = z.view(-1, 1)  # n_cls, 1
            initc = initc / (1e-8 + z)

            # assign labels
            dist = self.pairwise_dist(lin_features, initc)  # m: data_size, n: k
            predict = torch.argmin(dist, dim=1)
            bin_cls = torch.eye(k, device=predict.device)  # k, k

            assign = bin_cls[predict]  # data_size, n_cls

            acc = (predict == all_img_lbs).float().mean() * 100.
            msg = f"SHOT - ACC pseudo-label image-class -- iter: {i}: {acc} %"
            DLLogger.log(fmsg(msg))


        # todo: for tracking only.
        acc = (predict == all_img_lbs).float().mean() * 100.

        msg = f"SHOT - ACC pseudo-label image-class: {acc} %"
        DLLogger.log(fmsg(msg))

        out = dict()
        predict = predict.long().detach().cpu().numpy()

        for i, img_id in enumerate(all_image_ids):
            out[img_id] = predict[i]

        return out

    def update_img_cls_pseudo_lbs(self) -> dict:

        with torch.no_grad():
            return self._update_img_cls_pseudo_lbs()
