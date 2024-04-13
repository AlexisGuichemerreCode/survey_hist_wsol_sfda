import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple
from itertools import cycle
from copy import deepcopy
from os.path import join

import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.transforms.functional import rgb_to_grayscale

import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.datasets.wsol_loader import get_data_loader

from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg

# Paper AdaDSA: "Unsupervised Domain Adaptation by Statistics Alignment
# for Deep Sleep Staging Networks", IEEE Transactions on Neural Systems and
# Rehabilitation Engineering, 2022.
# https://ieeexplore.ieee.org/document/9684410.


__all__ = ['freeze_all_params',
           'adadsa_freeze_all_model_except_bn_a',
           'AdadsaFusedBn',
           'replace_all_bn_with_adadsa_bn',
           'AdadsaEstimateTrgBnStats',
           'Adadsa'
           ]


def adadsa_freeze_all_model_except_bn_a(model):
    """
    Set only BN alpha scalar to be learnable. Freeze everything else.

    :param model:
    :return:
    """
    for module in (model.modules()):

        assert not isinstance(
            module,
            (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d,
             torch.nn.BatchNorm1d)), type(module)

        for param in module.parameters():
            param.requires_grad = False

        if isinstance(module, torch.nn.Dropout):
            module.eval()

        if isinstance(module, AdadsaFusedBn):
            for name, param in module.named_parameters():
                if name == '_alpha':
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    return model


def freeze_all_params(model):
    for module in (model.modules()):

        for param in module.parameters():
            param.requires_grad = False

        if isinstance(module, torch.nn.BatchNorm3d):
            module.eval()

        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()

        if isinstance(module, torch.nn.Dropout):
            module.eval()


class AdadsaFusedBn(nn.Module):
    """
    A fixed Batchnormxd, x in {1, 2, 3}. It fuses the running stats of 2
    batchnormxd modules. It has only a single scalar as a learnable parameter
    to be trained using SGD.

    μ_{ts} = αμ_t + (1 − α)μ_s
    σ2_{ts} = α(σ2_t + (μ_t −μ_{ts})^2) + (1 − α)(σ2_s + (μ_s −μ_{ts})^2)

    alpha (a) is the only learnable parameter of this module.
    See https://ieeexplore.ieee.org/document/9684410.
    All standard parameters (affine params) are frozen using the provided ones.
    source and 'target' running stats are fixed.
    The running stats of this module change only in function of alpha (a).

    WARNING:
        1. STANDARD BATCHNORNXD MAY HAVE PARAMETERS (W, B) WHICH ARE 2
        VECTORS. ONCE FUSED USING THIS CLASS, THEY ARE NO LONGER LEARNABLE (
        REQUIRE_GRAD = FALSE). IN ADDITION, THIS CLASS WILL CREATE A NEW SCALAR
        PARAMETER TO BE LEARNABLE. IT ADDS MORE NON-LEARNABLE PARAMETERS
        BASED ON SOURCE/TARGET STATS (MEAN, VAR).
    """
    def __init__(self,
                 bn_dim: int,
                 s_mean: torch.Tensor,
                 s_var: torch.Tensor,
                 t_mean: torch.Tensor,
                 t_var: torch.Tensor,
                 eps: float = 1e-5,
                 affine: bool = False,
                 weight: torch.Tensor = None,
                 bias: torch.Tensor = None
                 ):
        super(AdadsaFusedBn, self).__init__()

        assert bn_dim in [1, 2, 3], f"Unsupported BN dim: {bn_dim}"
        self.bn_dim = bn_dim

        self._alpha = Parameter(data=torch.tensor([1.]))  # init a = 1.

        self.n_features: int = s_mean.numel()

        for i, v in enumerate([s_mean, s_var, t_mean, t_var]):
            assert isinstance(v, torch.Tensor),f"{i}: {type(v)}"
            assert v.ndim == 1, f"{i}: {v.ndim}"
            assert v.numel() == self.n_features, f"{i}: {v.numel()} | " \
                                                 f"{self.n_features}"

        # Stored as params. to be stats_dict loaded.
        self.s_mean = Parameter(data=s_mean, requires_grad=False)
        self.s_var = Parameter(data=s_var, requires_grad=False)
        self.t_mean = Parameter(data=t_mean, requires_grad=False)
        self.t_var = Parameter(data=t_var, requires_grad=False)

        assert eps > 0, eps
        assert isinstance(eps, float), type(eps)
        self.eps = eps
        self.affine = affine

        if affine:
            assert isinstance(weight, torch.Tensor), type(weight)
            assert weight.shape == s_mean.shape, f"{weight.shape} | " \
                                                 f"{s_mean.shape}"

            assert isinstance(bias, torch.Tensor), type(bias)
            assert bias.shape == s_mean.shape, f"{bias.shape} | " \
                                                 f"{s_mean.shape}"


            # Set as parameters to be stored/loaded with model.
            # todo: deal with store/load of this class.

            self.weight = Parameter(data=weight, requires_grad=False)
            self.bias = Parameter(data=bias, requires_grad=False)

        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @property
    def alpha(self):
        return torch.clamp(self._alpha, 0.0, 1.0)

    @property
    def ts_mean(self):
        return self.alpha * self.t_mean + (1. - self.alpha) * self.s_mean

    @property
    def ts_var(self):
        r = self.alpha * (self.t_var + (self.t_mean - self.ts_mean)**2)
        r = r + (1. - self.alpha) * (self.s_var +
                                     (self.s_mean - self.ts_mean)**2)

        return r

    def check_input(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), type(x)

        if self.bn_dim == 1:
            dim = 2

        elif self.bn_dim == 2:
            dim = 4

        elif self.bn_dim == 3:
            dim = 5

        else:
            raise NotImplementedError(self.bn_dim)

        assert x.ndim == dim, f"{x.ndim} | {dim}"

    def expand_tensor(self, v: torch.Tensor) -> torch.Tensor:

        assert v.ndim == 1, v.ndim


        if self.bn_dim == 1:
            return v[None, :]

        elif self.bn_dim == 2:
            return v[None, :, None, None]

        elif self.bn_dim == 3:
            return v[None, :, None, None, None]

        else:
            raise NotImplementedError(v.ndim)


    def forward(self, x: torch.Tensor):

        self.check_input(x)

        mean = self.expand_tensor(self.ts_mean)
        var  = self.expand_tensor(self.ts_var)

        out = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            w = self.expand_tensor(self.weight)
            b = self.expand_tensor(self.bias)
            out = out * w + b

        return out

    def __str__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"(bn_dim={self.bn_dim}, " \
               f"eps={self.eps}, " \
               f"affine={self.affine}" \
               f")"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"(bn_dim={self.bn_dim}, " \
               f"eps={self.eps}, " \
               f"affine={self.affine}" \
               f")"


def replace_all_bn_with_adadsa_bn(model,
                                  s_model,
                                  t_model,
                                  batch_norm_cl,
                                  device
                                  ):

    assert batch_norm_cl in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],\
        batch_norm_cl

    if batch_norm_cl == nn.BatchNorm1d:
        bn_dim = 1

    elif batch_norm_cl == nn.BatchNorm2d:
        bn_dim = 2

    elif batch_norm_cl == nn.BatchNorm3d:
        bn_dim = 3

    else:
        raise NotImplementedError(batch_norm_cl)

    for m, s, t in zip(model.named_children(),
                       s_model.named_children(),
                       t_model.named_children()
                       ):

        n, module = m
        n_s, s_module = s
        n_t, t_module = t

        assert n == n_s, f"{n} | {n_s}"
        assert n == n_t, f"{n} | {n_t}"

        if len(list(module.children())) > 0:
            replace_all_bn_with_adadsa_bn(module,
                                          s_module,
                                          t_module,
                                          batch_norm_cl,
                                          device
                                          )

        if isinstance(module, batch_norm_cl):
            src = s_module
            trg = t_module

            affine = trg.affine
            weight = None
            bias = None
            if affine:
                weight = torch.clone(trg.weight.data.detach())
                bias = torch.clone(trg.bias.data.detach())

            setattr(model,
                    n,
                    AdadsaFusedBn(bn_dim=bn_dim,
                                  s_mean=torch.clone(src.running_mean.detach()),
                                  s_var=torch.clone(src.running_var.detach()),
                                  t_mean=torch.clone(trg.running_mean.detach()),
                                  t_var=torch.clone(trg.running_var.detach()),
                                  eps=trg.eps,
                                  affine=trg.affine,
                                  weight=weight,
                                  bias=bias
                                  ).to(device)
                    )


class AdadsaEstimateTrgBnStats(object):
    def __init__(self, args, model_src):
        self.args = args
        self.model_src = model_src

        assert args.sf_uda
        assert args.adadsa

        bsz = args.adadsa_eval_batch_size

        assert (bsz > 0) or (bsz == -1), bsz

        if bsz == -1:
            bsz = self.get_number_of_samples_subset(split=constants.TRAINSET)

        loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            eval_batch_size=bsz,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            load_tr_masks=False,
            mask_root='',
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=None,
            get_splits_eval=[constants.TRAINSET],
            sfuda_faust=False,
            sfuda_n_rnd_views=0
        )

        self.train_eval_loader = loaders[constants.TRAINSET]

    def get_number_of_samples_subset(self, split: str) -> int:
        metadata_root = join(self.args.metadata_root, split)
        pth_image_ids = join(metadata_root, 'image_ids.txt')

        image_ids = []
        with open(pth_image_ids, 'r') as f:
            for line in f.readlines():
                image_ids.append(line.strip('\n'))

        return len(image_ids)


    def reset_bn_running_stats(self):

        for module in (self.model_src.modules()):

            if isinstance(module, (torch.nn.BatchNorm3d,
                                   torch.nn.BatchNorm2d,
                                   torch.nn.BatchNorm1d)):
                module.reset_running_stats()


    def estimate_bn_stats(self):
        self.freeze_all_params_keep_bn_stats_updatable(self.model_src)
        self.reset_bn_running_stats()  # clean up src stats. start over for
        # new stats over trg.

        loader = self.train_eval_loader

        with torch.no_grad():
            for i, (images, _, _, _, _, _, _, _) in enumerate(loader):
                images = images.cuda()
                self.model_src(images)

        return self.model_src

    def freeze_all_params(self, model):

        for module in (model.modules()):

            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm3d):
                module.eval()

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.BatchNorm1d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def set_all_bn_running_stats_to_be_updatable(self, model):
        for module in (model.modules()):

            if isinstance(module, torch.nn.BatchNorm3d):
                module.train()

            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()

            if isinstance(module, torch.nn.BatchNorm1d):
                module.train()

    def freeze_all_params_keep_bn_stats_updatable(self, model):
        """
        Utility: AdaDSA method for SFUDA method.
        Freeze all parameters.
        :return:
        """
        self.freeze_all_params(model)
        self.set_all_bn_running_stats_to_be_updatable(model)


class Adadsa(object):
    def __init__(self, args, model_src, model_trg):

        self.args = args
        assert args.sf_uda
        assert args.adadsa
        assert isinstance(args.adadsa_a, float), type(args.adadsa_a)
        assert args.adadsa_a > 0, args.adadsa_a

        self.model_src = model_src
        self.model_src.eval()

        self.model_trg = model_trg

        loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,  # use eval_batch_size
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            load_tr_masks=False,
            mask_root='',
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=None,
            get_splits_eval=[constants.TRAINSET],
            sfuda_faust=False,
            sfuda_n_rnd_views=0
        )

        self.train_eval_loader = loaders[constants.TRAINSET]

        # lambda
        self.nbr_train_samples = self.get_number_of_samples_subset(
            constants.TRAINSET)
        self.n_mbatches = max(1, int(self.nbr_train_samples / args.batch_size))
        self.total_steps = args.max_epochs * self.n_mbatches
        self.lambda_ = 0.

    def get_number_of_samples_subset(self, split: str) -> int:
        metadata_root = join(self.args.metadata_root, split)
        pth_image_ids = join(metadata_root, 'image_ids.txt')

        image_ids = []
        with open(pth_image_ids, 'r') as f:
            for line in f.readlines():
                image_ids.append(line.strip('\n'))

        return len(image_ids)

    def set_lambda_(self, v: float):
        assert 0 <= v <= 1., v
        self.lambda_ = v

    def update_lambda_(self, epoch: int, mb_idx: int):
        a = self.args.adadsa_a
        start_steps = self.n_mbatches * epoch

        p = float(mb_idx + start_steps) / self.total_steps
        v = (2. / (1. + np.exp(-a * p))) - 1.

        v = min(max(0., v), 1.)

        self.set_lambda_(v)

    def _update_img_cls_pseudo_lbs(self) -> dict:
        """
        Pseudo-label entire trainset at once.
        """

        self.model_trg.eval()
        self.model_src.eval()

        loader = self.train_eval_loader
        all_image_ids = []
        all_img_lbs = None
        all_plbs = None

        for i, (images, targets, _, image_ids, _, _, _, _) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()  # todo: track targets to MEASURE acc of
            # pseudo-labels.
            out_trg = self.model_trg(images)
            out_src = self.model_src(images)
            if self.args.task == constants.STD_CL:
                cl_logits_trg = out_trg
                cl_logits_src = out_src

            elif self.args.task in [constants.NEGEV]:
                cl_logits_trg, fcams, im_recon = out_trg
                cl_logits_src, fcams, im_recon = out_src

            else:
                raise NotImplementedError

            prob_src = torch.softmax(cl_logits_src, dim=1)
            prob_trg = torch.softmax(cl_logits_trg, dim=1)

            com_prob = (1. - self.lambda_) * prob_src + self.lambda_ * prob_trg
            plbs = torch.argmax(com_prob, dim=1, keepdim=False)
            assert plbs.ndim == 1, plbs.ndim  # bsz

            if all_plbs is None:
                all_plbs = plbs

                # todo: for tracking only.
                all_img_lbs = targets

            else:
                all_plbs = torch.cat((all_plbs, plbs), dim=0)

                # todo: for tracking only.
                all_img_lbs = torch.cat((all_img_lbs, targets), dim=0)

            all_image_ids.extend(image_ids)

        # todo: for tracking only.
        acc = (all_plbs == all_img_lbs).float().mean() * 100.

        msg = f"AdaDSA - ACC pseudo-label image-class: {acc} %"
        DLLogger.log(fmsg(msg))

        out = dict()
        all_plbs = all_plbs.long().detach().cpu().numpy()

        for i, img_id in enumerate(all_image_ids):
            out[img_id] = all_plbs[i]

        return out

    def _pseudo_label_imgs(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pseudo-label some images.
        :param images: torch.Tensor. images of shape bsz, c, h, w.
        :return: long torch.Tensor of image class pseudo-labels.

        Note: it is recommended to pseudo-label images every SGD step. This
        function is recommended to yield on the fly pseudo-labels.
        """

        assert isinstance(images, torch.Tensor), type(images)
        assert images.ndim == 4, images.ndim  # bsz, c, h, w

        self.model_trg.eval()
        self.model_src.eval()

        out_trg = self.model_trg(images)
        out_src = self.model_src(images)

        if self.args.task == constants.STD_CL:
            cl_logits_trg = out_trg
            cl_logits_src = out_src

        elif self.args.task in [constants.NEGEV]:
            cl_logits_trg, fcams, im_recon = out_trg
            cl_logits_src, fcams, im_recon = out_src

        else:
            raise NotImplementedError

        prob_src = torch.softmax(cl_logits_src, dim=1)
        prob_trg = torch.softmax(cl_logits_trg, dim=1)

        com_prob = (1. - self.lambda_) * prob_src + self.lambda_ * prob_trg
        plbs = torch.argmax(com_prob, dim=1, keepdim=False)
        assert plbs.ndim == 1, plbs.ndim  # bsz

        plbs = plbs.long().detach()

        return plbs


    def update_img_cls_pseudo_lbs(self) -> dict:

        with torch.no_grad():
            return self._update_img_cls_pseudo_lbs()

    def pseudo_label_imgs(self, images: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            return self._pseudo_label_imgs(images)


def run_replace_all_bn_with_adadsa_bn():
    # see: dlib.std.classifier.py
    pass

def run_demo_AdadsaFusedBn():
    nf1 = 10
    nf2 = 20
    nf3 = 30
    std_1dbn = torch.nn.BatchNorm1d(num_features=nf1, eps=1e-5, momentum=0.1,
                                    affine=True, track_running_stats=True
                                    ).cuda()
    std_2dbn = torch.nn.BatchNorm2d(num_features=nf2, eps=1e-5, momentum=0.1,
                                    affine=True, track_running_stats=True).cuda()
    std_3dbn = torch.nn.BatchNorm3d(num_features=nf3, eps=1e-5, momentum=0.1,
                                    affine=True, track_running_stats=True).cuda()

    print(f"1D - w.shape: {std_1dbn.weight.shape}")
    print(f"2D - w.shape: {std_2dbn.weight.shape}")
    print(f"3D - w.shape: {std_3dbn.weight.shape}")

    # d = 1
    print(f"---- DIM = 1 ----")

    c_std_1dbn = AdadsaFusedBn(bn_dim=1,
                               s_mean=torch.clone(
                                   std_1dbn.running_mean.detach()),
                               s_var=torch.clone(
                                   std_1dbn.running_var.detach()),
                               t_mean=torch.clone(
                                   std_1dbn.running_mean.detach()),
                               t_var=torch.clone(
                                   std_1dbn.running_var.detach()),
                               eps=1e-5,
                               affine=True,
                               weight=torch.clone(
                                   std_1dbn.weight.data.detach()),
                               bias=torch.clone(
                                   std_1dbn.bias.data.detach())
                               ).cuda()


    x1 = torch.rand(32, nf1).cuda()
    out_x1 = c_std_1dbn(x1)
    print(x1.shape, out_x1.shape)
    c_std_1dbn._alpha.data = torch.tensor([0.5]).cuda()
    out_x1 = c_std_1dbn(x1)

    print('dim = 1', x1.shape, out_x1.shape)

    for p in c_std_1dbn.parameters():
        print(p, p.requires_grad)

    print(f"---- DIM = 1 ---- END.")

    # d = 2
    print(f"---- DIM = 2 ----")
    c_std_2dbn = AdadsaFusedBn(bn_dim=2,
                               s_mean=torch.clone(
                                   std_2dbn.running_mean.detach()),
                               s_var=torch.clone(
                                   std_2dbn.running_var.detach()),
                               t_mean=torch.clone(
                                   std_2dbn.running_mean.detach()),
                               t_var=torch.clone(
                                   std_2dbn.running_var.detach()),
                               eps=1e-5,
                               affine=True,
                               weight=torch.clone(
                                   std_2dbn.weight.data.detach()),
                               bias=torch.clone(
                                   std_2dbn.bias.data.detach())
                               ).cuda()

    x2 = torch.rand(32, nf2, 32, 32).cuda()
    out_x2 = c_std_2dbn(x2)
    print(x2.shape, out_x2.shape)
    c_std_2dbn._alpha.data = torch.tensor([0.5]).cuda()
    out_x2 = c_std_2dbn(x2)

    print('dim = 2', x2.shape, out_x2.shape)

    for p in c_std_2dbn.parameters():
        print(p, p.requires_grad)

    print(f"---- DIM = 2 ---- END.")

    # d = 3
    print(f"---- DIM = 3 ----")
    c_std_3dbn = AdadsaFusedBn(bn_dim=3,
                               s_mean=torch.clone(
                                   std_3dbn.running_mean.detach()),
                               s_var=torch.clone(
                                   std_3dbn.running_var.detach()),
                               t_mean=torch.clone(
                                   std_3dbn.running_mean.detach()),
                               t_var=torch.clone(
                                   std_3dbn.running_var.detach()),
                               eps=1e-5,
                               affine=True,
                               weight=torch.clone(
                                   std_3dbn.weight.data.detach()),
                               bias=torch.clone(
                                   std_3dbn.bias.data.detach())
                               ).cuda()

    x3 = torch.rand(32, nf3, 4, 32, 32).cuda()
    out_x3 = c_std_3dbn(x3)
    print(x3.shape, out_x3.shape)
    c_std_3dbn._alpha.data = torch.tensor([0.5]).cuda()
    out_x3 = c_std_3dbn(x3)

    print('dim = 3', x3.shape, out_x3.shape)

    for p in c_std_3dbn.parameters():
        print(p, p.requires_grad)

    print(f"---- DIM = 3 ---- END.")

    # Params names
    for name, param in c_std_3dbn.named_parameters():
        print(name, f"{name} == '_alpha': {name == '_alpha'}",
              param, param.requires_grad)


if __name__ == '__main__':
    run_demo_AdadsaFusedBn()

