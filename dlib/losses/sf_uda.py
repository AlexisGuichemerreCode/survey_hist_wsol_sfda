import sys
from os.path import dirname, abspath

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.reproducibility import set_seed

from dlib.losses.entropy import Entropy
from dlib.losses.entropy import ContCrossEntropy
from dlib.losses.element import ElementaryLoss
from dlib.div_classifiers.parts.spg import get_loss as get_spg_loss
from dlib.div_classifiers.parts.acol import get_loss as get_acol_loss
from dlib.losses.elb import ELB
from dlib.losses.cdd import CDD
from dlib.losses.cdcl import Cdcl
from dlib.losses.nrc import NA, ENA, KL
from dlib.configure import constants


__all__ = [
    # SHOT
    'UdaCrossEntropyImgPseudoLabels',
    'UdaTargetClassProbEntropy',
    'UdaDiversityTargetClass',
    # next are equivalent to UdaCrossEntropyImgPseudoLabels for
    # specific WSOL methods
    'UdaSpgLoss',
    'UdaAcolLoss',
    'UdaCutMixLoss',
    'UdaMaxMinLoss',

    # FAUST
    'UdaFeatureViewsConsistencyFaust',
    'UdaClassProbsViewsSoftLabelsFaust',
    'UdaMcDropoutVarMinFaust',
    'UdaClassProbsEntropyFaust',
    
    #SFDE
    'UdaCdd'

    # NRC  
    'UdaNANrc'
    ]


class UdaCrossEntropyImgPseudoLabels(ElementaryLoss):
    """
    Cross entropy loss over image global classes using pseudo-labels as
    supervision.
    """
    def __init__(self, **kwargs):
        super(UdaCrossEntropyImgPseudoLabels, self).__init__(**kwargs)

        self.ce_label_smoothing: float = 0.0

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = False

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaCrossEntropyImgPseudoLabels, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        loss = self.loss(input=cl_logits, target=pseudo_glabel)

        return self.lambda_ * loss


class UdaTargetClassProbEntropy(ElementaryLoss):
    """
    Minimize entropy over image class probability of target.
    """
    def __init__(self, **kwargs):
        super(UdaTargetClassProbEntropy, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaTargetClassProbEntropy, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert cl_logits.ndim == 2, cl_logits.ndim

        probs = torch.softmax(cl_logits, dim=1)
        loss = self.loss(probs).mean()

        return self.lambda_ * loss


class UdaDiversityTargetClass(ElementaryLoss):
    """
    Diversity loss over image global class prediction. Push expected prob
    vector over minibatch to follow uniform dist. 'Discriminative Clustering
    by Regularized Information Maximization', 2010.
    """
    def __init__(self, **kwargs):
        super(UdaDiversityTargetClass, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaDiversityTargetClass, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert cl_logits.ndim == 2, cl_logits.ndim

        probs = torch.softmax(cl_logits, dim=1)
        prob_avg = probs.mean(dim=0, keepdim=True)  # 1, ncls
        loss = - self.loss(prob_avg).mean()

        return self.lambda_ * loss


class UdaSpgLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(UdaSpgLoss, self).__init__(**kwargs)

        self.spg_threshold_1h = None
        self.spg_threshold_1l = None
        self.spg_threshold_2h = None
        self.spg_threshold_2l = None
        self.spg_threshold_3h = None
        self.spg_threshold_3l = None

        self.hyper_p_set = False

        self.ce_label_smoothing: float = 0.0
        self.already_set = False

    @property
    def spg_thresholds(self):
        assert self.hyper_p_set

        h1 = self.spg_threshold_1h
        l1 = self.spg_threshold_1l

        h2 = self.spg_threshold_2h
        l2 = self.spg_threshold_2l

        h3 = self.spg_threshold_3h
        l3 = self.spg_threshold_3l

        return (h1, l1), (h2, l2), (h3, l3)

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaSpgLoss, self).forward(epoch=epoch)

        assert self.hyper_p_set
        assert self.already_set

        if not self.is_on():
            return self._zero

        return get_spg_loss(output_dict=model.logits_dict,
                            target=pseudo_glabel,
                            spg_thresholds=self.spg_thresholds,
                            ce_label_smoothing=self.ce_label_smoothing
                            ) * self.lambda_


class UdaAcolLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(UdaAcolLoss, self).__init__(**kwargs)

        self.ce_label_smoothing: float = 0.0
        self.already_set = False

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaAcolLoss, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        return get_acol_loss(output_dict=model.logits_dict,
                             gt_labels=pseudo_glabel,
                             ce_label_smoothing=self.ce_label_smoothing
                             ) * self.lambda_


class UdaCutMixLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(UdaCutMixLoss, self).__init__(**kwargs)

        self.ce_label_smoothing: float = 0.0

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = False

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaCutMixLoss, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        if cutmix_holder is None:
            return self.loss(input=cl_logits,
                             target=pseudo_glabel) * self.lambda_

        assert isinstance(cutmix_holder, list)
        assert len(cutmix_holder) == 3
        target_a, target_b, lam = cutmix_holder
        loss = (self.loss(cl_logits, target_a) * lam + self.loss(
            cl_logits, target_b) * (1. - lam))

        return loss * self.lambda_


class UdaMaxMinLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(UdaMaxMinLoss, self).__init__(**kwargs)

        self.dataset_name: str = ''
        assert isinstance(self.elb, ELB)
        self.lambda_size = 0.
        self.lambda_neg = 0.

        self._lambda_size_set = False
        self._lambda_neg_set = False

        self.ce_label_smoothing: float = 0.0

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.BCE = nn.BCEWithLogitsLoss(reduction="mean").to(self._device)

        self.softmax = nn.Softmax(dim=1)

    def set_ce_label_smoothing(self, ce_label_smoothing: float = 0.0):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

    def set_lambda_neg(self, lambda_neg: float):
        assert isinstance(lambda_neg, float)
        assert lambda_neg >= 0.
        self.lambda_neg = lambda_neg

        self._lambda_neg_set = True

    def set_lambda_size(self, lambda_size: float):
        assert isinstance(lambda_size, float)
        assert lambda_size >= 0.
        self.lambda_size = lambda_size

        self._lambda_size_set = True

    def set_dataset_name(self, dataset_name: str):
        self._assert_dataset_name(dataset_name=dataset_name)
        self.dataset_name = dataset_name

    def _is_ready(self):
        assert self._lambda_size_set
        assert self._lambda_neg_set
        self._assert_dataset_name(dataset_name=self.dataset_name)

    def _assert_dataset_name(self, dataset_name: str):
        assert isinstance(dataset_name, str)
        assert dataset_name in [constants.GLAS, constants.CAMELYON512]

    def kl_uniform_loss(self, logits):
        assert logits.ndim == 2
        logsoftmax = torch.log2(self.softmax(logits))
        return (-logsoftmax).mean(dim=1).mean()

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaMaxMinLoss, self).forward(epoch=epoch)

        self._is_ready()

        if not self.is_on():
            return self._zero

        logits = model.logits_dict['logits']
        logits_pos = model.logits_dict['logits_pos']
        logits_neg = model.logits_dict['logits_neg']

        cam = model.logits_dict['cam']
        cam_logits = model.logits_dict['cam_logits']
        
        assert cam.ndim == 4
        assert cam.shape[1] == 1
        assert cam.shape == cam_logits.shape
        bs, _, _, _ = cam.shape

        cl_losss = self.loss(input=logits, target=pseudo_glabel)
        total_l = cl_losss
        size = cam.contiguous().view(bs, -1).sum(dim=-1).view(-1, )

        if self.dataset_name == constants.GLAS:
            size_loss = self.elb(-size) + self.elb(-1. + size)
            total_l = total_l + self.lambda_size * size_loss * 0.0

            total_l = total_l + self.loss(input=logits_pos,
                                          target=pseudo_glabel) * 0.
            total_l = total_l + self.lambda_neg * self.kl_uniform_loss(
                logits=logits_neg) * 0.0

        if self.dataset_name == constants.CAMELYON512:
            # pos
            ind_metas = (pseudo_glabel == 1).nonzero().view(-1)
            if ind_metas.numel() > 0:
                tmps = size[ind_metas]
                size_loss = self.elb(-tmps) + self.elb(-1. + tmps)
                total_l = total_l + self.lambda_size * size_loss

            # neg
            ind_normal = (pseudo_glabel == 0).nonzero().view(-1)
            if ind_normal.numel() > 0:
                trg_cams = torch.zeros(
                    (ind_normal.numel(), 1, cam.shape[2], cam.shape[3]),
                    dtype=torch.float, device=cam.device)

                total_l = total_l + self.BCE(input=cam_logits[ind_normal],
                                             target=trg_cams)

        return total_l * self.lambda_


# FAUST

class UdaFeatureViewsConsistencyFaust(ElementaryLoss):
    """
    Views features alignment [views_ft_consist, FAUST].
    Minimize cosine dissimilarity between features of augmented image and
    features of non-augmented image.
    """
    def __init__(self, **kwargs):
        super(UdaFeatureViewsConsistencyFaust, self).__init__(**kwargs)

        self.already_set = True

    def pairwise_cosine_dissim(self,
                               x1: torch.Tensor,
                               x2: torch.Tensor) -> torch.Tensor:

        assert x1.ndim == 2, x1.ndim  # m, d
        assert x2.ndim == 2, x2.ndim  # n, d
        assert x1.shape[1] == x2.shape[1], f"{x1.shape[1]} | {x2.shape[1]}"

        dist = 1. - F.cosine_similarity(x1[:, :, None],
                                        x2.t()[None, :, :], dim=1)  # m, n

        assert dist.ndim == 2, dist.ndim
        assert dist.shape[0] == x1.shape[0], f"{dist.shape[0]} | {x1.shape[0]}"
        assert dist.shape[1] == x2.shape[0], f"{dist.shape[1]} | {x2.shape[0]}"

        return dist

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaFeatureViewsConsistencyFaust, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        assert key_arg is not None
        ft: list = key_arg['faust_views_lin_ft']  # list of tensors: b, ft
        assert len(ft) > 1, len(ft)  # non-aug, v0, v1, ... [at least 1 view]

        for d in ft:
            assert d.shape == ft[0].shape, f"{d.shape} | {ft[0].shape}"

        clean_ft = ft[0].detach()
        ft_views = ft[1:]

        loss = self._zero
        n = len(ft_views)
        assert n >= 1, n

        for i in range(n):
            c_sine_diss = self.pairwise_cosine_dissim(ft_views[i], clean_ft)
            loss = loss + c_sine_diss.mean()

        loss = loss / float(n)

        return self.lambda_ * loss


class UdaClassProbsViewsSoftLabelsFaust(ElementaryLoss):
    """
    Views class probabilities alignment [ce_views_soft_pl, FAUST].
    Align each class probabilities of each view to the estimated class
    soft-labels.
    """
    def __init__(self, **kwargs):
        super(UdaClassProbsViewsSoftLabelsFaust, self).__init__(**kwargs)

        self.loss = ContCrossEntropy(sumit=True)

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaClassProbsViewsSoftLabelsFaust, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        assert key_arg is not None

        cl_logits: list = key_arg['faust_views_cl_logits']  # list of
        # tensors: b, ncls
        cl_soft_lbls = key_arg['faust_views_soft_pl']  # tensor: b, ncls.
        # logits (unnormalized)

        assert len(cl_logits) > 1, len(cl_logits)  # non-aug, v0, v1,
        # ... [at least 1 view]

        for d in cl_logits:
            assert d.shape == cl_soft_lbls.shape, f"{d.shape} | " \
                                                  f"{cl_soft_lbls.shape}"
            assert d.ndim == 2, d.ndim  # b, ncls

        prob_trgs = torch.softmax(cl_soft_lbls, dim=1).detach()
        logits_views = cl_logits[1:]

        loss = self._zero
        n = len(logits_views)
        assert n >= 1, n

        for i in range(n):
            probs = torch.softmax(logits_views[i], dim=1)
            ce = self.loss(p=prob_trgs, q=probs)  # b
            loss = loss + ce.mean()

        loss = loss / float(n)

        return self.lambda_ * loss


class UdaMcDropoutVarMinFaust(ElementaryLoss):
    """
    Minimize the l2-norm of the variance estimated over class probabilities
    computed over different runs via Mc-dropout over a non-augmented input
    image. [mc_var_prob, FAUST]
    """
    def __init__(self, **kwargs):
        super(UdaMcDropoutVarMinFaust, self).__init__(**kwargs)

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaMcDropoutVarMinFaust, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        assert key_arg is not None

        std = key_arg['mc_var_prob_std']  # b, ncls
        assert std.ndim == 2, std.ndim

        loss = torch.norm(std, p=2, dim=1, keepdim=False).mean()  # b

        return self.lambda_ * loss


class UdaClassProbsEntropyFaust(ElementaryLoss):
    """
    Minimize entropy over class probabilities of non-augmented image [
    min_prob_entropy, FAUST].
    """
    def __init__(self, **kwargs):
        super(UdaClassProbsEntropyFaust, self).__init__(**kwargs)

        self.loss = Entropy()

        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaClassProbsEntropyFaust, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        assert key_arg is not None

        cl_logits = key_arg['min_prob_entropy_logits']  # b, cls
        assert cl_logits.ndim == 2, cl_logits.ndim

        probs = torch.softmax(cl_logits, dim=1)  # b, cls
        loss = self.loss(probs).mean()

        return self.lambda_ * loss
    
class UdaNANrc(ElementaryLoss):
    """
    Compute NANrc loss 
    """
    def __init__(self, **kwargs):
        super(UdaNANrc, self).__init__(**kwargs)
        self.loss = NA()

        self.already_set = False

    def set_it(self,nrc_na_lambda):
        self.nrc_na_lambda = nrc_na_lambda
        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaNANrc, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        output_re = key_arg['output_re']
        score_near_kk = key_arg['score_near_kk']
        weight_kk = key_arg['weight_kk']
        loss = self.loss.forward(output_re, score_near_kk, weight_kk)

        return loss
    
class UdaENANrc(ElementaryLoss):
    """
    Compute ENANrc loss 
    """
    def __init__(self, **kwargs):
        super(UdaENANrc, self).__init__(**kwargs)
        self.loss = ENA()

        self.already_set = False

    def set_it(self,nrc_ena_lambda):
        self.nrc_ena_lambda = nrc_ena_lambda
        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaENANrc, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        softmax_out_un = key_arg['softmax_out_un']
        score_near = key_arg['score_near']
        weight = key_arg['weight']
        loss = self.loss.forward(softmax_out_un, score_near, weight)*self.nrc_ena_lambda

        return loss
    
class UdaKLNrc(ElementaryLoss):
    """
    Compute KLNrc loss 
    """
    def __init__(self, **kwargs):
        super(UdaKLNrc, self).__init__(**kwargs)
        self.epsilon = 1e-5
        self.loss = KL(self.epsilon)
        self.already_set = False

    def set_it(self, nrc_kl_lambda, epsilon):
        self.nrc_kl_lambda = nrc_kl_lambda
        self.epsilon = epsilon
        self.already_set = True

    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None
                ):
        super(UdaKLNrc, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        msoftmax = key_arg['msoftmax']
        loss = self.loss.forward(msoftmax, self.epsilon)*self.nrc_kl_lambda

        return loss
    
        
class UdaCdd(ElementaryLoss):
    """
    Compute CDD loss based on the paper: https://arxiv.org/abs/2204.11257. Use MMD loss for the intra-class loss and inter-class loss.
    """
    def __init__(self, **kwargs):
        super(UdaCdd, self).__init__(**kwargs)
        
        self.num_layers = 2
        self.kernel_num = [5,5]
        self.kernel_mul = [2,2]
        self.num_classes = 2

        self.loss = CDD(num_layers=self.num_layers, kernel_num=self.kernel_num, kernel_mul=self.kernel_mul, num_classes=self.num_classes)

        self.already_set = False
    
    def set_it(self, num_layers: int, kernel_num: list, kernel_mul: list, num_classes: int, lambda_: float):

        self.num_layers = num_layers
        self.kernel_num= kernel_num
        self.kernel_mul=kernel_mul
        self.num_classes=num_classes
        self.lambda_ = lambda_ 
        self.loss = CDD(num_layers=self.num_layers, kernel_num=self.kernel_num, kernel_mul=self.kernel_mul, num_classes=self.num_classes)
        self.already_set = True
    
    def forward(self,
                epoch=0,
                model=None,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                seg_logits=None,
                glabel=None,
                pseudo_glabel=None,
                masks=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                cutmix_holder=None,
                key_arg: dict = None  # holds multiple input at once. access
                # via appropriate key.
                ):
        super(UdaCdd, self).forward(epoch=epoch)

        assert self.already_set

        uniques_classes= set(pseudo_glabel.cpu().numpy())

        normal_sampler= key_arg['surrogate_sampler']
        target_features = key_arg['target_features']
        grouped_indices_dict = {}

        for value in uniques_classes:
            indices = [i for i, x in enumerate(pseudo_glabel) if x == value]
            grouped_indices = [indices[n:n+3] for n in range(0, len(indices), 3)]
            # Remove last group if its size is not 3
            if len(grouped_indices[-1]) != 3:
                grouped_indices = grouped_indices[:-1]
            
            grouped_indices_dict[value] = grouped_indices

        num_samples = 3
        nums_cls=[num_samples,num_samples]
        loss = self._zero
        
        if len(grouped_indices_dict) <= 1:
            return loss 
        else:
            for group0,group1 in zip(grouped_indices_dict[0],grouped_indices_dict[1]):
                Features = []
                list_of_source = []
                samples_0 = normal_sampler[0].sample(sample_shape=(num_samples,)).to(self._device).requires_grad_(True)
                samples_1 = normal_sampler[1].sample(sample_shape=(num_samples,)).to(self._device).requires_grad_(True)
                Features.extend([target_features.squeeze()[group0],target_features.squeeze()[group1]])
                list_of_source.extend([samples_0,samples_1])

                result_target = torch.cat(Features, dim=0)
                result_source = torch.cat(list_of_source, dim=0)
                

                if "tscam" in model.name:
                    result_target_resize = torch.cat(Features, dim=0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 14, 14)
                    result_source_resize = torch.cat(list_of_source, dim=0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 14, 14)

                    output_target = model.head(result_target_resize)
                    output_source = model.head(result_source_resize)

                    probabilities_target = model.avgpool(output_target).squeeze(3).squeeze(2)
                    probabilities_source = model.avgpool(output_source).squeeze(3).squeeze(2)
                    
                else: 
                    result_target = torch.cat(Features, dim=0)
                    result_source = torch.cat(list_of_source, dim=0)

                    output_target = model.classification_head(result_target.unsqueeze(-1).unsqueeze(-1))
                    output_source = model.classification_head(result_source.unsqueeze(-1).unsqueeze(-1))

                    probabilities_target = F.softmax(output_target, dim=1)
                    probabilities_source = F.softmax(output_source, dim=1)
                #loss_tmp= self.loss.forward([result_source,probabilities_source], [result_target,probabilities_target],nums_cls, nums_cls)['cdd']
                #print(f"loss_tmp :{loss_tmp}")
                
                loss = loss + self.loss.forward([result_source,probabilities_source], [result_target,probabilities_target],nums_cls, nums_cls)['cdd']
                #print(f"loss :{loss}")
                
            return loss* self.lambda_
        
class UdaCdcl(ElementaryLoss):
    """
    Compute CDCL loss based on the paper: https://arxiv.org/pdf/2106.05528.pdf.
    """
    def __init__(self, **kwargs):
        super(UdaCdcl, self).__init__(**kwargs)
        
        self.tau = 0.01
        self.weights = torch.randn(2048, 2)
        self.loss = Cdcl(tau=self.tau)   
        self.cdcl_lambda = 1.0
        self.already_set = False
    
    def set_it(self, tau: float, cdcl_lambda: float):

        self.tau = tau
        self.cdcl_lambda = cdcl_lambda
        self.loss = Cdcl(tau=self.tau) 
        self.already_set = True

    def forward(self,
            epoch=0,
            model=None,
            cams_inter=None,
            fcams=None,
            cl_logits=None,
            seg_logits=None,
            glabel=None,
            pseudo_glabel=None,
            masks=None,
            raw_img=None,
            x_in=None,
            im_recon=None,
            seeds=None,
            cutmix_holder=None,
            key_arg: dict = None  # holds multiple input at once. access
            # via appropriate key.
            ):
        super(UdaCdcl, self).forward(epoch=epoch)

        assert self.already_set

        self.weights=model.get_linear_weights

        target_features = key_arg['target_features']

        loss = self.loss.forward(target_features,pseudo_glabel,self.weights).sum()
        
        return loss* self.cdcl_lambda
