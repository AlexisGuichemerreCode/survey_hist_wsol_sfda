import sys
from os.path import dirname, abspath

import torch
import torch.nn as nn
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.reproducibility import set_seed

from dlib.losses.element import ElementaryLoss
from dlib.div_classifiers.parts.spg import get_loss as get_spg_loss
from dlib.div_classifiers.parts.acol import get_loss as get_acol_loss
from dlib.losses.elb import ELB

from dlib.configure import constants


__all__ = ['UdaSddaAdvGenerator',
           'UdaSddaAdvDiscriminator',
           'UdaSddaSrcModelCeFakeImage',
           'UdaSddaTrgModelCeFakeImage',
           'UdaSddaDomainDiscriminator',
           'UdaSddaSrcModelPxLikelihood',
           'SpgUdaSddaTrgModelCeFakeImage',
           'AcolUdaSddaTrgModelCeFakeImage',
           'CutMixUdaSddaTrgModelCeFakeImage',
           'MaxMinUdaSddaTrgModelCeFakeImage'
           ]


class UdaSddaAdvGenerator(ElementaryLoss):
    """
    Generator adversarial loss for the SFUDA SDDA method.
    """
    def __init__(self, **kwargs):
        super(UdaSddaAdvGenerator, self).__init__(**kwargs)

        self.loss = nn.BCEWithLogitsLoss(reduction="mean").to(self._device)

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
        super(UdaSddaAdvGenerator, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        discr_logits_fake = sdda_holder['discr_logits_fake']
        assert discr_logits_fake.ndim == 2, discr_logits_fake.ndim  # bsz, 1
        bsz = discr_logits_fake.shape[0]
        trg = torch.ones((bsz, 1), dtype=torch.float,
                         device=discr_logits_fake.device, requires_grad=False)

        loss = self.loss(input=discr_logits_fake, target=trg)

        return self.lambda_ * loss


class UdaSddaAdvDiscriminator(ElementaryLoss):
    """
    Discriminator adversarial loss for the SFUDA SDDA method.
    """

    def __init__(self, **kwargs):
        super(UdaSddaAdvDiscriminator, self).__init__(**kwargs)

        self.loss = nn.BCEWithLogitsLoss(reduction="mean").to(self._device)

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
        super(UdaSddaAdvDiscriminator, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        discr_logits_fake = sdda_holder['discr_logits_fake']
        discr_logits_real = sdda_holder['discr_logits_real']

        assert discr_logits_fake.ndim == 2, discr_logits_fake.ndim  # bsz, 1
        bsz_f = discr_logits_fake.shape[0]
        device = discr_logits_fake.device

        trg_f = torch.zeros((bsz_f, 1), dtype=torch.float,
                            device=device,
                            requires_grad=False)

        loss_fake = self.loss(input=discr_logits_fake, target=trg_f)

        assert discr_logits_real.ndim == 2, discr_logits_real.ndim  # bsz, 1
        bsz_r = discr_logits_real.shape[0]
        trg_r = torch.ones((bsz_r, 1), dtype=torch.float,
                           device=device, requires_grad=False)

        loss_real = self.loss(input=discr_logits_real, target=trg_r)

        loss = (loss_fake + loss_real) / 2.

        return self.lambda_ * loss


class UdaSddaSrcModelCeFakeImage(ElementaryLoss):
    """
    Cross-entropy over the source model class prediction over generated images.
    """
    def __init__(self, **kwargs):
        super(UdaSddaSrcModelCeFakeImage, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

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
        super(UdaSddaSrcModelCeFakeImage, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        src_logits_fake = sdda_holder['src_logits_fake']
        fake_labels = sdda_holder['fake_labels']

        assert src_logits_fake.ndim == 2, src_logits_fake.ndim  # bsz, n_cls
        assert src_logits_fake.shape[1] > 1, src_logits_fake.shape[1]  #
        # n_cls > 1.
        assert fake_labels.ndim == 1, fake_labels.ndim  # bsz
        msg = f"{src_logits_fake.shape[0]} | {fake_labels.shape[0]}"
        assert src_logits_fake.shape[0] == fake_labels.shape[0], msg

        loss = self.loss(input=src_logits_fake, target=fake_labels)

        return self.lambda_ * loss


class UdaSddaTrgModelCeFakeImage(ElementaryLoss):
    """
    Cross-entropy over the source model class prediction over generated images.
    """
    def __init__(self, **kwargs):
        super(UdaSddaTrgModelCeFakeImage, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

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
        super(UdaSddaTrgModelCeFakeImage, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        trg_logits_fake = sdda_holder['trg_logits_fake']
        fake_labels = sdda_holder['fake_labels']

        assert trg_logits_fake.ndim == 2, trg_logits_fake.ndim  # bsz, n_cls
        assert trg_logits_fake.shape[1] > 1, trg_logits_fake.shape[1]  #
        # n_cls > 1.
        assert fake_labels.ndim == 1, fake_labels.ndim  # bsz
        msg = f"{trg_logits_fake.shape[0]} | {fake_labels.shape[0]}"
        assert trg_logits_fake.shape[0] == fake_labels.shape[0], msg

        loss = self.loss(input=trg_logits_fake, target=fake_labels)

        return self.lambda_ * loss


class UdaSddaDomainDiscriminator(ElementaryLoss):
    """
    Cross-entropy over the source model class prediction over generated images.
    """
    def __init__(self, **kwargs):
        super(UdaSddaDomainDiscriminator, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

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
        super(UdaSddaDomainDiscriminator, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        dom_d_logits_fake = sdda_holder['dom_d_logits_fake']
        dom_d_logits_real = sdda_holder['dom_d_logits_real']

        assert dom_d_logits_fake.ndim == 2, dom_d_logits_fake.ndim  # bsz, n_cls
        assert dom_d_logits_fake.shape[1] == 2, dom_d_logits_fake.shape[1]  #
        # 2 classes.

        assert dom_d_logits_real.ndim == 2, dom_d_logits_real.ndim  # bsz, n_cls
        assert dom_d_logits_real.shape[1] == 2, dom_d_logits_real.shape[1]  #
        # 2 classes.

        device = dom_d_logits_fake.device

        trg_f = torch.zeros((dom_d_logits_fake.shape[0],), dtype=torch.long,
                           device=device, requires_grad=False)

        trg_r = torch.ones((dom_d_logits_real.shape[0],), dtype=torch.long,
                            device=device, requires_grad=False)

        loss_f = self.loss(input=dom_d_logits_fake, target=trg_f)
        loss_r = self.loss(input=dom_d_logits_real, target=trg_r)

        loss = (loss_f + loss_r) / 2.

        return self.lambda_ * loss


class UdaSddaSrcModelPxLikelihood(ElementaryLoss):
    """
    Maximize the likelihood p(x) using source model over generated samples.
    This is achieved using energy-based model. See Eq.2, 3 in
    https://arxiv.org/pdf/2102.09003.pdf
    In Eq.2, gradient estimation in the first term is done using average over
    samples FROM MINIBATCH and WITHOUT SAMPLING.
    In https://arxiv.org/pdf/1912.03263.pdf they sample from input sample.
    It is better to have good enough number of samples in the minibatch.
    """
    def __init__(self, **kwargs):
        super(UdaSddaSrcModelPxLikelihood, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

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
        super(UdaSddaSrcModelPxLikelihood, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        src_logits_fake = sdda_holder['src_logits_fake']

        assert src_logits_fake.ndim == 2, src_logits_fake.ndim  # bsz, n_cls
        assert src_logits_fake.shape[1] > 1, src_logits_fake.shape[1]  #
        # n_cls > 1.

        logsumexp = torch.logsumexp(src_logits_fake, dim=1, keepdim=False)
        # bsz
        enrg = - logsumexp
        avg_enrg = enrg.mean()
        log_px = avg_enrg - enrg

        loss = (- log_px).mean()

        return self.lambda_ * loss


# Model based: UdaSddaTrgModelCeFakeImage

class SpgUdaSddaTrgModelCeFakeImage(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SpgUdaSddaTrgModelCeFakeImage, self).__init__(**kwargs)

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
        super(SpgUdaSddaTrgModelCeFakeImage, self).forward(epoch=epoch)

        assert self.hyper_p_set
        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        trg_logits_fake = sdda_holder['trg_logits_fake']
        fake_labels = sdda_holder['fake_labels']

        return get_spg_loss(output_dict=trg_logits_fake,
                            target=fake_labels,
                            spg_thresholds=self.spg_thresholds,
                            ce_label_smoothing=self.ce_label_smoothing
                            ) * self.lambda_


class AcolUdaSddaTrgModelCeFakeImage(ElementaryLoss):
    def __init__(self, **kwargs):
        super(AcolUdaSddaTrgModelCeFakeImage, self).__init__(**kwargs)

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
        super(AcolUdaSddaTrgModelCeFakeImage, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        trg_logits_fake = sdda_holder['trg_logits_fake']
        fake_labels = sdda_holder['fake_labels']

        return get_acol_loss(output_dict=trg_logits_fake,
                             gt_labels=fake_labels,
                             ce_label_smoothing=self.ce_label_smoothing
                             ) * self.lambda_


class CutMixUdaSddaTrgModelCeFakeImage(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CutMixUdaSddaTrgModelCeFakeImage, self).__init__(**kwargs)

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
        super(CutMixUdaSddaTrgModelCeFakeImage, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        sdda_holder = key_arg['sfuda_sdda']
        trg_logits_fake = sdda_holder['trg_logits_fake']
        fake_labels = sdda_holder['fake_labels']
        cutmix_holder = sdda_holder['cutmix_holder']

        cl_logits = trg_logits_fake

        if cutmix_holder is None:
            return self.loss(input=trg_logits_fake,
                             target=fake_labels) * self.lambda_

        assert isinstance(cutmix_holder, list)
        assert len(cutmix_holder) == 3
        target_a, target_b, lam = cutmix_holder
        loss = (self.loss(cl_logits, target_a) * lam + self.loss(
            cl_logits, target_b) * (1. - lam))

        return loss * self.lambda_


class MaxMinUdaSddaTrgModelCeFakeImage(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxMinUdaSddaTrgModelCeFakeImage, self).__init__(**kwargs)

        raise NotImplementedError  # todo.

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
        super(MaxMinUdaSddaTrgModelCeFakeImage, self).forward(epoch=epoch)

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