import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple
from os.path import join

import torch
import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg

from dlib.datasets.wsol_loader import get_data_loader

# Paper SDDA: "Domain Impression: A Source Data Free Domain Adaptation
# Method", WACV, 2021.
# https://arxiv.org/abs/2102.09003.


__all__ = ['Sdda']


class Sdda(object):
    def __init__(self, args, model_trg):

        self.args = args
        assert args.sf_uda
        assert args.sdda
        assert isinstance(args.ce_dom_d_sdda_a, float), type(args.ce_dom_d_sdda_a)
        assert args.ce_dom_d_sdda_a > 0, args.ce_dom_d_sdda_a

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

        # lambda: adaptive lambda for the reversed-layer.
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
        a = self.args.sdda
        start_steps = self.n_mbatches * epoch

        p = float(mb_idx + start_steps) / self.total_steps
        v = (2. / (1. + np.exp(-a * p))) - 1.

        v = min(max(0., v), 1.)

        self.set_lambda_(v)

    def _update_img_cls_pseudo_lbs(self) -> dict:
        """
        Pseudo-label entire trainset at once.
        WARNING: THIS IS NOT NECESSARY FOR THIS METHOD. IT IS ADDED TO CHECK
        THE QUALITY OF THE IMAGE-CLASS PSEUDO-LABELS.
        """

        self.model_trg.eval()

        loader = self.train_eval_loader
        all_image_ids = []
        all_img_lbs = None
        all_plbs = None

        for i, (images, targets, _, image_ids, _, _, _, _) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()  # todo: track targets to MEASURE acc of
            # pseudo-labels.
            out_trg = self.model_trg(images)
            if self.args.task == constants.STD_CL:
                cl_logits_trg = out_trg

            elif self.args.task in [constants.NEGEV]:
                cl_logits_trg, fcams, im_recon = out_trg

            else:
                raise NotImplementedError

            prob_trg = torch.softmax(cl_logits_trg, dim=1)

            plbs = torch.argmax(prob_trg, dim=1, keepdim=False)
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

        msg = f"SDDA - ACC pseudo-label image-class: {acc} %"
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
        """

        assert isinstance(images, torch.Tensor), type(images)
        assert images.ndim == 4, images.ndim  # bsz, c, h, w

        self.model_trg.eval()

        out_trg = self.model_trg(images)


        if self.args.task == constants.STD_CL:
            cl_logits_trg = out_trg

        elif self.args.task in [constants.NEGEV]:
            cl_logits_trg, fcams, im_recon = out_trg

        else:
            raise NotImplementedError

        prob_trg = torch.softmax(cl_logits_trg, dim=1)
        plbs = torch.argmax(prob_trg, dim=1, keepdim=False)
        assert plbs.ndim == 1, plbs.ndim  # bsz

        plbs = plbs.long().detach()

        return plbs


    def update_img_cls_pseudo_lbs(self) -> dict:

        with torch.no_grad():
            return self._update_img_cls_pseudo_lbs()

    def pseudo_label_imgs(self, images: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            return self._pseudo_label_imgs(images)