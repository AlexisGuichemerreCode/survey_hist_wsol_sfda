import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple
from copy import deepcopy
import pickle as pkl
import math
import datetime as dt


import numpy as np
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import torch.nn.functional as F

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torchvision.utils as vutils


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.datasets.wsol_loader import get_data_loader

from dlib.learning.train_wsol import Trainer
from dlib import losses
from dlib.process.instantiators import get_loss
from dlib.process.instantiators import get_optimizer_of_model
from dlib.process.instantiators import get_optimizer_for_params
from dlib.process.instantiators import sfuda_get_gan_sdda_model
from dlib.process.instantiators import sfuda_get_domain_discriminator_sdda_model
from dlib.process.instantiators import sfuda_sdda_get_generation_adaptation_loss
from dlib.process.instantiators import sfuda_sdda_get_adv_discriminator_loss

from dlib.div_classifiers.parts.has import has as wsol_has
from dlib.div_classifiers.parts.cutmix import cutmix as wsol_cutmix

import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg

IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
IMAGE_STD_VALUE = [0.229, 0.224, 0.225]

__all__ = ['TrainerSdda']


class TrainerSdda(Trainer):
    def __init__(self,
                 args,
                 model,
                 classifier=None,
                 model_src=None
                 ):
        super(TrainerSdda, self).__init__(args=args,
                                          model=model,
                                          classifier=classifier,
                                          model_src=model_src
                                          )

        assert args.sf_uda
        assert args.sdda

        g, d = sfuda_get_gan_sdda_model(args, eval=False)

        self.discriminator = d.cuda(args.c_cudaid)
        self.generator = g.cuda(args.c_cudaid)
        # todo: get featues_dim from model.
        # domain discriminator.
        features_dim = model.encoder_n_out_channels
        self.d_discriminator = sfuda_get_domain_discriminator_sdda_model(
            args, features_dim=features_dim, eval=False).cuda(args.c_cudaid)

        params = []
        for m in [model, self.generator, self.d_discriminator]:
            params = params + [p for p in m.parameters()]

        optimizer, lr_scheduler = get_optimizer_for_params(args.optimizer,
                                                           params,
                                                           'opt'
                                                           )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        optimizer_d, lr_scheduler_d = get_optimizer_for_params(
            vars(args),
            self.discriminator.parameters(),
            'sdda_d'
        )
        self.optimizer_d = optimizer_d
        self.lr_scheduler_d = lr_scheduler_d

        self.loss = sfuda_sdda_get_generation_adaptation_loss(args)
        self.adv_d_loss = sfuda_sdda_get_adv_discriminator_loss(args)

        # Samples generation (for illustration)
        self.gen_holder_input = self._init_gen_holder_input()

    def _init_gen_holder_input(self) -> list:
        args = self.args

        bsz = args.batch_size  # todo: config how much to sample per class.
        n_cls = args.num_classes
        assert n_cls > 1, n_cls
        device = torch.device(f"cuda:{args.c_cudaid}")
        latent_dim = args.sdda_gan_latent_dim

        path = os.path.normpath(join(root_dir, dirname(args.metadata_root)))

        with open(join(path, 'encoding.yaml'), 'r') as f:
            encoding = yaml.safe_load(f)  # name: int

        int_cl = {v: k for k, v in encoding.items()}

        gen_holder_input = []
        for i in range(n_cls):
            fake_labels = torch.zeros((bsz,), device=device).long() + i
            fake_labels = fake_labels.long()
            noise = torch.randn((bsz, latent_dim), device=device)

            str_name_cl = int_cl[i]
            gen_holder_input.append([str_name_cl, fake_labels, noise])

        return gen_holder_input


    def on_epoch_start(self):
        torch.cuda.empty_cache()

        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        if self.args.sf_uda:
            self._sf_uda_before_epoch_process()

        # final
        self.model.train()

        if self.args.sf_uda:
            if self.args.sdda:

                self.model.train()

                self.discriminator.train()
                self.generator.train()
                self.d_discriminator.train()

                self.model_src.eval()

            else:  # todo
                raise NotImplementedError('Add more SFUDA methods.')

    def _one_step_train(self,
                        images,
                        fake_images,
                        fake_labels,
                        raw_imgs,
                        targets,
                        p_glabel,
                        std_cams,
                        masks,
                        views
                        ):

        args = self.args
        assert args.sf_uda
        assert args.sdda

        y_global = targets
        y_pl_global = p_glabel

        z_label = targets

        # todo: add more SFUDA methods.
        if args.sf_uda:
            z_label = p_glabel

        if args.method == constants.METHOD_HAS:
            pass  # HAS is in inference mode.

        cutmix_holder = None
        if args.method == constants.METHOD_CUTMIX:
            pass  # CUTMIX is in inference mode. we cant alter the fake image.

        if args.sf_uda:
            if args.task == constants.STD_CL:

                if args.sdda:
                    discr_logits_fake = self.discriminator(fake_images)
                    _min = images.min()
                    _max = images.max()
                    _in_img_d = F.interpolate(images,
                                              size=fake_images.shape[2:],
                                              mode='bilinear',
                                              align_corners=True
                                              )
                    _in_img_d = torch.clip(_in_img_d, _min, _max)
                    discr_logits_real = self.discriminator(_in_img_d)

                    src_logits_fake = self.model_src(fake_images)

                    # adaptation step
                    a = self.sfuda_master.lambda_
                    trg_logits_fake = self.model(fake_images)

                    ft = self.model.encoder_last_features
                    assert ft.ndim == 4, ft.ndim  # bsz, d, h, w
                    ft_fake = F.adaptive_avg_pool2d(ft, 1)  # bsz, d, 1, 1
                    ft_fake = ft_fake.squeeze((2, 3))  # bsz, d
                    dom_d_logits_fake = self.d_discriminator(ft_fake, a)

                    trg_logits_real = self.model(images)
                    ft = self.model.encoder_last_features
                    assert ft.ndim == 4, ft.ndim  # bsz, d, h, w
                    ft_real = F.adaptive_avg_pool2d(ft, 1)  # bsz, d
                    ft_real = ft_real.squeeze((2, 3))  # bsz, d
                    dom_d_logits_real = self.d_discriminator(ft_real, a)

                    key_arg = {'sfuda_sdda':
                                   {'discr_logits_fake': discr_logits_fake,
                                    'discr_logits_real': discr_logits_real,
                                    'src_logits_fake': src_logits_fake,
                                    'trg_logits_fake': trg_logits_fake,
                                    'dom_d_logits_fake': dom_d_logits_fake,
                                    'dom_d_logits_real': dom_d_logits_real,
                                    'fake_labels': fake_labels,
                                    'cutmix_holder': cutmix_holder}}

                    cl_logits = trg_logits_fake
                    loss = self.loss(epoch=self.epoch,
                                     model=self.model,
                                     cl_logits=cl_logits,
                                     glabel=y_global,
                                     pseudo_glabel=y_pl_global,
                                     key_arg=key_arg
                                     )
                    logits = cl_logits

            elif args.task == constants.F_CL:
                raise NotImplementedError

            elif args.task == constants.NEGEV:
                raise NotImplementedError

            elif args.task == constants.SEG:
                raise NotImplementedError


        else:

            raise ValueError

        return logits, loss

    def train(self, split: str, epoch: int) -> dict:
        self.epoch = epoch
        self.random()
        self.on_epoch_start()

        args = self.args
        assert args.sf_uda
        assert args.sdda

        assert split == constants.TRAINSET
        loader = self.loaders[split]

        total_loss = None
        num_correct = 0
        num_images = 0

        scaler = GradScaler(enabled=args.amp)
        scaler_d = GradScaler(enabled=args.amp)

        mbatchsz = 0
        device = torch.device(f"cuda:{args.c_cudaid}")

        for batch_idx, (images, targets, p_glabel, _,
                        raw_imgs, std_cams, masks, views) in tqdm(
            enumerate(loader), ncols=constants.NCOLS, total=len(loader)):

            self.random()
            if batch_idx == 0:
                mbatchsz = images.shape[0]

            # SFUDA: todo: warning: std_cams must be extracted using the
            #  pseudo-labels not the true labels (offline, online extraction).
            # fill
            images = self._fill_minibatch(images, mbatchsz)
            targets = self._fill_minibatch(targets, mbatchsz)
            p_glabel = self._fill_minibatch(p_glabel, mbatchsz)
            raw_imgs = self._fill_minibatch(raw_imgs, mbatchsz)

            images = images.cuda(args.c_cudaid)
            targets = targets.cuda(args.c_cudaid)
            p_glabel = p_glabel.cuda(args.c_cudaid)

            if views.ndim == 1:
                views = None

            else:
                views = self._fill_minibatch(views, mbatchsz)
                views = views.cuda(args.c_cudaid)

            if masks.ndim == 1:
                masks = None
            else:
                masks = self._fill_minibatch(masks, mbatchsz)
                masks = masks.cuda(args.c_cudaid)

            if std_cams.ndim == 1:
                std_cams = None
            else:
                assert std_cams.ndim == 4
                std_cams = self._fill_minibatch(std_cams, mbatchsz)
                std_cams = std_cams.cuda(args.c_cudaid)

                with autocast(enabled=args.amp):
                    with torch.no_grad():
                        std_cams = self.prepare_std_cams_disq(
                            std_cams=std_cams, image_size=images.shape[2:])

            self.optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                # generate fake images / labels --------------------------------
                bsz = images.shape[0]
                n_cls = args.num_classes
                assert n_cls > 1, n_cls

                fake_labels = torch.randint(0, n_cls, (bsz,),
                                            device=device).long()
                latent_dim = args.sdda_gan_latent_dim
                noise = torch.randn((bsz, latent_dim), device=device)
                fake_images = self.generator(noise, fake_labels)
                # --------------------------------------------------------------

                # Adv. discriminator
                discr_logits_fake = self.discriminator(fake_images.detach())
                _min = images.min()
                _max = images.max()
                _in_img_d = F.interpolate(images,
                                          size=fake_images.shape[2:],
                                          mode='bilinear',
                                          align_corners=True
                                          )
                _in_img_d = torch.clip(_in_img_d, _min, _max)
                discr_logits_real = self.discriminator(_in_img_d)
                loss_adv_d = self.adv_d_loss(
                    epoch=self.epoch,
                    key_arg={'sfuda_sdda':
                                 {'discr_logits_fake': discr_logits_fake,
                                  'discr_logits_real': discr_logits_real}}
                )

            # Update Adv. discriminator
            if loss_adv_d.requires_grad:
                scaler_d.scale(loss_adv_d).backward()
                scaler_d.step(self.optimizer_d)
                scaler_d.update()

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.args.amp):
                logits, loss = self._one_step_train(images,
                                                    fake_images,
                                                    fake_labels,
                                                    raw_imgs,
                                                    targets,
                                                    p_glabel,
                                                    std_cams,
                                                    masks,
                                                    views
                                                    )

            with torch.no_grad():
                if self.args.task != constants.SEG:
                    pred = logits.argmax(dim=1)
                    num_correct += (pred == fake_labels).sum().detach()

            if total_loss is None:
                total_loss = loss.detach().squeeze() * images.size(0)
            else:
                total_loss += loss.detach().squeeze() * images.size(0)
            num_images += images.size(0)

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

        loss_average = total_loss.item() / float(num_images)

        classification_acc = 0.0
        if self.args.task != constants.SEG:
            classification_acc = num_correct.item() / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        # Plotting generated samples
        self.plot_some_generated_samples()

        self.on_epoch_end()

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def adjust_learning_rate(self):
        super(TrainerSdda, self).adjust_learning_rate()

        self.lr_scheduler_d.step()

    def on_epoch_end(self):
        self.loss.update_t()
        self.adv_d_loss.update_t()
        print('Self.loss status:')
        self.loss.check_losses_status()

        print('Self.adv_d_loss status:')
        self.adv_d_loss.check_losses_status()

        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        DLLogger.log(fmsg(f'Train epoch runtime: {delta_t}'))

        torch.cuda.empty_cache()

    def store_extra_models(self):
        # todo: - track best models (generator, discriminator, domain
        #  discriminator) the same way as for the target model (self.model).
        #  save best at best_cl, and best_loc. - Store them. the only model
        #  that one may need is the generator. it can be used to generate
        #  samples for illustration.
        raise NotImplementedError

    def plot_some_generated_samples(self):
        # todo: improve the visualization.
        outd = join(self.args.outd, f'generator-samples')
        training = self.generator.training
        self.generator.eval()

        device = torch.device(f'cuda:{self.args.c_cudaid}')
        m = torch.tensor(IMAGE_MEAN_VALUE, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD_VALUE, device=device).view(1, 3, 1, 1)


        for item in self.gen_holder_input:
            str_name_cl, fake_labels, noise = item
            _out = join(outd, str_name_cl)
            os.makedirs(_out, exist_ok=True)
            with torch.no_grad():
                fake_images = self.generator(noise, fake_labels)

            assert fake_images.ndim == 4, fake_images.ndim  # bsz, 3, h, w
            assert fake_images.shape[1] == 3, fake_images.shape[1]  # RGB

            fake_images = fake_images * std + m
            fake_images = fake_images.detach().cpu()
            vutils.save_image(tensor=fake_images,
                              fp=join(_out, f"{self.epoch}.png")
                              )



        if training:
            self.generator.train()