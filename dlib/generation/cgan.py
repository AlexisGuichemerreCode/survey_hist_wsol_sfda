import torch
import torch.nn as nn
from functools import partial
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

import sys
from os.path import dirname, abspath, join

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.shared import count_params

# Implement CGAN: "Conditional Generative Adversarial Nets",
# https://arxiv.org/pdf/1411.1784.pdf

class Generator(nn.Module):
    def __init__(self,
                 n_classes: int,
                 latent_dim: int,
                 img_shape: list
                 ):
        super(Generator, self).__init__()

        assert isinstance(n_classes, int), type(n_classes)
        assert n_classes > 0, n_classes
        self.n_classes = n_classes

        assert isinstance(latent_dim, int), type(latent_dim)
        assert latent_dim > 0, latent_dim
        self.latent_dim = latent_dim

        assert isinstance(img_shape, list), type(img_shape)
        assert len(img_shape) == 3, len(img_shape)  # c, h, w
        self.img_shape = img_shape

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, noise: torch.Tensor, labels: torch.Tensor):
        assert noise.shape[0] == labels.shape[0], f"{noise.shape[0]} | " \
                                                  f"{labels.shape[0]}"

        condi_inputs = torch.cat([noise, self.label_embedding(labels)], dim=-1)
        out = self.model(condi_inputs)
        imgs = out.reshape(out.size(0), *self.img_shape)

        return imgs

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    """
    This is discriminator that does not need image class as originally
    presented in https://arxiv.org/pdf/1411.1784.pdf.
    This follows:
    Paper SDDA: "Domain Impression: A Source Data Free Domain Adaptation
    Method", WACV, 2021.
    https://arxiv.org/abs/2102.09003.

    It predicts the validity of an input image INDEPENDENTLY OF THE CLASS
    CONDITION.
    """
    def __init__(self,
                 img_shape: list
                 ):
        super(Discriminator, self).__init__()


        assert isinstance(img_shape, list), type(img_shape)
        assert len(img_shape) == 3, len(img_shape)  # c, h, w
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

        self._initialize_weights()

    def forward(self, imgs: torch.Tensor):
        assert imgs.ndim == 4, imgs.ndim  # bsz, c, h, w

        inputs = torch.flatten(imgs, 1)
        validity = self.model(inputs)
        return validity

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class _Discriminator(nn.Module):
    """
    Standard discriminator that requires the image class as in:
    https://arxiv.org/pdf/1411.1784.pdf.
    """
    def __init__(self,
                 n_classes: int,
                 img_shape: list
                 ):
        super(_Discriminator, self).__init__()

        assert isinstance(n_classes, int), type(n_classes)
        assert n_classes > 0, n_classes
        self.n_classes = n_classes

        assert isinstance(img_shape, list), type(img_shape)
        assert len(img_shape) == 3, len(img_shape)  # c, h, w
        self.img_shape = img_shape

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

        self._initialize_weights()

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor):
        assert imgs.ndim == 4, imgs.ndim  # bsz, c, h, w

        assert imgs.shape[0] == labels.shape[0], f"{imgs.shape[0]} | " \
                                                 f"{labels.shape[0]}"

        inputs = torch.flatten(imgs, 1)
        conditional = self.label_embedding(labels)
        conditional_inputs = torch.cat([inputs, conditional], dim=-1)
        validity = self.model(conditional_inputs)
        return validity

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def run():
    from dlib.utils.reproducibility import set_seed
    from dlib.utils.shared import count_params

    set_seed(0)
    cuda = "0"
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else
                          "cpu")

    n_cls = 3
    latent_dim = 100
    img_shape = [3, 128, 128]
    bsz = 32
    g = Generator(n_classes=n_cls,
                  latent_dim=latent_dim,
                  img_shape=img_shape
                  ).to(device)
    d = Discriminator(img_shape=img_shape
                      ).to(device)

    print(f"Generator: {count_params(g)} params.")
    print(f"Discriminator: {count_params(d)} params.")

    z = torch.randn((bsz, latent_dim), device=device)
    fake_labels = torch.randint(0, n_cls, (bsz,), device=device).long()

    with torch.no_grad():
        fake_images = g(z, fake_labels)
        validity = d(fake_images)

    print(f"Generated images shape: {fake_images.shape}")
    print(f"Discriminator validity shape: {validity.shape}")


if __name__ == "__main__":
    run()




