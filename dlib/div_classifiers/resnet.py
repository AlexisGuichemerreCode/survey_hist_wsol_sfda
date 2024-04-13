"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import sys
from os.path import dirname, abspath, join


import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.div_classifiers.parts import AcolBase
from dlib.div_classifiers.parts import ADL
from dlib.div_classifiers.parts import spg
from dlib.div_classifiers.parts.util import normalize_tensor
from dlib.div_classifiers.util import remove_layer
from dlib.div_classifiers.util import replace_layer
from dlib.div_classifiers.util import initialize_weights

from dlib.div_classifiers.core import CoreClassifier

from dlib.configure import constants

__all__ = ['ResNet50Adl', 'ResNet50Acol', 'ResNet50Spg']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

IMG_NET_W_FD = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)

_ADL_POSITION = [[], [], [], [0], [0, 2]]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCam(CoreClassifier):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(ResNetCam, self).__init__()
        block = Bottleneck
        layers = [3, 4, 6, 3]

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        print(self.fc.weight.shape)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print(x.shape)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            print(cam_weights.shape, cam_weights.view(*feature_map.shape[:2],
                                                      1, 1).shape)
            print((cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).shape)
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return cams
        return {'logits': logits}

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class ClassifierA(nn.Module):
    def __init__(self, block_expansion: int, num_classes: int):
        super(ClassifierA, self).__init__()

        self.conv1 = nn.Conv2d(512 * block_expansion, 1024, 3, 1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(1024, 1024, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(1024, num_classes, 1, 1, padding=0)

        # SFUDA
        self.lin_ft = None  # linear features of the last layer in net to
        # produce image global class logits.

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def flush(self):
        self.lin_ft = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        ft = self.avg_pool(x)
        ft = ft.reshape(ft.size(0), -1)
        self.lin_ft = ft  # bsz, sz

        x = self.conv3(x)

        return x


class ResNet50Acol(AcolBase, CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 acol_drop_threshold=0.1,
                 scale_in=1.,
                 in_channels: int = 3,
                 spatial_dropout: float = 0.0
                 ):
        super(ResNet50Acol, self).__init__()

        self.encoder_name = constants.RESNET50
        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(in_channels, int)
        assert in_channels == 3
        self._in_channels = in_channels
        self.name = self.encoder_name
        self.encoder_weights = encoder_weights
        self.method = constants.METHOD_ACOL
        self.arch = constants.ACOLARCH

        assert isinstance(spatial_dropout, float), spatial_dropout
        assert 0. <= spatial_dropout <= 1., spatial_dropout
        self.p_dropout2d = spatial_dropout

        self.logits_dict = None

        block = Bottleneck
        layers = [3, 4, 6, 3]

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.label = None
        self.drop_threshold = acol_drop_threshold

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # FAUST, SFUDA, MC-dropout
        self.dropout_2d = torch.nn.Dropout2d(p=self.p_dropout2d, inplace=False)

        self.encoder_last_features = None
        self.encoder_n_out_channels = 512 * block.expansion

        self.classifier_A = ClassifierA(block.expansion, num_classes)

        self.classifier_B = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, 1, 1, padding=0),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        initialize_weights(self.modules(), init_mode='he')

        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()

    @property
    def lin_ft(self):
        # SFUDA
        # linear features of the last layer in net to
        # produce image global class logits.
        return self.classifier_A.lin_ft

    def flush(self):
        self.classifier_A.flush()
        self.encoder_last_features = None

    def forward(self, x, labels=None):
        x_shape = x.shape
        if self.scale_in != 1.:
            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self.x_in = x

        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        self.encoder_last_features = feature  # bsz, 512 * expansion, h, w

        feature = self.dropout_2d(feature)

        self.logits_dict = self._acol_logits(feature=feature, labels=labels,
                                             drop_threshold=self.drop_threshold)

        if labels is not None:
            normalized_a = normalize_tensor(
                self.logits_dict['feat_map_a'].detach().clone())
            normalized_b = normalize_tensor(
                self.logits_dict['feat_map_b'].detach().clone())
            feature_map = torch.max(normalized_a, normalized_b)
            self.cams = feature_map[range(batch_size), labels].detach()

        return self.logits_dict['logits']

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.classifier_A)
        self.freeze_part(self.classifier_B)

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, architecture_type=self.arch,
                              path=None, dataset_name='')

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


class ResNet50Spg(CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 scale_in=1.,
                 in_channels: int = 3,
                 spatial_dropout: float = 0.0
                 ):
        super(ResNet50Spg, self).__init__()

        self.encoder_name = constants.RESNET50

        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(in_channels, int)
        assert in_channels == 3
        self._in_channels = in_channels
        self.name = self.encoder_name
        self.encoder_weights = encoder_weights
        self.method = constants.METHOD_SPG
        self.arch = constants.SPGARCH

        assert isinstance(spatial_dropout, float), spatial_dropout
        assert 0. <= spatial_dropout <= 1., spatial_dropout
        self.p_dropout2d = spatial_dropout

        self.logits_dict = None

        block = Bottleneck
        layers = [3, 4, 6, 3]

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block=block, planes=64,
                                       blocks=layers[0],
                                       stride=1, split=False)
        self.layer2 = self._make_layer(block=block, planes=128,
                                       blocks=layers[1],
                                       stride=2, split=False)
        self.SPG_A1, self.SPG_A2 = self._make_layer(block=block,
                                                    planes=256,
                                                    blocks=layers[2],
                                                    stride=stride_l3,
                                                    split=True)
        self.layer4 = self._make_layer(block=block,
                                       planes=512,
                                       blocks=layers[3],
                                       stride=1, split=False)
        self.SPG_A4 = nn.Conv2d(512 * block.expansion, num_classes,
                                kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        # FAUST, SFUDA, MC-dropout
        self.dropout_2d = torch.nn.Dropout2d(p=self.p_dropout2d,
                                             inplace=False)

        self.encoder_last_features = None
        self.encoder_n_out_channels = 512 * block.expansion

        initialize_weights(self.modules(), init_mode='xavier')

        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()

        # SFUDA
        self.lin_ft = None  # linear features of the last layer in net to
        # produce image global class logits.

    def flush(self):
        self.lin_ft = None
        self.encoder_last_features = None

    def _make_layer(self, block, planes, blocks, stride, split=None):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        first_layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        other_layers = []
        for _ in range(1, blocks):
            other_layers.append(block(self.inplanes, planes))

        if split:
            return nn.Sequential(*first_layers), nn.Sequential(*other_layers)
        else:
            return nn.Sequential(*(first_layers + other_layers))

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.SPG_A4)
        self.freeze_part(self.SPG_C)

    def forward(self, x, labels=None):
        x_shape = x.shape
        if self.scale_in != 1.:
            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self.x_in = x

        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.SPG_A1(x)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.SPG_A2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.layer4(x)

        self.encoder_last_features = x  # bsz, 512 * expansion, h, w

        x = self.dropout_2d(x)

        # SFUDA
        ft = self.avgpool(x)
        ft = ft.reshape(ft.size(0), -1)
        self.lin_ft = ft  # bsz, sz

        feat_map = self.SPG_A4(x)

        logits_c = self.SPG_C(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        if labels is not None:
            feature_map = feat_map.clone().detach()
            self.cams = feature_map[range(batch_size), labels].detach()

        self.logits_dict = {'attention': attention,
                            'fused_attention': fused_attention,
                            'logits': logits, 'logits_b1': logits_b1,
                            'logits_b2': logits_b2, 'logits_c': logits_c}

        return logits

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, architecture_type=self.arch,
                              path=None, dataset_name='')


class ResNet50Adl(CoreClassifier):
    def __init__(self,
                 encoder_weights=constants.IMAGENET,
                 num_classes=1000,
                 large_feature_map=False,
                 adl_drop_rate=.4,
                 adl_drop_threshold=.1,
                 scale_in=1.,
                 in_channels: int = 3,
                 spatial_dropout: float = 0.0
                 ):
        super(ResNet50Adl, self).__init__()

        self.encoder_name = constants.RESNET50

        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(in_channels, int)
        assert in_channels == 3
        self._in_channels = in_channels
        self.name = self.encoder_name
        self.encoder_weights = encoder_weights
        self.method = constants.METHOD_ADL
        self.arch = constants.ADLARCH

        assert isinstance(spatial_dropout, float), spatial_dropout
        assert 0. <= spatial_dropout <= 1., spatial_dropout
        self.p_dropout2d = spatial_dropout

        self.logits_dict = None

        block = Bottleneck
        layers = [3, 4, 6, 3]

        self.stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.adl_drop_rate = adl_drop_rate
        self.adl_threshold = adl_drop_threshold

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=1,
                                       split=_ADL_POSITION[1])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2,
                                       split=_ADL_POSITION[2])
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=self.stride_l3,
                                       split=_ADL_POSITION[3])
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=1,
                                       split=_ADL_POSITION[4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # FAUST, SFUDA, MC-dropout
        self.dropout_2d = torch.nn.Dropout2d(p=self.p_dropout2d, inplace=False)

        self.encoder_last_features = None
        self.encoder_n_out_channels = 512 * block.expansion

        initialize_weights(self.modules(), init_mode='xavier')

        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()

        # SFUDA
        self.lin_ft = None  # linear features of the last layer in net to
        # produce image global class logits.

    def flush(self):
        self.lin_ft = None
        self.encoder_last_features = None

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.fc)

    def forward(self, x, labels=None):
        x_shape = x.shape
        if self.scale_in != 1.:
            h, w = x_shape[2], x_shape[3]
            x = F.interpolate(
                input=x,
                size=[int(h * self.scale_in), int(w * self.scale_in)],
                mode='bilinear',
                align_corners=True
            )

        self.x_in = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        self.encoder_last_features = x

        x = self.dropout_2d(x)

        # SFUDA -----
        ft = self.avgpool(x)
        ft = ft.reshape(ft.size(0), -1)
        self.lin_ft = ft  # bsz, sz
        # -----------

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if labels is not None:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            self.cams = cams.detach()

        self.logits_dict = {'logits': logits}
        return logits

    def _init_load_pretrained_w(self):
        load_pretrained_model(model=self, architecture_type=self.arch,
                              path=None, dataset_name='')

    def _make_layer(self, block, planes, blocks, stride, split=None):
        layers = self._layer(block, planes, blocks, stride)
        for pos in reversed(split):
            layers.insert(pos + 1, ADL(self.adl_drop_rate, self.adl_threshold))
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


def align_layer(state_dict):
    keys = [key for key in sorted(state_dict.keys())]
    for key in reversed(keys):
        move = 0
        if 'layer' not in key:
            continue
        key_sp = key.split('.')
        layer_idx = int(key_sp[0][-1])
        block_idx = key_sp[1]
        if not _ADL_POSITION[layer_idx]:
            continue

        for pos in reversed(_ADL_POSITION[layer_idx]):
            if pos < int(block_idx):
                move += 1

        key_sp[1] = str(int(block_idx) + move)
        new_key = '.'.join(key_sp)
        state_dict[new_key] = state_dict.pop(key)
    return state_dict


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'layer3.0.', 'SPG_A1.0.')
    state_dict = replace_layer(state_dict, 'layer3.1.', 'SPG_A2.0.')
    state_dict = replace_layer(state_dict, 'layer3.2.', 'SPG_A2.1.')
    state_dict = replace_layer(state_dict, 'layer3.3.', 'SPG_A2.2.')
    state_dict = replace_layer(state_dict, 'layer3.4.', 'SPG_A2.3.')
    state_dict = replace_layer(state_dict, 'layer3.5.', 'SPG_A2.4.')
    return state_dict


def load_pretrained_model(model, architecture_type, path=None,
                          dataset_name=None):
    strict_rule = True

    if path:
        state_dict = torch.load(os.path.join(path, 'resnet50.pth'))
    else:
        state_dict = load_url(model_urls['resnet50'], progress=True,
                              model_dir=IMG_NET_W_FD)

    if architecture_type == constants.ADLARCH:
        state_dict = align_layer(state_dict)
    elif architecture_type == constants.SPGARCH:
        state_dict = batch_replace_layer(state_dict)

    assert dataset_name != constants.ILSVRC

    if (dataset_name != constants.ILSVRC) or architecture_type in (
            constants.ACOLARCH,  constants.SPGARCH):
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def resnet50(architecture_type, pretrained=False, pretrained_path=None,
             **kwargs):

    model = {constants.ACOLARCH: ResNet50Acol,
             constants.SPGARCH: ResNet50Spg,
             constants.ADLARCH: ResNet50Adl}[architecture_type](**kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path)
    return model


def findout_names(model, architecture):

    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],  # features
        'resnet': ['layer4.', 'fc.'],           # CLASSIFIER
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],  # features
    }

    param_features = []
    param_classifiers = []

    def param_features_substring_list(architecture):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if architecture.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}"
                       .format(architecture))

    for name, parameter in model.named_parameters():
        if string_contains_any(
                name,
                param_features_substring_list(architecture)):
            if architecture in ('vgg16', 'inception_v3'):
                # param_features.append(parameter)
                print(name, '==>', 'feature')
            elif architecture == 'resnet50':
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
        else:
            if architecture in ('vgg16', 'inception_v3'):
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
            elif architecture == 'resnet50':
                # param_features.append(parameter)
                print(name, '==>', 'feature')

def run_resnet():
    import datetime as dt

    import torch.nn.functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt = {0: 'torch', 1: 'wsol'}

    pretrained = True
    encoder_weights = ''
    if pretrained:
        encoder_weights = constants.IMAGENET

    num_classes = 200
    for arch in [constants.ADLARCH, constants.SPGARCH, constants.ACOLARCH]:
        if arch == constants.ADLARCH:
            model = resnet50(arch, pretrained, num_classes=num_classes,
                             adl_drop_rate=.4, adl_drop_threshold=.1,
                             large_feature_map=True, spatial_dropout=0.5)

        elif arch == constants.ACOLARCH:
            model = resnet50(arch, pretrained, num_classes=num_classes,
                             acol_drop_threshold=.1,
                             large_feature_map=True, spatial_dropout=0.5)

        elif arch == constants.SPGARCH:
            model = resnet50(arch, pretrained, num_classes=num_classes,
                             large_feature_map=True, spatial_dropout=0.5)
        else:
            raise NotImplementedError

        model.to(device)
        bsize = 1
        h, w = 224, 224
        x = torch.rand(bsize, 3, 224, 224).to(device)
        labels = torch.zeros((bsize,), dtype=torch.long)
        model(x, labels=labels)

        t0 = dt.datetime.now()
        model(x, labels=labels)
        cams = model.cams
        print(cams.shape)
        if cams.shape != (1, h, w):
            tx = dt.datetime.now()
            full_cam = F.interpolate(
                input=cams.unsqueeze(0),
                size=[h, w],
                mode='bilinear',
                align_corners=True)
        print(x.shape, cams.shape)
        print('time: {}'.format(dt.datetime.now() - t0))


if __name__ == "__main__":
    run_resnet()

