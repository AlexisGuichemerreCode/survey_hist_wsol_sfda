import copy
import sys
from os.path import dirname, abspath
from typing import Optional, Union, List

import torch
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.base import STDClModel

from dlib import poolings

from dlib.configure import constants


class STDClassifier(STDClModel):
    """
    Standard classifier.
    """

    def __init__(
        self,
        task: str,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        aux_params: Optional[dict] = None,
        scale_in: float = 1.,
        spatial_dropout: float = 0.0
    ):
        super(STDClassifier, self).__init__()

        self.encoder_name = encoder_name
        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(spatial_dropout, float), spatial_dropout
        assert 0. <= spatial_dropout <= 1., spatial_dropout
        self.p_dropout2d = spatial_dropout

        self.x_in = None

        self.encoder = get_encoder(
            task,
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.dropout_2d = torch.nn.Dropout2d(p=self.p_dropout2d,
                                             inplace=False)

        assert aux_params is not None
        pooling_head = aux_params['pooling_head']
        self.support_background =aux_params['support_background']
        aux_params.pop('pooling_head')
        self.classification_head = poolings.__dict__[pooling_head](
            in_channels=self.encoder.out_channels[-1], **aux_params
        )

        self.name = "u-{}".format(encoder_name)
        self.initialize()

        # SFUDA
        self.src_ft = None  # can be used to hold source features. assigned
        # from outside

        self.encoder_last_features = None  # features of the output of the
        # last layer of the encoder. expected to be 2d: bsz, d, h, w.

    @property
    def encoder_n_out_channels(self) -> int:
        """
        Number of output channels in the last layer of the encoder.
        """
        return self.encoder.out_channels[-1]

    @property
    def lin_ft(self):
        # linear features to obtain image class logits.
        return self.classification_head.lin_ft

    @property
    def get_linear_weights(self):
        # weight of the classifier
        return self.classification_head.get_linear_weights()
    
    def flush(self):
        if hasattr(self.classification_head, 'flush'):
            self.classification_head.flush()

        self.src_ft = None
        self.encoder_last_features = None

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.classification_head.freeze_cl_hypothesis()

    def freeze_all_params(self):

        for module in (self.modules()):

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

    def set_all_bn_running_stats_to_be_updatable(self):
        for module in (self.modules()):

            if isinstance(module, torch.nn.BatchNorm3d):
                module.train()

            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()

            if isinstance(module, torch.nn.BatchNorm1d):
                module.train()

    def freeze_all_params_keep_bn_stats_updatable(self):
        """
        Utility: AdaDSA method for SFUDA method.
        Freeze all parameters.
        :return:
        """
        self.freeze_all_params()
        self.set_all_bn_running_stats_to_be_updatable()


def findout_names(model, architecture):
    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['encoder.features.'],  # features
        'resnet': ['encoder.layer4.', 'classification_head.'],  # CLASSIFIER
        'inception': ['encoder.Mixed', 'encoder.Conv2d_1', 'encoder.Conv2d_2',
                      'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
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
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                # param_features.append(parameter)
                print(name, '==>', 'feature')
            elif architecture == constants.RESNET50:
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
        else:
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
            elif architecture == constants.RESNET50:
                # param_features.append(parameter)
                print(name, '==>', 'feature')


def list_modules(model):

    print('Listing ...')

    for module in model.modules():
        print(module)

        for param in module.parameters():
            param.requires_grad = False

        if isinstance(module, torch.nn.BatchNorm2d):
            print('batch2d')

        if isinstance(module, torch.nn.BatchNorm1d):
            print('batch1d')

        if isinstance(module, torch.nn.Dropout):
            print('dropout')


    print('End listing ...')

def name_parts(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            name_parts(module)

        if isinstance(module, torch.nn.BatchNorm2d):
            print(n, module)

if __name__ == "__main__":
    import datetime as dt
    import dlib
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed
    from dlib.sf_uda.adadsa import replace_all_bn_with_adadsa_bn

    set_seed(0)
    cuda = "1"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    vgg_encoders = dlib.encoders.vgg_encoders
    in_channels = 3
    SZ = 224
    sample = torch.rand((32, in_channels, SZ, SZ)).to(DEVICE)
    encoders = [constants.RESNET50, constants.INCEPTIONV3, constants.VGG16]

    amp = True

    for encoder_name in encoders:

        announce_msg("Testing backbone {}".format(encoder_name))
        if encoder_name == constants.VGG16:
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            encoder_depth = 5

        # task: STD_CL
        model = STDClassifier(
            task=constants.STD_CL,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=constants.IMAGENET,
            in_channels=in_channels,
            aux_params=dict(pooling_head="WildCatCLHead", classes=200)
        ).to(DEVICE)
        announce_msg(f"TESTING: {model} -- amp={amp} \n"
                     f" {model.get_info_nbr_params()}"
                     )
        t0 = dt.datetime.now()
        with torch.no_grad():
            with autocast(enabled=amp):
                cl_logits = model(sample).detach()

        torch.cuda.empty_cache()
        with torch.no_grad():
            with autocast(enabled=amp):
                cl_logitsx = model(sample[0].unsqueeze(0)).detach()
        print('forward time {}'.format(dt.datetime.now() - t0))
        print("x: {} \t cl_logits: {}".format(sample.shape, cl_logits.shape))
        print('logits', cl_logits)
        val, ind = torch.sort(cl_logits.cpu(), dim=1, descending=True,
                              stable=True)
        print(val, ind)

        # findout_names(model, encoder_name)

        list_modules(model)
        announce_msg(f"NBR-PARAM BEFORE BN-FUSION: {model} \n"
                     f" {model.get_info_nbr_params()}"
                     )
        replace_all_bn_with_adadsa_bn(model,
                                      copy.deepcopy(model),
                                      copy.deepcopy(model),
                                      torch.nn.BatchNorm2d,
                                      DEVICE)
        list_modules(model)
        announce_msg(f"NBR-PARAM AFTER BN-FUSION: {model} \n"
                     f" {model.get_info_nbr_params()}"
                     )
        # name_parts(model)
        sys.exit()
