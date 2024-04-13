# Sel-contained-as-possible module handles parsing the input using argparse.
# handles seed, and initializes some modules for reproducibility.
import copy
import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import argparse
from copy import deepcopy
import warnings
import subprocess
import fnmatch
import glob
import shutil
import datetime as dt

import yaml
import munch
import numpy as np
import torch

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.configure import config
from dlib.utils import reproducibility

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc
from dlib.utils.shared import get_tag_device
from dlib.utils.tools import get_tag


def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd)


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def find_files_pattern(fd_in_, pattern_):
    assert os.path.exists(fd_in_), "Folder {} does not exist.".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def str_to_int_list(s):
    return [int(item) for item in s.strip('[]').split(',')]


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_args(args: dict, eval: bool = False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=None,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--MYSEED", type=str, default=None, help="Seed.")
    parser.add_argument("--debug_subfolder", type=str, default=None,
                        help="Name of subfold for debugging. Default: ''.")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Name of the dataset.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold of dataset.")
    parser.add_argument("--magnification", type=str, default=None,
                        help="Magnififcation of BreakHis dataset.")
    parser.add_argument("--runmode", type=str, default=None,
                        help="Run mode: hyper-parameter search or final.")

    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes in the dataset.")

    parser.add_argument("--crop_size", type=int, default=None,
                        help="Crop size (int) of the patches in training.")
    parser.add_argument("--resize_size", type=int, default=None,
                        help="Resize image into this size before processing.")

    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Max epoch.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size (optimizer).")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers for dataloader multi-proc.")
    parser.add_argument("--exp_id", type=str, default=None, help="Exp id.")
    parser.add_argument("--verbose", type=str2bool, default=None,
                        help="Verbosity (bool).")
    parser.add_argument("--fd_exp", type=str, default=None,
                        help="Relative path to exp folder.")

    # ======================================================================
    #                      WSOL
    # ======================================================================
    parser.add_argument('--data_root', default=None,
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default=None)
    parser.add_argument('--mask_root', default=None,
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool,
                        default=None,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=None,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')
    parser.add_argument('--cam_curve_interval', type=float, default=None,
                        help='CAM curve interval')
    parser.add_argument('--multi_contour_eval', type=str2bool, default=None)
    parser.add_argument('--multi_iou_eval', type=str2bool, default=None)
    parser.add_argument('--box_v2_metric', type=str2bool, default=None)
    parser.add_argument('--eval_checkpoint_type', type=str, default=None)
    # ======================================================================
    #                      OPTIMIZER
    # ======================================================================
    # opt0: optimizer for the model.
    parser.add_argument("--opt__name_optimizer", type=str, default=None,
                        help="Name of the optimizer 'sgd', 'adam'.")
    parser.add_argument("--opt__lr", type=float, default=None,
                        help="Learning rate (optimizer)")
    parser.add_argument("--opt__momentum", type=float, default=None,
                        help="Momentum (optimizer)")
    parser.add_argument("--opt__dampening", type=float, default=None,
                        help="Dampening for Momentum (optimizer)")
    parser.add_argument("--opt__nesterov", type=str2bool, default=None,
                        help="Nesterov or not for Momentum (optimizer)")
    parser.add_argument("--opt__weight_decay", type=float, default=None,
                        help="Weight decay (optimizer)")
    parser.add_argument("--opt__beta1", type=float, default=None,
                        help="Beta1 for adam (optimizer)")
    parser.add_argument("--opt__beta2", type=float, default=None,
                        help="Beta2 for adam (optimizer)")
    parser.add_argument("--opt__eps_adam", type=float, default=None,
                        help="eps for adam (optimizer)")
    parser.add_argument("--opt__amsgrad", type=str2bool, default=None,
                        help="amsgrad for adam (optimizer)")
    parser.add_argument("--opt__lr_scheduler", type=str2bool, default=None,
                        help="Whether to use or not a lr scheduler")
    parser.add_argument("--opt__name_lr_scheduler", type=str, default=None,
                        help="Name of the lr scheduler")
    parser.add_argument("--opt__gamma", type=float, default=None,
                        help="Gamma of the lr scheduler. (mystep)")
    parser.add_argument("--opt__last_epoch", type=int, default=None,
                        help="Index last epoch to stop adjust LR(mystep)")
    parser.add_argument("--opt__min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--opt__t_max", type=float, default=None,
                        help="T_max, maximum epochs to restart. (cosine)")
    parser.add_argument("--opt__step_size", type=int, default=None,
                        help="Step size for lr scheduler.")
    parser.add_argument("--opt__lr_classifier_ratio", type=float, default=None,
                        help="Multiplicative factor for the classifier head "
                             "learning rate.")

    # ======================================================================
    #                              MODEL
    # ======================================================================
    parser.add_argument("--arch", type=str, default=None,
                        help="model's name.")
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="Name of the backbone")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Input channels number.")
    parser.add_argument("--strict", type=str2bool, default=None,
                        help="strict mode for loading weights.")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Pre-trained weights.")
    parser.add_argument("--path_pre_trained", type=str, default=None,
                        help="Absolute/relative path to file of weights.")
    parser.add_argument("--support_background", type=str2bool, default=None,
                        help="use or not 1 extra plan for background cams.")
    parser.add_argument("--scale_in", type=float, default=None,
                        help="How much to scale the input.")

    parser.add_argument("--freeze_cl", type=str2bool, default=None,
                        help="whether or not to freeze the classifier.")
    parser.add_argument("--folder_pre_trained_cl", type=str, default=None,
                        help="NAME of folder containing classifier's "
                             "weights.")
    parser.add_argument("--spatial_dropout", type=float, default=None,
                        help="2d dropout at the last feature layer ([0, 1.]).")

    # ======================================================================
    #                    CLASSIFICATION SPATIAL POOLING
    # ======================================================================
    parser.add_argument("--method", type=str, default=None,
                        help="Name of method.")
    parser.add_argument("--spatial_pooling", type=str, default=None,
                        help="Name of spatial pooling for classification.")
    # ======================================================================
    #                        WILDCAT POOLING
    # ======================================================================

    parser.add_argument("--wc_alpha", type=float, default=None,
                        help="Alpha (classifier, wildcat)")
    parser.add_argument("--wc_kmax", type=float, default=None,
                        help="Kmax (classifier, wildcat)")
    parser.add_argument("--wc_kmin", type=float, default=None,
                        help="Kmin (classifier, wildcat)")
    parser.add_argument("--wc_dropout", type=float, default=None,
                        help="Dropout (classifier, wildcat)")
    parser.add_argument("--wc_modalities", type=int, default=None,
                        help="Number of modalities (classifier, wildcat)")

    parser.add_argument("--lse_r", type=float, default=None,
                        help="LSE r pooling.")

    # ======================================================================
    #                         EXTRA - MODE
    # ======================================================================

    parser.add_argument("--seg_mode", type=str, default=None,
                        help="Segmentation mode.")
    parser.add_argument("--task", type=str, default=None,
                        help="Type of the task.")
    parser.add_argument("--multi_label_flag", type=str2bool, default=None,
                        help="Whether the dataset is multi-label.")
    # ======================================================================
    #                         ELB
    # ======================================================================
    parser.add_argument("--elb_init_t", type=float, default=None,
                        help="Init t for elb.")
    parser.add_argument("--elb_max_t", type=float, default=None,
                        help="Max t for elb.")
    parser.add_argument("--elb_mulcoef", type=float, default=None,
                        help="Multi. coef. for elb..")

    # ======================================================================
    #                         CONSTRAINTS
    # ======================================================================
    parser.add_argument("--crf_fc", type=str2bool, default=None,
                        help="CRF over fcams flag.")
    parser.add_argument("--crf_lambda", type=float, default=None,
                        help="Lambda for crf flag.")
    parser.add_argument("--crf_sigma_rgb", type=float, default=None,
                        help="sigma rgb of crf flag.")
    parser.add_argument("--crf_sigma_xy", type=float, default=None,
                        help="sigma xy for crf flag.")
    parser.add_argument("--crf_scale", type=float, default=None,
                        help="scale factor for crf flag.")
    parser.add_argument("--crf_start_ep", type=int, default=None,
                        help="epoch start crf loss.")
    parser.add_argument("--crf_end_ep", type=int, default=None,
                        help="epoch end crf loss. use -1 for end training.")

    parser.add_argument("--entropy_fc", type=str2bool, default=None,
                        help="Entropy over fcams flag.")
    parser.add_argument("--entropy_fc_lambda", type=float, default=None,
                        help="lambda for entropy over fcams flag.")

    parser.add_argument("--max_sizepos_fc", type=str2bool, default=None,
                        help="Max size pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_lambda", type=float, default=None,
                        help="lambda for max size low pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_start_ep", type=int, default=None,
                        help="epoch start maxsz loss.")
    parser.add_argument("--max_sizepos_fc_end_ep", type=int, default=None,
                        help="epoch end maxsz. -1 for end training.")

    parser.add_argument("--im_rec", type=str2bool, default=None,
                        help="image reconstruction flag.")
    parser.add_argument("--im_rec_lambda", type=float, default=None,
                        help="Lambda for image reconstruction.")
    parser.add_argument("--im_rec_elb", type=str2bool, default=None,
                        help="use/not elb for image reconstruction.")

    parser.add_argument("--sl_fc", type=str2bool, default=None,
                        help="Self-learning over fcams.")
    parser.add_argument("--sl_fc_lambda", type=float, default=None,
                        help="Lambda for self-learning fcams.")
    parser.add_argument("--sl_start_ep", type=int, default=None,
                        help="Start epoch for self-learning fcams.")
    parser.add_argument("--sl_end_ep", type=int, default=None,
                        help="End epoch for self-learning fcams.")
    parser.add_argument("--sl_min", type=int, default=None,
                        help="MIN for self-learning fcams.")
    parser.add_argument("--sl_max", type=int, default=None,
                        help="MAX for self-learning fcams.")
    parser.add_argument("--sl_ksz", type=int, default=None,
                        help="Kernel size for dilation for self-learning "
                             "fcams.")
    parser.add_argument("--sl_min_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "background to sample from.")
    parser.add_argument("--sl_fg_erode_k", type=int, default=None,
                        help="Kernel size of erosion for foreground.")
    parser.add_argument("--sl_fg_erode_iter", type=int, default=None,
                        help="Number of time to perform erosion over "
                             "foreground.")
    parser.add_argument("--sl_min_ext", type=int, default=None,
                        help="MIN extent for self-learning fcams.")
    parser.add_argument("--sl_max_ext", type=int, default=None,
                        help="MAX extent for self-learning fcams.")
    parser.add_argument("--sl_block", type=int, default=None,
                        help="Size of the blocks for self-learning fcams.")

    parser.add_argument("--seg_ignore_idx", type=int, default=None,
                        help="Ignore index for segmentation.")
    parser.add_argument("--amp", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "training.")
    parser.add_argument("--amp_eval", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "inference.")

    parser.add_argument('--adl_drop_rate', type=float, default=None,
                        help='Float.drop-rate for ADL.')
    parser.add_argument('--adl_drop_threshold', type=float, default=None,
                        help='Float. threshold for ADL.')
    parser.add_argument('--adl_large_feature_map', type=str2bool, default=None,
                        help='Use/not large feature maps for ADL.')

    parser.add_argument('--acol_drop_threshold', type=float, default=None,
                        help='Float. threshold for ACOL.')
    parser.add_argument('--acol_large_feature_map', type=str2bool, default=None,
                        help='Use/not large feature maps for ACOL.')

    parser.add_argument('--spg_threshold_1h', type=float, default=None,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_1l', type=float, default=None,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2h', type=float, default=None,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2l', type=float, default=None,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3h', type=float, default=None,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3l', type=float, default=None,
                        help='SPG threshold')
    parser.add_argument('--spg_large_feature_map', type=str2bool, default=None,
                        help='Use/not large feature maps for SPG.')

    parser.add_argument('--has_grid_size', type=int, default=None,
                        help='HAS patch size. int.')
    parser.add_argument('--has_drop_rate', type=float, default=None,
                        help='HAS. percentage of patches to be dropped.[0, 1[')

    parser.add_argument('--cutmix_beta', type=float, default=None,
                        help='CUTMIX beta.')
    parser.add_argument('--cutmix_prob', type=float, default=None,
                        help='CUTMIX. probablity to do it over a minibatch.')

    parser.add_argument('--mil_mid_channels', type=int, default=None,
                        help='Deep mil mid-channels.')
    parser.add_argument('--mil_gated', type=str2bool, default=None,
                        help='Deep mil attention type.')

    parser.add_argument('--maxmin_w', type=float, default=None,
                        help='maxmin w.')
    parser.add_argument('--minmax_lambda_size', type=float, default=None,
                        help='maxmin lambda size.')
    parser.add_argument('--minmax_lambda_neg', type=float, default=None,
                        help='minmax lambda negative info.')

    parser.add_argument('--prm_ks', type=int, default=None,
                        help='PRM: kernel size.')
    parser.add_argument('--prm_st', type=int, default=None,
                        help='PRM: kernel stride.')

    # NEGEV method
    parser.add_argument('--sl_ng', type=str2bool, default=None,
                        help='negev: self-learning on/off.')
    parser.add_argument('--sl_ng_seeder', type=str, default=None,
                        help='negev: self-learning: seeder type.')
    parser.add_argument('--sl_ng_lambda', type=float, default=None,
                        help='negev: self-learning: lambda.')
    parser.add_argument('--sl_ng_start_ep', type=int, default=None,
                        help='negev: self-learning: start epoch.')
    parser.add_argument('--sl_ng_end_ep', type=int, default=None,
                        help='negev: self-learning: end epoch.')
    parser.add_argument('--sl_ng_min', type=int, default=None,
                        help='negev: self-learning: seeds to sample '
                             'background.')
    parser.add_argument('--sl_ng_max', type=int, default=None,
                        help='negev: self-learning: seeds to sample '
                             'foreground.')
    parser.add_argument('--sl_ng_ksz', type=int, default=None,
                        help='negev: self-learning: kernel size dilation.')
    parser.add_argument('--sl_ng_min_ext', type=int, default=None,
                        help='negev: self-learning: extent background region.')
    parser.add_argument('--sl_ng_max_ext', type=int, default=None,
                        help='negev: self-learning: extent foreground region.')
    parser.add_argument('--sl_ng_block', type=int, default=None,
                        help='negev: self-learning: size sampling block for '
                             'seeds.')
    parser.add_argument('--sl_ng_min_p', type=float, default=None,
                        help='negev: self-learning: percentage to be '
                             'considered background.')
    parser.add_argument('--sl_ng_fg_erode_k', type=int, default=None,
                        help='negev: self-learning: Erosion kernel size.')
    parser.add_argument('--sl_ng_fg_erode_iter', type=int, default=None,
                        help='negev: self-learning: number erosion iterations.')

    parser.add_argument('--crf_ng', type=str2bool, default=None,
                        help='negev: crf: on/off.')
    parser.add_argument('--crf_ng_lambda', type=float, default=None,
                        help='negev: crf: lambda.')
    parser.add_argument('--crf_ng_sigma_rgb', type=float, default=None,
                        help='negev: crf: sigma rgb.')
    parser.add_argument('--crf_ng_sigma_xy', type=float, default=None,
                        help='negev: crf: sigma xy.')
    parser.add_argument('--crf_ng_scale', type=float, default=None,
                        help='negev: crf: scale image.')
    parser.add_argument('--crf_ng_start_ep', type=int, default=None,
                        help='negev: crf: start epoch.')
    parser.add_argument('--crf_ng_end_ep', type=int, default=None,
                        help='negev: crf: end epoch.')

    parser.add_argument('--jcrf_ng', type=str2bool, default=None,
                        help='negev: jcrf: on/off.')
    parser.add_argument('--jcrf_ng_lambda', type=float, default=None,
                        help='negev: jcrf: lambda.')
    parser.add_argument('--jcrf_ng_sigma_rgb', type=float, default=None,
                        help='negev: jcrf: sigma rgb.')
    parser.add_argument('--jcrf_ng_scale', type=float, default=None,
                        help='negev: jcrf: scale image.')
    parser.add_argument('--jcrf_ng_start_ep', type=int, default=None,
                        help='negev: jcrf: start epoch.')
    parser.add_argument('--jcrf_ng_end_ep', type=int, default=None,
                        help='negev: jcrf: end epoch.')
    parser.add_argument('--jcrf_ng_pair_mode', type=str, default=None,
                        help='negev: jcrf: pairing mode.')
    parser.add_argument('--jcrf_ng_n', type=int, default=None,
                        help='negev: jcrf: number of samples to pair with.')

    parser.add_argument('--max_sizepos_ng', type=str2bool, default=None,
                        help='negev: size const: on/off.')
    parser.add_argument('--max_sizepos_ng_lambda', type=float, default=None,
                        help='negev: size const: lambda.')
    parser.add_argument('--max_sizepos_ng_start_ep', type=int, default=None,
                        help='negev: size const: start epoch.')
    parser.add_argument('--max_sizepos_ng_end_ep', type=int, default=None,
                        help='negev: size const: end epoch.')

    parser.add_argument('--neg_samples_ng', type=str2bool, default=None,
                        help='negev: negative samples: on/off.')
    parser.add_argument('--neg_samples_ng_lambda', type=float, default=None,
                        help='negev: negative samples: lambda.')
    parser.add_argument('--neg_samples_ng_start_ep', type=int, default=None,
                        help='negev: negative samples: start epoch.')
    parser.add_argument('--neg_samples_ng_end_ep', type=int, default=None,
                        help='negev: negative samples: end epoch.')

    parser.add_argument('--negev_ptretrained_cl_cp', type=str, default=None,
                        help='negev: checkpoint for pretrained classifier.')

    parser.add_argument('--ce_label_smoothing', type=float, default=None,
                        help='CE: label smoothing -- source model.')
    
    #SAT params. use default values from the SAT code (defined in cofig).
    parser.add_argument('--sat_drop_rate', type=float, default=None,
                        help='SAT. percentage of patches to be dropped.[0, 1]')
    parser.add_argument('--sat_drop_path_rate', type=float, default=None,
                        help='SAT. drop path for stochastic depth same value in SAT code')
    parser.add_argument('--sat_area_th', type=float, default=None,
                        help='SAT. area threshold..')

    # Domain adaptation
    parser.add_argument('--sf_uda', type=str2bool, default=None,
                        help='If ture, we set the source free unsupervised '
                             'domain adaptation setup, if not, '
                             'it is standard classification/segmentation.')

    parser.add_argument('--sf_uda_source_ds', type=str, default=None,
                        help='name of the source dataset.')
    parser.add_argument('--sf_uda_source_ds_fold', type=int, default=None,
                        help='Fold of the source dataset.')
    parser.add_argument('--sf_uda_source_encoder_name', type=str, default=None,
                        help='Name of the source encoder name.')
    parser.add_argument('--sf_uda_source_checkpoint_type', type=str, default=None,
                        help='Check point type of the source model.')
    parser.add_argument('--sf_uda_source_wsol_method', type=str, default=None,
                        help='Name of the source wsol method.')
    parser.add_argument('--sf_uda_source_wsol_arch', type=str, default=None,
                        help='Name of the source wsol arch.')
    parser.add_argument('--sf_uda_source_wsol_spatial_pooling', type=str, default=None,
                        help='Name of the source wsol spatial pooling '
                             'technique.')
    parser.add_argument('--sf_uda_source_folder', type=str, default=None,
                        help='Full path to the source folder. if it is '
                             'empty "", we determine automatically this '
                             'folder based on the provided info.')

    # methods SFUDA
    # 1- SHOT
    parser.add_argument('--shot', type=str2bool, default=None,
                        help='USE/NOT SHOT method for SFUDA.')
    parser.add_argument('--shot_freq_epoch', type=int, default=None,
                        help='Epoch frequency to recompute the image class '
                             'pseudo-label.')
    parser.add_argument('--shot_dist_type', type=str, default=None,
                        help='Type distance for SHOT method.')

    # 2- FAUST
    parser.add_argument('--faust', type=str2bool, default=None,
                        help='USE/NOT FAUST method for SFUDA.')
    parser.add_argument('--faust_n_views', type=int, default=None,
                        help='Number of views for FAUST.')

    # 2- NRC
    parser.add_argument('--nrc', type=str2bool, default=None,
                        help='USE/NOT NRC method for SFUDA.')
    parser.add_argument('--r_nrc', type=float, default=None,
                        help='Affinity Value for NRC.')
    # SFUDA DE
    parser.add_argument('--sfde', type=str2bool, default=None,
                        help='USE/NOT SFDE method for SFUDA.')
    parser.add_argument('--sfde_threshold', type=float, default=None,
                        help='Threshold, 0<T<=1.')
    # 2- CDCL
    parser.add_argument('--cdcl', type=str2bool, default=None,
                        help='USE/NOT CDCL method for SFUDA.')
    parser.add_argument('--cdcl_threshold', type=float, default=None,
                        help='Pourcentage of sample to keep for the method for each class.')


    # losses SFUDA
    parser.add_argument('--ce_pseudo_lb', type=str2bool, default=None,
                        help='Cross-entropy over image class pseudo-labels.')
    parser.add_argument('--ce_pseudo_lb_lambda', type=float, default=None,
                        help='ce_pseudo_lb: Lambda, >=0.')
    parser.add_argument('--ce_pseudo_lb_start_ep', type=int, default=None,
                        help='ce_pseudo_lb: start epoch to apply this loss.')
    parser.add_argument('--ce_pseudo_lb_end_ep', type=int, default=None,
                        help='ce_pseudo_lb: epoch to end using this loss.')
    parser.add_argument('--ce_pseudo_lb_smooth', type=float, default=None,
                        help='ce_pseudo_lb: smoothing value [0., 1.].')

    parser.add_argument('--ent_pseudo_lb', type=str2bool, default=None,
                        help='Minimize entropy over image class prediction. '
                             'See config.py')
    parser.add_argument('--ent_pseudo_lb_lambda', type=float, default=None,
                        help='ent_pseudo_lb: Lambda, >=0.')
    parser.add_argument('--ent_pseudo_lb_start_ep', type=int, default=None,
                        help='ent_pseudo_lb: start epoch to apply this loss.')
    parser.add_argument('--ent_pseudo_lb_end_ep', type=int, default=None,
                        help='ent_pseudo_lb: epoch to end using this loss.')

    parser.add_argument('--div_pseudo_lb', type=str2bool, default=None,
                        help='Diversity loss over image global class '
                             'predictions. See config.py')
    parser.add_argument('--div_pseudo_lb_lambda', type=float, default=None,
                        help='div_pseudo_lb: Lambda, >=0.')
    parser.add_argument('--div_pseudo_lb_start_ep', type=int, default=None,
                        help='div_pseudo_lb: start epoch to apply this loss.')
    parser.add_argument('--div_pseudo_lb_end_ep', type=int, default=None,
                        help='div_pseudo_lb: epoch to end using this loss.')
    
    parser.add_argument('--cdd_lambda', type=float, default=None,
                        help='Lambda, >=0.')
    parser.add_argument('--cdd_pseudo_lb', type=str2bool, default=None,
                        help='CDD loss over image class prediction')
    parser.add_argument('--cdd_pseudo_lb_kernel_num', type=str_to_int_list, default=None,
                        help='cdd_pseudo_lb_kernel_num: type [x,x].')
    parser.add_argument('--cdd_pseudo_lb_kernel_mul', type=str_to_int_list, default=None,
                        help='cdd_pseudo_lb_kernel_mul: type [y,y]')
    parser.add_argument('--cdd_pseudo_lb_num_layers', type=int, default=None,
                        help='Number of layers')
    parser.add_argument('--cdd_pseudo_lb_num_classes', type=int, default=None,
                        help='Number of classes')
    parser.add_argument('--cdd_pseudo_lb_intra_only', type=str2bool, default=None,
                        help='Intra only or intra + inter for CDD') 
    parser.add_argument('--cdd_variance', type=float, default=None,
                        help='Float to controle variance of the gaussian estimation for source features distribution')
    

    parser.add_argument('--views_ft_consist', type=str2bool, default=None,
                        help='Features views consistency [FAUST]. See '
                             'config.py')
    parser.add_argument('--views_ft_consist_lambda', type=float, default=None,
                        help='views_ft_consist: Lambda, >=0.')
    parser.add_argument('--views_ft_consist_start_ep', type=int, default=None,
                        help='views_ft_consist: start epoch to apply this '
                             'loss.')
    parser.add_argument('--views_ft_consist_end_ep', type=int, default=None,
                        help='views_ft_consist: epoch to end using this loss.')

    parser.add_argument('--ce_views_soft_pl', type=str2bool, default=None,
                        help='Soft-class pseudo-labels views consistency ['
                             'FAUST]. See config.py')
    parser.add_argument('--ce_views_soft_pl_t', type=float, default=None,
                        help='ce_views_soft_pl: softmax heat, > 0.')
    parser.add_argument('--ce_views_soft_pl_lambda', type=float, default=None,
                        help='ce_views_soft_pl: Lambda, >=0.')
    parser.add_argument('--ce_views_soft_pl_start_ep', type=int, default=None,
                        help='ce_views_soft_pl: start epoch to apply this '
                             'loss.')
    parser.add_argument('--ce_views_soft_pl_end_ep', type=int, default=None,
                        help='ce_views_soft_pl: epoch to end using this loss.')

    parser.add_argument('--mc_var_prob', type=str2bool, default=None,
                        help='Soft-class pseudo-labels views consistency ['
                             'FAUST]. See config.py')
    parser.add_argument('--mc_var_prob_n_dout', type=int, default=None,
                        help='mc_var_prob: number of time to do mc-dropout, '
                             '> 1')
    parser.add_argument('--mc_var_prob_lambda', type=float, default=None,
                        help='mc_var_prob: Lambda, >=0.')
    parser.add_argument('--mc_var_prob_start_ep', type=int, default=None,
                        help='mc_var_prob: start epoch to apply this loss.')
    parser.add_argument('--mc_var_prob_end_ep', type=int, default=None,
                        help='mc_var_prob: epoch to end using this loss.')

    parser.add_argument('--min_prob_entropy', type=str2bool, default=None,
                        help='Soft-class pseudo-labels views consistency ['
                             'FAUST]. See config.py')
    parser.add_argument('--min_prob_entropy_lambda', type=float, default=None,
                        help='min_prob_entropy: Lambda, >=0.')
    parser.add_argument('--min_prob_entropy_start_ep', type=int, default=None,
                        help='min_prob_entropy: start epoch to apply this '
                             'loss.')
    parser.add_argument('--min_prob_entropy_end_ep', type=int, default=None,
                        help='min_prob_entropy: epoch to end using this loss.')

    # 3- AdaDSA
    parser.add_argument('--adadsa', type=str2bool, default=None,
                        help='USE/NOT AdaDSA method for SFUDA.')
    parser.add_argument('--adadsa_a', type=float, default=None,
                        help='a (scaling of lambda) for AdaDSA.')
    parser.add_argument('--adadsa_eval_batch_size', type=int, default=None,
                        help='Evaluation batch size to estimate Batchnorm '
                             'stats over target set for AdaDSA. '
                             'Use -1 to use entire target set at once.')

    # 4- SDDA
    parser.add_argument('--sdda', type=str2bool, default=None,
                        help='USE/NOT SDDA method for SFUDA.')
    parser.add_argument('--sdda_gan_type', type=str, default=None,
                        help='GAN type for SDDA.')
    parser.add_argument('--sdda_gan_latent_dim', type=int, default=None,
                        help='GAN latent embedding [SDDA].')
    parser.add_argument('--sdda_gan_h', type=int, default=None,
                        help='Height for GAN image generation [SDDA].')
    parser.add_argument('--sdda_gan_w', type=int, default=None,
                        help='Width for GAN image generation [SDDA].')
    parser.add_argument('--adaptation_start_epoch', type=int, default=None,
                        help='When to start adaption phase [SDDA].')

    # 4.1 SDDA ADV. discriminator optimizer
    parser.add_argument("--sdda_d__name_optimizer", type=str, default=None,
                        help="Name of the optimizer 'sgd', 'adam'.")
    parser.add_argument("--sdda_d__lr", type=float, default=None,
                        help="Learning rate (optimizer)")
    parser.add_argument("--sdda_d__momentum", type=float, default=None,
                        help="Momentum (optimizer)")
    parser.add_argument("--sdda_d__dampening", type=float, default=None,
                        help="Dampening for Momentum (optimizer)")
    parser.add_argument("--sdda_d__nesterov", type=str2bool, default=None,
                        help="Nesterov or not for Momentum (optimizer)")
    parser.add_argument("--sdda_d__weight_decay", type=float, default=None,
                        help="Weight decay (optimizer)")
    parser.add_argument("--sdda_d__beta1", type=float, default=None,
                        help="Beta1 for adam (optimizer)")
    parser.add_argument("--sdda_d__beta2", type=float, default=None,
                        help="Beta2 for adam (optimizer)")
    parser.add_argument("--sdda_d__eps_adam", type=float, default=None,
                        help="eps for adam (optimizer)")
    parser.add_argument("--sdda_d__amsgrad", type=str2bool, default=None,
                        help="amsgrad for adam (optimizer)")
    parser.add_argument("--sdda_d__lr_scheduler", type=str2bool, default=None,
                        help="Whether to use or not a lr scheduler")
    parser.add_argument("--sdda_d__name_lr_scheduler", type=str, default=None,
                        help="Name of the lr scheduler")
    parser.add_argument("--sdda_d__gamma", type=float, default=None,
                        help="Gamma of the lr scheduler. (mystep)")
    parser.add_argument("--sdda_d__last_epoch", type=int, default=None,
                        help="Index last epoch to stop adjust LR(mystep)")
    parser.add_argument("--sdda_d__min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--sdda_d__t_max", type=float, default=None,
                        help="T_max, maximum epochs to restart. (cosine)")
    parser.add_argument("--sdda_d__step_size", type=int, default=None,
                        help="Step size for lr scheduler.")

    # 4.2 SDDA: losses

    parser.add_argument("--adv_d_sdda", type=str2bool, default=None,
                        help="Adversarial discriminative loss. [SDDA]")
    parser.add_argument("--adv_d_sdda_lambda", type=float, default=None,
                        help="Lambda for adv_d_sdda loss. [SDDA]")
    parser.add_argument("--adv_d_sdda_start_ep", type=int, default=None,
                        help="Starting epoch for adv_d_sdda loss. [SDDA]")
    parser.add_argument("--adv_d_sdda_end_ep", type=int, default=None,
                        help="Ending epoch for adv_d_sdda loss. [SDDA]")

    parser.add_argument("--adv_g_sdda", type=str2bool, default=None,
                        help="Adversarial generative loss. [SDDA]")
    parser.add_argument("--adv_g_sdda_lambda", type=float, default=None,
                        help="Lambda for adv_g_sdda loss. [SDDA]")
    parser.add_argument("--adv_g_sdda_start_ep", type=int, default=None,
                        help="Starting epoch for adv_g_sdda loss. [SDDA]")
    parser.add_argument("--adv_g_sdda_end_ep", type=int, default=None,
                        help="Ending epoch for adv_g_sdda loss. [SDDA]")

    parser.add_argument("--px_sdda", type=str2bool, default=None,
                        help="P(X) loss. [SDDA]")
    parser.add_argument("--px_sdda_lambda", type=float, default=None,
                        help="Lambda for px_sdda loss. [SDDA]")
    parser.add_argument("--px_sdda_start_ep", type=int, default=None,
                        help="Starting epoch for px_sdda loss. [SDDA]")
    parser.add_argument("--px_sdda_end_ep", type=int, default=None,
                        help="Ending epoch for px_sdda loss. [SDDA]")

    parser.add_argument("--ce_src_m_fake_sdda", type=str2bool, default=None,
                        help="Cross-entrpy of SRC model over generated samples "
                             "loss. [SDDA]")
    parser.add_argument("--ce_src_m_fake_sdda_lambda", type=float, default=None,
                        help="Lambda for ce_src_m_fake_sdda loss. [SDDA]")
    parser.add_argument("--ce_src_m_fake_sdda_start_ep", type=int, default=None,
                        help="Starting epoch for ce_src_m_fake_sdda loss. [SDDA]")
    parser.add_argument("--ce_src_m_fake_sdda_end_ep", type=int, default=None,
                        help="Ending epoch for ce_src_m_fake_sdda loss. [SDDA]")

    parser.add_argument("--ce_trg_m_fake_sdda", type=str2bool, default=None,
                        help="Cross-entrpy of TRG model over generated samples "
                             "loss. [SDDA]")
    parser.add_argument("--ce_trg_m_fake_sdda_lambda", type=float, default=None,
                        help="Lambda for ce_trg_m_fake_sdda loss. [SDDA]")
    parser.add_argument("--ce_trg_m_fake_sdda_end_ep", type=int, default=None,
                        help="Ending epoch for ce_trg_m_fake_sdda loss. [SDDA]")

    parser.add_argument("--ce_dom_d_sdda", type=str2bool, default=None,
                        help="Cross-entrpy of TRG model over generated samples "
                             "loss. [SDDA]")
    parser.add_argument("--ce_dom_d_sdda_a", type=float, default=None,
                        help="a for the reversed gradient layer. "
                             "Loss: ce_dom_d_sdda. [SDDA]")
    parser.add_argument("--ce_dom_d_sdda_end_ep", type=int, default=None,
                        help="Ending epoch for ce_dom_d_sdda loss. [SDDA]")
    


    # 5- NRC
    parser.add_argument("--nrc_neighborhood", type=str2bool, default=None,
                        help="nrc_neighborhood  loss. [NRC]")
    parser.add_argument("--nrc_na", type=str2bool, default=None, 
                        help="nrc_neighborhood_na  loss. [NRC]")
    parser.add_argument("--nrc_na_lambda", type=float, default=None, 
                        help="nrc_neighborhood_na_lambda  loss. [NRC]")
    parser.add_argument("--nrc_ena", type=str2bool, default=None, 
                        help="nrc_neighborhood_ena  loss. [NRC]")
    parser.add_argument("--nrc_ena_lambda", type=float, default=None, 
                        help="nrc_neighborhood_ena_lambda  loss. [NRC]")
    parser.add_argument("--nrc_kl", type=str2bool, default=None, 
                        help="nrc_neighborhood_kl  loss. [NRC]")
    parser.add_argument("--nrc_kl_lambda", type=float, default=None, 
                        help="nrc_neighborhood_kl_lambda  loss. [NRC]")
    parser.add_argument("--nrc_k", type=int, default=None, 
                        help="nrc_k_neighbors. [NRC]")
    parser.add_argument("--nrc_kk", type=int, default=None, 
                        help="nrc_k_neighbors. [NRC]")
    parser.add_argument("--nrc_epsilon", type=float, default=None, 
                        help="nrc_neighborhood_epsilon  loss. [NRC]")
    
    # 6- CDCL

    parser.add_argument('--cdcl_pseudo_lb', type=str2bool, default=None,
                        help='contrastive loss over image class pseudo-labels.')
    parser.add_argument('--cdcl_tau', type=float, default=None,
                        help='tau for contrastive loss.')
    parser.add_argument('--cdcl_lambda', type=float, default=None,
                        help='Parameter lambda for contrastive loss.')
    
    input_parser = parser.parse_args()

    def warnit(name, vl_old, vl):
        """
        Warn that the variable with the name 'name' has changed its value
        from 'vl_old' to 'vl' through command line.
        :param name: str, name of the variable.
        :param vl_old: old value.
        :param vl: new value.
        :return:
        """
        if vl_old != vl:
            print("Changing {}: {}  -----> {}".format(name, vl_old, vl))
        else:
            print("{}: {}".format(name, vl_old))

    attributes = input_parser.__dict__.keys()

    for k in attributes:
        val_k = getattr(input_parser, k)
        if k in args.keys():
            if val_k is not None:
                warnit(k, args[k], val_k)
                args[k] = val_k
            else:
                warnit(k, args[k], args[k])

        elif k in args['model'].keys():  # try model
            if val_k is not None:
                warnit('model.{}'.format(k), args['model'][k], val_k)
                args['model'][k] = val_k
            else:
                warnit('model.{}'.format(k), args['model'][k],
                       args['model'][k])

        elif k in args['optimizer'].keys():  # try optimizer 0
            if val_k is not None:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k], val_k)
                args['optimizer'][k] = val_k
            else:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k],
                    args['optimizer'][k]
                )
        else:
            raise ValueError("Key {} was not found in args. ..."
                             "[NOT OK]".format(k))

    # add the current seed to the os env. vars. to be shared across this
    # process.
    # this seed is expected to be local for this process and all its
    # children.
    # running a parallel process will not have access to this copy not
    # modify it. Also, this variable will not appear in the system list
    # of variables. This is the expected behavior.
    # TODO: change this way of sharing the seed through os.environ. [future]
    # the doc mentions that the above depends on `putenv()` of the
    # platform.
    # https://docs.python.org/3.7/library/os.html#os.environ
    os.environ['MYSEED'] = str(args["MYSEED"])

    args['outd'], args['subpath'] = outfd(Dict2Obj(args), eval=eval)
    args['outd_backup'] = args['outd']
    if is_cc():
        _tag = '{}__{}'.format(
            basename(normpath(args['outd'])), '{}'.format(
                np.random.randint(low=0, high=10000000, size=1)[0]))
        args['outd'] = join(os.environ["SLURM_TMPDIR"], _tag)
        mkdir(args['outd'])

    cmdr = not constants.OVERRUN
    cmdr &= not eval
    if is_cc():
        cmdr &= os.path.isfile(join(args['outd_backup'], 'passed.txt'))
    else:
        cmdr &= os.path.isfile(join(args['outd'], 'passed.txt'))
    if cmdr:
        warnings.warn('EXP {} has already been done. EXITING.'.format(
            args['outd']))
        sys.exit(0)

    args['scoremap_paths'] = configure_scoremap_output_paths(Dict2Obj(args))

    if args['box_v2_metric']:
        args['multi_contour_eval'] = True
        args['multi_iou_eval'] = True
    else:
        args['multi_contour_eval'] = False
        args['multi_iou_eval'] = False

    if args['model']['freeze_cl']:
        if args['task'] == constants.NEGEV:
            cl_cp = args['negev_ptretrained_cl_cp']
            std_cl_args = deepcopy(args)
            std_cl_args['task'] = constants.STD_CL
            tag = get_tag(Dict2Obj(std_cl_args), checkpoint_type=cl_cp)

        else:
            cl_cp = args['eval_checkpoint_type']
            tag = get_tag(Dict2Obj(args), checkpoint_type=cl_cp)

        args['model']['folder_pre_trained_cl'] = join(
            root_dir, 'pretrained', tag)

        zz = args['model']['folder_pre_trained_cl']
        assert os.path.isdir(args['model']['folder_pre_trained_cl']), zz

    if args['sf_uda'] and (args['sf_uda_source_folder'] == ''):
        tmp_config = copy.deepcopy(args)
        assert args['task'] in [constants.STD_CL, constants.NEGEV], args['task']

        if args['task'] == constants.STD_CL:
            tmp_config['dataset'] = args['sf_uda_source_ds']
            tmp_config['fold'] = args['sf_uda_source_ds_fold']
            tmp_config['model']['encoder_name'] = args['sf_uda_source_encoder_name']
            tmp_config['method'] = args['sf_uda_source_wsol_method']
            tmp_config['spatial_pooling'] = args['sf_uda_source_wsol_spatial_pooling']

        elif args['task'] == constants.NEGEV:

            tmp_config['dataset'] = args['sf_uda_source_ds']
            tmp_config['fold'] = args['sf_uda_source_ds_fold']
            tmp_config['model']['encoder_name'] = args[
                'sf_uda_source_encoder_name']
            tmp_config['model']['arch'] = args['sf_uda_source_wsol_arch']
            tmp_config['method'] = args['sf_uda_source_wsol_method']
            tmp_config['spatial_pooling'] = args[
                'sf_uda_source_wsol_spatial_pooling']
            checkpoint_type = args['sf_uda_source_checkpoint_type']

        else:
            raise NotImplementedError(args['task'])

        checkpoint_type = args['sf_uda_source_checkpoint_type']


        tag = get_tag(Dict2Obj(tmp_config), checkpoint_type=checkpoint_type)
        path_fd = join(root_dir, constants.SOURCE_MODELS_FD, tag)
        assert os.path.isdir(path_fd), path_fd

        args['sf_uda_source_folder'] = path_fd


    elif args['sf_uda'] and (args['sf_uda_source_folder'] != ''):
        assert os.path.isdir(args['sf_uda_source_folder']), args[
            'sf_uda_source_folder']


    if args['task'] in [constants.F_CL, constants.NEGEV]:
        for split in constants.SPLITS:

            if args['task'] == constants.NEGEV:
                cl_cp = args['negev_ptretrained_cl_cp']
                std_cl_args = deepcopy(args)
                std_cl_args['task'] = constants.STD_CL
                tag = get_tag(Dict2Obj(std_cl_args), checkpoint_type=cl_cp)
            else:
                cl_cp = args['eval_checkpoint_type']
                tag = get_tag(Dict2Obj(args), checkpoint_type=cl_cp)

            tag += '_cams_{}'.format(split)

            if is_cc():
                baseurl_sc = "{}/datasets/wsol-done-right".format(
                    os.environ["SCRATCH"])
                scratch_path = join(baseurl_sc, '{}.tar.gz'.format(tag))

                if os.path.isfile(scratch_path):
                    slurm_dir = config.get_root_wsol_dataset()
                    cmds = [
                        'cp {} {} '.format(scratch_path, slurm_dir),
                        'cd {} '.format(slurm_dir),
                        'tar -xf {}'.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                    assert os.path.isdir(join(slurm_dir, tag))
                    args['std_cams_folder'][split] = join(slurm_dir, tag)

            else:
                path_cams = join(root_dir, constants.DATA_CAMS, tag)
                cndx = not os.path.isdir(path_cams)
                cndx &= os.path.isfile('{}.tar.gz'.format(path_cams))
                if cndx:
                    cmds_untar = [
                        'cd {} '.format(join(root_dir, constants.DATA_CAMS)),
                        'tar -xf {} '.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds_untar)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                cndx = os.path.isdir(path_cams)
                cndx &= os.path.isfile('{}.tar.gz'.format(path_cams))
                if cndx:
                    args['std_cams_folder'][split] = path_cams

    # cuda   ------------------------------------------------------------------
    ngpus_per_node = torch.cuda.device_count()
    torch.cuda.set_device(0)
    args['c_cudaid'] = 0
    # --------------------------------------------------------------------------

    reproducibility.set_to_deterministic(seed=int(args["MYSEED"]), verbose=True)

    args_dict = deepcopy(args)
    args = Dict2Obj(args)

    # sanity check ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    spatial_dropout = args.model['spatial_dropout']
    assert isinstance(spatial_dropout, float), spatial_dropout
    assert 0. <= spatial_dropout <= 1., spatial_dropout

    assert args.batch_size > 0, args.batch_size
    assert args.eval_batch_size > 0, args.eval_batch_size
    # if args.nrc:
    #     assert not args.amp

    if args.sf_uda:
        assert args.task in [constants.STD_CL, constants.NEGEV], args.task

        l_sf_uda_techs = [args.shot, args.faust, args.adadsa, args.sdda, args.nrc, args.sfde, args.cdcl]

        assert any(l_sf_uda_techs)
        assert sum(l_sf_uda_techs) == 1, 'Only one SFUDA must be active.'

        if args.shot:
            assert any([args.ce_pseudo_lb,
                        args.ent_pseudo_lb,
                        args.div_pseudo_lb
                        ])

        elif args.faust:
            assert any([args.views_ft_consist,
                        args.ce_views_soft_pl,
                        args.mc_var_prob,
                        args.min_prob_entropy
                        ])

            if args.mc_var_prob:
                assert args.model['spatial_dropout'] > 0, args.model[
                    'spatial_dropout']
                assert args.mc_var_prob_n_dout > 1, args.mc_var_prob_n_dout

            if args.views_ft_consist or args.ce_views_soft_pl:
                assert args.faust_n_views >= 1, args.faust_n_views

            if args.ce_views_soft_pl:
                assert args.ce_views_soft_pl_t > 0, args.ce_views_soft_pl_t
        elif args.cdcl:
            pass

        elif args.adadsa:
            assert any([args.ce_pseudo_lb,
                        args.ent_pseudo_lb
                        ])

        elif args.sdda:
            assert args.adv_d_sdda
            assert args.adv_g_sdda

            assert args.adaptation_start_epoch >= 0, args.adaptation_start_epoch
            assert args.sdda_gan_h > 0, args.sdda_gan_h
            assert args.sdda_gan_w > 0, args.sdda_gan_w
            assert args.sdda_gan_latent_dim > 0, args.sdda_gan_latent_dim

            if args.ce_dom_d_sdda:
                assert args.ce_dom_d_sdda_a > 0, args.ce_dom_d_sdda_a

        elif args.nrc:
            assert args.nrc_neighborhood
        
        elif args.sfde:
            pass

        else:
            raise NotImplementedError
        
        l_sf_uda_techs = [tech for tech in [args.shot, ] if tech]
    
        

    if args.task == constants.NEGEV:
        assert args.negev_ptretrained_cl_cp in [constants.BEST_LOC,
                                                constants.BEST_CL]

    assert args.runmode in [constants.RMODE_FINAL, constants.RMODE_SEARCH]

    if args.method == constants.METHOD_CUTMIX:
        assert 0. <= args.cutmix_prob <= 1.
        assert args.cutmix_beta > 0

    if args.method == constants.METHOD_HAS:
        assert args.has_grid_size > 0
        assert isinstance(args.has_grid_size, int)
        assert 0. <= args.has_drop_rate < 1.

    if args.dataset == constants.BREAKHIS:
        assert args.magnification in constants.MAGNIFICATIONSBHIS

    assert args.fold in list(range(5))
    if args.task == constants.SEG:
        assert args.dataset in [constants.GLAS, constants.CAMELYON512]

    assert args.spatial_pooling == constants.METHOD_2_POOLINGHEAD[args.method]

    assert args.model['encoder_name'] in constants.BACKBONES
    
    assert not args.multi_label_flag
    assert args.seg_mode == constants.BINARY_MODE

    if isinstance(args.resize_size, int):
        if isinstance(args.crop_size, int):
            assert args.resize_size >= args.crop_size

    # todo
    assert args.model['scale_in'] > 0.
    assert isinstance(args.model['scale_in'], float)

    if args.task == constants.STD_CL:
        assert not args.model['freeze_cl']
        assert args.model['folder_pre_trained_cl'] in [None, '', 'None']

    used_constraints_f_cl = [args.sl_fc,
                             args.crf_fc,
                             args.entropy_fc,
                             args.max_sizepos_fc]
    used_constraints_negev = [args.sl_ng,
                              args.crf_ng,
                              args.jcrf_ng,
                              args.max_sizepos_ng,
                              args.neg_samples_ng]

    if args.task == constants.STD_CL:
        assert not any(used_constraints_f_cl)
        assert not any(used_constraints_negev)

    assert args.resize_size == constants.RESIZE_SIZE
    assert args.crop_size == constants.CROP_SIZE

    if args.task == constants.F_CL:
        assert any(used_constraints_f_cl)
        assert args.model['arch'] == constants.UNETFCAM

        assert args.eval_checkpoint_type == constants.BEST_LOC

    if args.task == constants.NEGEV:
        assert any(used_constraints_negev)
        assert args.model['arch'] == constants.UNETNEGEV

        assert args.eval_checkpoint_type == constants.BEST_LOC

        if args.neg_samples_ng:
            assert constants.DS_HAS_NEG_SAM[args.dataset]


    if args.task == constants.SEG:
        assert args.dataset in [constants.GLAS, constants.CAMELYON512]
        assert args.model['arch'] in [constants.UNET]
        assert args.eval_checkpoint_type == constants.BEST_LOC
        assert args.method == constants.METHOD_SEG
        assert args.spatial_pooling == constants.NONEPOOL

    assert args.model['arch'] in constants.ARCHS

    assert not args.im_rec

    return args, args_dict


def configure_scoremap_output_paths(args):
    scoremaps_root = join(args.outd, 'scoremaps')
    scoremaps = mch()
    for split in (constants.TRAINSET, constants.VALIDSET, constants.TESTSET):
        scoremaps[split] = join(scoremaps_root, split)
        if not os.path.isdir(scoremaps[split]):
            os.makedirs(scoremaps[split])
    return scoremaps


def outfd(args, eval=False):

    tag = [('id', args.exp_id),
           ('tsk', args.task),
           ('ds', args.dataset),
           ('fold', args.fold),
           ('mag', args.magnification if args.dataset == constants.BREAKHIS
           else 'None'),
           ('runmode', args.runmode),
           ('mth', args.method),
           ('spooling', args.spatial_pooling),
           # ('sd', args.MYSEED),
           ('arch', args.model['arch']),
           ('ecd', args.model['encoder_name']),
           # too long name. causes error for tar.
           # ('epx', args.max_epochs),
           # ('bsz', args.batch_size),
           # ('lr', '{:.4f}'.format(args.optimizer['opt__lr'])),
           # ('box_v2_metric', args.box_v2_metric),
           # ('amp', args.amp),
           # ('amp_eval', args.amp_eval)
           ]

    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    if args.task == constants.F_CL:
        # todo: add hyper-params.
        tag2 = []

        if args.sl_fc:
            tag2.append(("sl_fc", 'yes'))

        if args.crf_fc:
            tag2.append(("crf_fc", 'yes'))

        if args.entropy_fc:
            tag2.append(("entropy_fc", 'yes'))

        if args.max_sizepos_fc:
            tag2.append(("max_sizepos_fc", 'yes'))

        if tag2:
            tag2 = [(el[0], str(el[1])) for el in tag2]
            tag2 = '-'.join(['_'.join(el) for el in tag2])
            tag = "{}-{}".format(tag, tag2)

    if args.runmode in [constants.RMODE_SEARCH, constants.RMODE_FINAL]:
        parent_lv = constants.FOLDER_EXP[args.runmode]
    else:
        raise ValueError

    if args.debug_subfolder not in ['', None, 'None']:
        parent_lv = join(parent_lv, args.debug_subfolder)

    subfd = join(args.dataset, args.model['encoder_name'], args.task,
                 args.method)
    _root_dir = root_dir
    if is_cc():
        _root_dir = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER)

    subpath = join(parent_lv,
                   subfd,
                   tag)
    if not eval:
        OUTD = join(_root_dir,
                    subpath
                    )
    else:
        OUTD = join(_root_dir, args.fd_exp)

    OUTD = expanduser(OUTD)
    os.makedirs(OUTD, exist_ok=True)

    return OUTD, subpath


def wrap_sys_argv_cmd(cmd: str, pre):
    splits = cmd.split(' ')
    el = splits[1:]
    pairs = ['{} {}'.format(i, j) for i, j in zip(el[::2], el[1::2])]
    pro = splits[0]
    sep = ' \\\n' + (len(pre) + len(pro) + 2) * ' '
    out = sep.join(pairs)
    return "{} {} {}".format(pre, pro, out)


def copy_code(dest,
              compress=False,
              verbose=False
              ):
    exts = tuple(["py", "sh", "yaml"])
    flds_files = ['.']

    for fld in flds_files:
        files = glob.iglob(os.path.join(root_dir, fld, "*"))
        subfd = join(dest, fld) if fld != "." else dest
        if not os.path.exists(subfd):
            os.makedirs(subfd, exist_ok=True)

        for file in files:
            if file.endswith(exts):
                if os.path.isfile(file):
                    shutil.copy(file, subfd)
    # cp dlib
    dirs = ["dlib", "cmds"]
    for dirx in dirs:
        cmds = [
            "cd {} && ".format(root_dir),
            "cp -r {} {} ".format(dirx, dest)
        ]
        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)

    if compress:
        head = dest.split(os.sep)[-1]
        if head == '':  # dest ends with '/'
            head = dest.split(os.sep)[-2]
        cmds = [
            "cd {} && ".format(dest),
            "cd .. && ",
            "tar -cf {}.tar.gz {}  && ".format(head, head),
            "rm -rf {}".format(head)
               ]

        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)


def amp_log(args: object):
    _amp = False
    if args.amp:
        DLLogger.log(fmsg('AMP: activated'))
        _amp = True

    if args.amp_eval:
        DLLogger.log(fmsg('AMP_EVAL: activated'))
        _amp = True

    if _amp:
        tag = get_tag_device(args=args)
        if 'P100' in get_tag_device(args=args):
            DLLogger.log(fmsg('AMP [train: {}, eval: {}] is ON but your GPU {} '
                              'does not seem to have tensor cores. Your code '
                              'may experience slowness. It is better to '
                              'deactivate AMP.'.format(args.amp,
                                                       args.amp_eval, tag)))


def parse_input(eval=False):
    """
    Parse the input.
    and
    initialize some modules for reproducibility.
    """
    parser = argparse.ArgumentParser()

    if not eval:
        parser.add_argument("--dataset", type=str,
                            help="Dataset name: {}.".format(constants.datasets))
        parser.add_argument("--fold", type=int, default=None,
                            help="Fold of dataset.")
        parser.add_argument("--magnification", type=str, default=None,
                            help="Magnififcation of BreakHis dataset.")
        input_args, _ = parser.parse_known_args()
        args: dict = config.get_config(ds=input_args.dataset,
                                       fold=input_args.fold,
                                       magnification=input_args.magnification)
        args, args_dict = get_args(args)

        log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 join(args.outd, "log.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 join(args.outd, "log.txt")),
        ]

        if args.verbose:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
        DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())


        DLLogger.log(fmsg("Start time: {}".format(args.t0)))

        amp_log(args=args)

        if args.task == constants.F_CL:
            __split = constants.TRAINSET
            if os.path.isdir(args.std_cams_folder[__split]):
                msg = 'Will be using PRE-computed cams for split {} from ' \
                      '{}'.format(__split, args.std_cams_folder[__split])
                warnings.warn(msg)
                DLLogger.log(msg)
            else:
                msg = 'Will RE-computed cams for split {}.'.format(__split)
                warnings.warn(msg)
                DLLogger.log(msg)

        outd = args.outd

        if not os.path.exists(join(outd, "code/")):
            os.makedirs(join(outd, "code/"))

        with open(join(outd, "code/config.yml"), 'w') as fyaml:
            yaml.dump(args_dict, fyaml)

        with open(join(outd, "config.yml"), 'w') as fyaml:
            yaml.dump(args_dict, fyaml)

        str_cmd = wrap_sys_argv_cmd(" ".join(sys.argv), "time python")
        with open(join(outd, "code/cmd.sh"), 'w') as frun:
            frun.write("#!/usr/bin/env bash \n")
            frun.write(str_cmd)

        copy_code(join(outd, "code/"), compress=True, verbose=False)
    else:

        raise NotImplementedError

        parser.add_argument("--fd_exp", type=str,
                            help="relative path to the exp folder.")
        input_args, _ = parser.parse_known_args()
        _root_dir = root_dir
        if is_cc():
            _root_dir = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER)

        fd = join(_root_dir, input_args.fd_exp)

        yaml_file = join(fd, 'config.yaml')
        with open(yaml_file, 'r') as fy:
            args = yaml.safe_load(fy)

        args, args_dict = get_args(args, eval)

    DLLogger.flush()
    return args, args_dict
