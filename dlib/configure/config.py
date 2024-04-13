import os
import sys
from os.path import join, dirname, abspath
import datetime as dt

import munch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants

__all__ = ['get_config']


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_data_paths(args, dsname=None):
    if dsname is None:
        dsname = args['dataset']

    if dsname in [constants.CUB, constants.ILSVRC, constants.OpenImages]:
        train = val = test = join(args['data_root'], dsname)
        data_paths = mch(train=train, val=val, test=test)
    elif dsname in [constants.GLAS, constants.CAMELYON512,
                    constants.BREAKHIS, constants.ICIAR]:

        _splits = [constants.TRAINSET, constants.PXVALIDSET,
                   constants.CLVALIDSET, constants.TESTSET]
        _dict = {
            split: join(args['data_root'], dsname) for split in _splits
        }
        data_paths = munch.Munch(_dict)
    else:
        raise NotImplementedError

    return data_paths


def configure_std_cams_folder(dsname):

    if dsname in [constants.CUB, constants.ILSVRC, constants.OpenImages]:
        folders = mch(train='', val='', test='')
    elif dsname in [constants.GLAS, constants.CAMELYON512,
                    constants.BREAKHIS, constants.ICIAR]:

        _splits = [constants.TRAINSET, constants.PXVALIDSET,
                   constants.CLVALIDSET, constants.TESTSET]
        _dict = {
            split: '' for split in _splits
        }
        folders = munch.Munch(_dict)
    else:
        raise NotImplementedError

    return folders


def get_root_wsol_dataset():
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/wsol-done-right".format(os.environ["DATASETSH"])
        elif os.environ['HOST_XXX'] == 'gsys':
            baseurl = "{}/wsol-done-right".format(os.environ["DATASETSH"])
        elif os.environ['HOST_XXX'] == 'ESON':
            baseurl = "{}/datasets".format(os.environ["DATASETSH"])

    elif "CC_CLUSTER" in os.environ.keys():
        if "SLURM_TMPDIR" in os.environ.keys():
            # if we are running within a job use the node disc:  $SLURM_TMPDIR
            baseurl = "{}/datasets/wsol-done-right".format(
                os.environ["SLURM_TMPDIR"])
        else:
            # if we are not running within a job, use the scratch.
            # this cate my happen if someone calls this function outside a job.
            baseurl = "{}/datasets/wsol-done-right".format(os.environ["SCRATCH"])

    msg_unknown_host = "Sorry, it seems we are enable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "We recommend you to add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def get_config(ds: str, fold: int, magnification: str) -> dict:
    assert ds in constants.datasets, ds
    assert ds in constants.SUPPORTED_DS

    assert fold in list(range(constants.FOLDS_NBR))
    if ds == constants.BREAKHIS:
        assert magnification in constants.MAGNIFICATIONSBHIS

    args = {
        # ======================================================================
        #                               GENERAL
        # ======================================================================
        "MYSEED": 0,  # Seed for reproducibility. int >= 0.
        "cudaid": '0',  # str. cudaid. form: '0,1,2,3' for cuda devices.
        "debug_subfolder": '',  # subfolder used for debug. if '', we do not
        # consider it.
        "dataset": ds,  # name of the dataset.
        'fold': fold,  # int. numberID of the fold.
        'magnification': magnification,  # magnification factor.
        "num_classes": constants.NUMBER_CLASSES[ds],  # Total number of classes.
        "crop_size": constants.CROP_SIZE,  # int. size of cropped patch.
        "resize_size": constants.RESIZE_SIZE,  # int. size to which the image
        # is resized before cropping.
        "batch_size": 8,  # the batch size for training.
        "eval_batch_size": 32,  # evaluation batch size.
        "num_workers": 4,  # number of workers for dataloader of the trainset.
        "exp_id": "123456789",  # exp id. random number unique for the exp.
        "verbose": True,  # if true, we print messages in stdout.
        'fd_exp': None,  # relative path to folder where the exp.
        'abs_fd_exp': None,  # absolute path to folder where the exp.
        'best_loc_epoch': 0,  # int. best localization epoch.
        'best_cl_epoch': 0,  # int. best classification epoch.
        'img_range': constants.RANGE_TANH,  # range of the image values after
        # normalization either in [0, 1] or [-1, 1]. see constants.
        't0': dt.datetime.now(),  # approximate time of starting the code.
        'tend': None,  # time when this code ends.
        'running_time': None,  # the time needed to run the entire code.
        'localization_avail': constants.LOCALIZATION_AVAIL[ds],  # whether there
        # is a localization information or not.
        'best_valid_tau_loc': None,  # float. best pixel localization tau
        # estimated over validset for model selected using best localization.
        # will be determined automatically and used over test set.
        'best_valid_tau_cl': None,  # float. best pixel localization tau
        # estimated over validset for model selected using best classification.
        # will be determined automatically and used over test set.
        'runmode': constants.RMODE_SEARCH,  # search mode: for hyper-parameter
        # search or final: use already found hyper-parameters. useful for
        # filtering experiments. does not have an impact on the experiments.
        # ======================================================================
        #                      WSOL DONE RIGHT
        # ======================================================================
        "data_root": get_root_wsol_dataset(),  # absolute path to data parent
        # folder.
        "metadata_root": constants.RELATIVE_META_ROOT,  # path to metadata.
        # contains splits.
        "mask_root": get_root_wsol_dataset(),  # path to masks.
        "proxy_training_set": False,  # efficient hyper-parameter search with
        # a proxy training set. true/false.
        "std_cams_folder": configure_std_cams_folder(ds),  # folders where
        # cams of std_cl are stored to be used for f_cl training. typically,
        # we store only training. this is an option since f_cl/negev can still
        # compute the std_cams. but, storing them making their access fast
        # to avoid re-computing them every time during training. the exact
        # location will be determined during parsing the input. this is
        # optional. if we do not find this folder, we recompute the cams.
        "num_val_sample_per_class": 0,  # number of full_supervision
        # validation sample per class. 0 means: use all available samples.
        'cam_curve_interval': .001,  # CAM curve interval.
        'multi_contour_eval': True,  # Bounding boxes are extracted from all
        # contours in the thresholded score map. You can use this feature by
        # setting multi_contour_eval to True (default). Otherwise,
        # bounding boxes are extracted from the largest connected
        # component of the score map.
        'multi_iou_eval': True,
        'iou_threshold_list': [30, 50, 70],
        'box_v2_metric': False,
        'eval_checkpoint_type': constants.BEST_LOC,  # just for
        # stand-alone inference. during training+inference, we evaluate both.
        # Necessary s well for the task F_CL during training to select the
        # init-model-weights-classifier.
        # ======================================================================
        #                      VISUALISATION OF REGIONS OF INTEREST
        # ======================================================================
        "alpha_visu": 100,  # transparency alpha for cams visualization. low is
        # opaque (matplotlib).
        "height_tag": 60,  # the height of the margin where the tag is written.
        # ======================================================================
        #                             OPTIMIZER (n0)
        #                            TRAIN THE MODEL
        # ======================================================================
        "optimizer": {  # the optimizer
            # ==================== SGD =======================
            "opt__name_optimizer": constants.SGD,  # str name. 'sgd', 'adam'
            "opt__lr": 0.001,  # Initial learning rate.
            "opt__momentum": 0.9,  # Momentum.
            "opt__dampening": 0.,  # dampening.
            "opt__weight_decay": 1e-4,  # The weight decay (L2) over the
            # parameters.
            "opt__nesterov": True,  # If True, Nesterov algorithm is used.
            # ==================== ADAM =========================
            "opt__beta1": 0.9,  # beta1.
            "opt__beta2": 0.999,  # beta2
            "opt__eps_adam": 1e-08,  # eps. for numerical stability.
            "opt__amsgrad": False,  # Use amsgrad variant or not.
            # ========== LR scheduler: how to adjust the learning rate. ========
            "opt__lr_scheduler": True,  # if true, we use a learning rate
            # scheduler.
            # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
            "opt__name_lr_scheduler": constants.MYSTEP,  # str name.
            "opt__step_size": 40,  # Frequency of which to adjust the lr.
            "opt__gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            "opt__last_epoch": -1,  # the index of the last epoch where to stop
            # adjusting the LR.
            "opt__min_lr": 1e-6,  # minimum allowed value for lr.
            "opt__t_max": 100,  # T_max for cosine schedule.
            "opt__lr_classifier_ratio": 10.,  # Multiplicative factor on the
            # classifier layer (head) learning rate.
        },
        # ======================================================================
        #                              MODEL
        # ======================================================================
        "model": {
            "arch": constants.UNETFCAM,  # name of the model.
            # see: constants.ARCHS.
            "encoder_name": constants.RESNET50,  # backbone for task of SEG.
            "encoder_weights": constants.IMAGENET,
            # pretrained weights or 'None'.
            "in_channels": 3,  # number of input channel.
            "path_pre_trained": None,
            # None, `None` or a valid str-path. if str,
            # it is the absolute/relative path to the pretrained model. This can
            # be useful to resume training or to force using a filepath to some
            # pretrained weights.
            "strict": True,  # bool. Must be always be True. if True,
            # the pretrained model has to have the exact architecture as this
            # current model. if not, an error will be raise. if False, we do the
            # best. no error will be raised in case of mismatch.
            "support_background": True,  # useful for classification tasks only:
            # std_cl, f_cl only. if true, an additional cam is used for the
            # background. this does not change the number of global
            # classification logits. irrelevant for segmentation task.
            "scale_in": 1.,  # float > 0.  how much to scale
            # the input image to not overflow the memory. This scaling is done
            # inside the model on the same device as the model.
            "freeze_cl": False,  # applied only for task F_CL/NEGEV. if true,
            # the classifier (encoder + head) is frozen.
            "folder_pre_trained_cl": None,
            # NAME of folder containing weights of
            # classifier. it must be in in 'pretrained' folder.
            # used in combination with `freeze_cl`. the folder contains
            # encoder.pt, head.pt weights of the encoder and head. the base name
            # of the folder is a tag used to make sure of compatibility between
            # the source (source of weights) and target model (to be frozen).
            # You do not need to set this parameters if `freeze_cl` is true.
            # we set it automatically when parsing the parameters.
            "spatial_dropout": 0.0,  # perform 2d dropout at the last feature
            # layer of the classifier encoder. Allows MC dropout.
        },
        # ======================================================================
        #                    CLASSIFICATION SPATIAL POOLING
        # ======================================================================
        "method": constants.METHOD_WILDCAT,
        "spatial_pooling": constants.WILDCATHEAD,
        # ======================================================================
        #                        SPATIAL POOLING:
        #                            WILDCAT
        # ======================================================================
        "wc_modalities": 5,
        "wc_kmax": 0.5,
        "wc_kmin": 0.1,
        "wc_alpha": 0.6,
        "wc_dropout": 0.0,
        # ================== LSE pooling
        "lse_r": 10.,  # r for logsumexp pooling.
        # ======= METHODS ======================================================
        # ADL
        "adl_drop_rate": 0.4,  # float. [0, 1]
        "adl_drop_threshold": 0.1,  # float. [0, 1]. percentage of maximum
        # intensity: val > adl_drop_threshold * maximum_intensity.
        "adl_large_feature_map": True,  # use large feature maps.
        # ACOL
        "acol_drop_threshold": 0.1,  # float. [0, 1]
        "acol_large_feature_map": True,  # use large feature maps.
        # SPG
        "spg_threshold_1h": 0.7,
        "spg_threshold_1l": 0.01,
        "spg_threshold_2h": 0.5,
        "spg_threshold_2l": 0.05,
        "spg_threshold_3h": 0.7,
        "spg_threshold_3l": 0.1,
        "spg_large_feature_map": True,  # use large feature maps.
        # HAS
        "has_grid_size": 5,  # int. size of patch.
        "has_drop_rate": 0.5,  # float [0, 1[ percentage of dropped patches for
        # CUTMIX.
        "cutmix_beta": 1.,  # float. hyper-parameter of beta distribution.
        "cutmix_prob": 1.,  # probability to perform cutmix.
        # MIL
        'mil_mid_channels': 128,  # dim mid-channels.
        'mil_gated': False,  # gated or not attention.
        # MAXMIN
        'maxmin_w': 8.,  # scaling factor.
        'maxmin_sigma': .15,
        'maxmin_sigma_delta': .001,
        'maxmin_sigma_max': .2,
        'minmax_lambda_size': 1e-3,
        'minmax_lambda_neg': 1e-7,
        # PRM
        'prm_ks': 3,  # kernel size for peak stimulation.
        'prm_st': 1,  # kernel stride.
        # NEGEV ----------------------------------------------------------------
        "negev_ptretrained_cl_cp": constants.BEST_LOC,  # check point for
        # pretrained classifier.
        # NEGEV:  self-learning
        "sl_ng": False,  # use self-learning over negev cams.
        "sl_ng_seeder": constants.SEED_PROB,  # type of seeder.
        "sl_ng_lambda": 1.,  # lambda for self-learning over negev.
        "sl_ng_start_ep": 0,  # epoch when to start sl loss.
        "sl_ng_end_ep": -1,  # epoch when to stop using sl loss. -1: never stop.
        "sl_ng_min": 10,  # int. number of pixels to be considered
        # background (after sorting all pixels). all seeder methods.
        "sl_ng_max": 10,  # number of pixels to be considered
        # foreground (after sorting all pixels). all seeder methods.
        "sl_ng_ksz": 1,  # int, kernel size for dilation around the pixel.
        # must be odd number. all seeder methods.
        'sl_ng_min_p': .2,  # percentage of pixels to be used for background
        # sampling. percentage from entire image size. threshold,
        # and prob_n_area.
        # all below is only for threshold seeder method.
        "sl_ng_min_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_min from.
        "sl_ng_max_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_max.
        "sl_ng_block": 1,  # size of the block. instead of selecting from pixel,
        # we allow initial selection from grid created from blocks of size
        # sl_blockxsl_block. them, for each selected block, we select a random
        # pixel. this helps selecting from fare ways regions. if you don't want
        # to use blocks, set this to 1 where the selection is done directly over
        # pixels without passing through blocks.
        'sl_ng_fg_erode_k': 11,  # int. size of erosion kernel to clean
        # foreground.
        'sl_ng_fg_erode_iter': 1,  # int. number of erosions for foreground.
        # NEGEV:  CRF
        "crf_ng": False,  # use or not crf over negev cams.  (penalty)
        "crf_ng_lambda": 2.e-9,  # crf lambda
        "crf_ng_sigma_rgb": 15.,
        "crf_ng_sigma_xy": 100.,
        "crf_ng_scale": 1.,  # scale factor for input, segm.
        "crf_ng_start_ep": 0,  # epoch when to start crf loss.
        "crf_ng_end_ep": -1,  # epoch when to stop using crf loss. -1: never
        # stop.
        # NEGEV:  joint CRF
        "jcrf_ng": False,  # use or not joint crf over negev cams over
        # multiple images. apply only color penalty.
        "jcrf_ng_lambda": 2.e-9,  # crf lambda
        "jcrf_ng_sigma_rgb": 15.,
        "jcrf_ng_scale": 1.,  # scale factor for input, segm.
        "jcrf_ng_start_ep": 0,  # epoch when to start crf loss.
        "jcrf_ng_end_ep": -1,  # epoch when to stop using crf loss. -1: never
        # stop.
        "jcrf_ng_pair_mode": constants.PAIR_SAME_C,  # samples will be paired
        # randomly with other samples in the same class or from different
        # classes.
        "jcrf_ng_n": 1,  # int. number of samples to pair with each sample.
        # NEGEV:  size
        "max_sizepos_ng": False,  # use absolute size (unsupervised) over all
        # negev cams. (elb)
        "max_sizepos_ng_lambda": 1.,
        "max_sizepos_ng_start_ep": 0,  # epoch when to start maxsz loss.
        "max_sizepos_ng_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.
        # NEGEV:  negative samples
        "neg_samples_ng": False,  # use or not negative samples. allowed only
        # for camelyon16 dataset. will throw an error otherwise.
        "neg_samples_ng_lambda": 1.,
        "neg_samples_ng_start_ep": 0,  # epoch when to start loss.
        "neg_samples_ng_end_ep": -1,  # epoch when to stop loss. -1: never stop.
        # NEGEV ----------------------------------------------------------------

        ####for sat
        "sat_drop_rate": 0.0,  # float [0, 1] stochastic depth decay rule same value in SAT code
        "sat_drop_path_rate": 0.1,  # float [0, 1] drop path for stochastic depth same value in SAT code
        "sat_area_th": 0.35,  # float [0, 1] area threshold for SAT same value in SAT code
        # ======================================================================
        #                          Segmentation mode
        # ======================================================================
        "seg_mode": constants.BINARY_MODE,
        # SEGMENTATION mode: bin only always.
        "task": constants.STD_CL,  # task: standard classification (able to
        # wsol or simply classification), full classification (FCAM),
        # segmentation.
        "multi_label_flag": False,
        # whether the dataset has multi-label or not.
        # ======================================================================
        #                          ELB
        # ==========================================================================
        "elb_init_t": 1.,  # used for ELB.
        "elb_max_t": 10.,  # used for ELB.
        "elb_mulcoef": 1.01,  # used for ELB.
        # ======================================================================
        #                            CONSTRAINTS:
        #                     'SuperResolution', sr
        #                     'ConRanFieldFcams', crf_fc
        #                     'EntropyFcams', entropy_fc
        #                     'PartUncerknowEntropyLowCams', partuncertentro_lc
        #                     'PartCertKnowLowCams', partcert_lc
        #                     'MinSizeNegativeLowCams', min_sizeneg_lc
        #                     'MaxSizePositiveLowCams', max_sizepos_lc
        #                     'MaxSizePositiveFcams' max_sizepos_fc
        # ======================================================================
        "max_epochs": 150,  # number of training epochs.
        # -----------------------  FCAM
        "sl_fc": False,  # use self-learning over fcams.
        "sl_fc_lambda": 1.,  # lambda for self-learning over fcams
        "sl_start_ep": 0,  # epoch when to start sl loss.
        "sl_end_ep": -1,  # epoch when to stop using sl loss. -1: never stop.
        "sl_min": 10,  # int. number of pixels to be considered
        # background (after sorting all pixels).
        "sl_max": 10,  # number of pixels to be considered
        # foreground (after sorting all pixels).
        "sl_min_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_min from.
        "sl_max_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_max.
        "sl_block": 1,  # size of the block. instead of selecting from pixel,
        # we allow initial selection from grid created from blocks of size
        # sl_blockxsl_block. them, for each selected block, we select a random
        # pixel. this helps selecting from fare ways regions. if you don't want
        # to use blocks, set this to 1 where the selection is done directly over
        # pixels without passing through blocks.
        "sl_ksz": 1,  # int, kernel size for dilation around the pixel. must be
        # odd number.
        'sl_min_p': .2,  # percentage of pixels to be used for background
        # sampling. percentage from entire image size.
        'sl_fg_erode_k': 11,  # int. size of erosion kernel to clean foreground.
        'sl_fg_erode_iter': 1,  # int. number of erosions for foreground.
        # ----------------------- FCAM
        "crf_fc": False,  # use or not crf over fcams.  (penalty)
        "crf_lambda": 2.e-9,  # crf lambda
        "crf_sigma_rgb": 15.,
        "crf_sigma_xy": 100.,
        "crf_scale": 1.,  # scale factor for input, segm.
        "crf_start_ep": 0,  # epoch when to start crf loss.
        "crf_end_ep": -1,  # epoch when to stop using crf loss. -1: never stop.
        # ======================================================================
        # ======================================================================
        #                                EXTRA
        # ======================================================================
        # ======================================================================
        # ----------------------- FCAM
        "entropy_fc": False,  # use or not the entropy over fcams. (penalty)
        "entropy_fc_lambda": 1.,
        # -----------------------  FCAM
        "max_sizepos_fc": False,  # use absolute size (unsupervised) over all
        # fcams. (elb)
        "max_sizepos_fc_lambda": 1.,
        "max_sizepos_fc_start_ep": 0,  # epoch when to start maxsz loss.
        "max_sizepos_fc_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.
        # ----------------------------------------------------------------------
        # ----------------------- NOT USED
        # ------------------------------- GENERIC
        "im_rec": False,  # image reconstruction loss.
        "im_rec_lambda": 1.,
        "im_rec_elb": False,  # use or not elb for image reconstruction.
        # ----------------------------- NOT USED
        # ----------------------------------------------------------------------
        # ======================================================================
        # ======================================================================
        # ======================================================================

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ GENERIC
        'seg_ignore_idx': -255,  # ignore index for segmentation alignment.
        'amp': False,  # if true, use automatic mixed-precision for training
        'amp_eval': False,  # if true, amp is used for inference.
        # ======================================================================
        'c_cudaid': 0,  # int. current cuda id. auto-set.
        # ======================================================================
        #                             SOURCE
        # ======================================================================
        "ce_label_smoothing": 0.0,  # label smoothing. [0., 1.] for
        # cross-entropy loss over global image class. applied over source
        # model training.
        # ======================================================================
        #                   DOMAIN ADAPTATION METHODS
        # ======================================================================
        'sf_uda': False,  # if true, we are in a source free unsupervised
        # domain adaptation setup. this requires a pretrained source model.
        # otherwise, we are training a source model.
        # if true, one of the SFUDA must be on, otherwise an error will be
        # thrown.
        # ---- source method
        'sf_uda_source_ds': constants.CAMELYON512,  # source dataset.
        'sf_uda_source_ds_fold': 0,  # source dataset, fold (if necessary).
        'sf_uda_source_encoder_name': constants.RESNET50,  # encoder name of
        # the source model.
        'sf_uda_source_checkpoint_type': constants.BEST_LOC,  # checkpoint of
        # the source model.
        'sf_uda_source_wsol_method': constants.METHOD_CAM,  # wsol method of
        # the source model.
        'sf_uda_source_wsol_arch': constants.STDCLASSIFIER,  # wsol arch of
        # the source model.
        'sf_uda_source_wsol_spatial_pooling': constants.WGAP,  # spatial
        # pooling of the source model.
        'sf_uda_source_folder': '',  # str. absolute path to folder where the
        # source files (encoder + head + config yaml file) reside. you can
        # force this. if not provided (i.e., ''), we will estimated it
        # automatically. they are expected to be in
        # constants.SOURCE_MODELS_FD in root code.
        # ---------------------------
        # target wsol method is specified in main, above. using method,
        # spatial_pooling, dataset, fold, encoder_name.

        # SFUDA methods ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ======================================================================
        #                               SHOT
        # ======================================================================
        # SHOT: "Do We Really Need to Access the Source Data? Source Hypothesis
        # Transfer for Unsupervised Domain Adaptation", ICML 2020,
        # https://arxiv.org/abs/2002.08546.
        'shot': False,  # SHOT method. ON/OFF.
        'shot_freq_epoch': 1,  # how often we re-estimate the target dataset
        # pseudo-labels.
        'shot_dist_type': constants.SHOT_COSINE,  # type of distance for
        # shot. see constants.SHOT_DISTS: euclidean, cosine (1. - cosine).
        # ======================================================================
        #                           END - SHOT
        # ======================================================================
        # ======================================================================
        #                           FAUST
        # ======================================================================
        # FAUST: "Feature Alignment by Uncertainty and Self-Training for
        # Source-Free Unsupervised Domain Adaptation", Neural networks, 2023.
        # https://arxiv.org/abs/2208.14888.
        'faust': False,  # FAUST method. ON/OFF. May require large memory.
        # Change batch size. Some of its losses enjoy large batch size.
        'faust_n_views': 0,  # number of views. used only when
        # losses are on: 'views_ft_consist', 'ce_views_soft_pl'.
        # IF REQUIRED: int (>= 1). Many views will require large memory
        # depending on the batch size.

        # ======================================================================
        #                           END - FAUST
        # ======================================================================
        # ======================================================================
        #                           CDCL
        # ======================================================================
        # CDCL: "Cross-domain Contrastive Learning for
        # SUnsupervised Domain Adaptation", IEEE TRANSACTIONS ON MULTIMEDIA, 2021.
        # https://arxiv.org/pdf/2106.05528.pdf.
        'cdcl': False,  # CDCL method. ON/OFF. May require large memory.
        'cdcl_threshold': 1.0,  # threshold for pseudo-labels.
        # ======================================================================
        #                           END - CDCL
        # ======================================================================

        # ======================================================================
        #                           AdaDSA
        # ======================================================================
        # Paper AdaDSA: "Unsupervised Domain Adaptation by Statistics Alignment
        # # for Deep Sleep Staging Networks", IEEE Transactions on Neural
        # Systems and Rehabilitation Engineering, 2022.
        # https://ieeexplore.ieee.org/document/9684410.
        # NOTE: the parameter of the target model are frozen. only a
        # single scalar (alpha) per batchnorm module is set to be a learnable
        # parameter. Therefore, this is fast in term of training. e.g. on
        # ResNet50, the total number of learnable parameters is around 53
        # which is the number of alpha(s).
        # Warning: since the parameters of model are not updated over the
        # target set, in particular classification head, this method can easily
        # lead to poor classification performance when dealing with large shift.
        'adadsa': False,  # AdaDSA method. ON/OFF.
        'adadsa_a': 10.,  # flot > 0. a to adapt lambda. recommended value: 10.
        'adadsa_eval_batch_size': 64,  # int. batch size used to forward
        # entire target set to estimate its BN stats. Use: -1 to use entire
        # target set at once instead of minibatches [better this way].
        # Estimation of target trainset BN stats is done only once.

        # ======================================================================
        #                           END - AdaDSA
        # ======================================================================

        # ======================================================================
        #                           SDDA
        # ======================================================================
        # Paper SDDA: "Domain Impression: A Source Data Free Domain Adaptation
        #   Method", WACV, 2021.
        # https://arxiv.org/abs/2102.09003.
        # It is composed of 2 parts:
        # 1. generation (cgan: generator +
        # discriminator models), and
        # 2. adaptation: target model + domain discriminator.
        'sdda': False,  # SDDA method. ON/OFF.
        'sdda_gan_type': constants.CGAN_ORIGINAL,  # type of the GAN used to
        # generate samples.
        'sdda_gan_latent_dim': 100,  # int, dim random noise.
        'sdda_gan_h': 64,  # int. dim generated image -> height.
        'sdda_gan_w': 64,  # int. dim generated image -> width.
        'adaptation_start_epoch': 25,  # int. when to start adaptation phase.
        # it affects 2 losses: `ce_trg_m_fake_sdda`, `ce_dom_d_sdda`.

        # Optimizer hyper-params for the adv. discriminator in the Generation
        # part.
        # The generator + target model + domain discriminator will be optimizer
        # with a different optimizer defined by the values in 'optimizer' above.
        'sdda_d__name_optimizer': constants.SGD,  # str name. 'sgd', 'adam'
        'sdda_d__lr': 0.001,  # Initial learning rate.
        'sdda_d__momentum': 0.9,  # Momentum.
        'sdda_d__dampening': 0.,  # dampening.
        'sdda_d__weight_decay': 1e-4,  # The weight decay (L2) over the
        # parameters.
        'sdda_d__nesterov': True,  # If True, Nesterov algorithm is used.
        # ==================== ADAM =========================
        'sdda_d__beta1': 0.9,  # beta1.
        'sdda_d__beta2': 0.999,  # beta2
        'sdda_d__eps_adam': 1e-08,  # eps. for numerical stability.
        'sdda_d__amsgrad': False,  # Use amsgrad variant or not.
        # ========== LR scheduler: how to adjust the learning rate. ========
        'sdda_d__lr_scheduler': True,  # if true, we use a learning rate
        # scheduler.
        # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
        'sdda_d__name_lr_scheduler': constants.MYSTEP,  # str name.
        'sdda_d__step_size': 40,  # Frequency of which to adjust the lr.
        'sdda_d__gamma': 0.1,  # the update coefficient: lr = gamma * lr.
        'sdda_d__last_epoch': -1,  # the index of the last epoch where to stop
        # adjusting the LR.
        'sdda_d__min_lr': 1e-6,  # minimum allowed value for lr.
        'sdda_d__t_max': 100,  # T_max for cosine schedule.

        # ======================================================================
        #                           END - SDDA
        # ======================================================================

        # ======================================================================
        #                               NRC
        # ======================================================================
        # NRC: "Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation", NeurIPS 2021,
        #  https://arxiv.org/abs/2110.04202.
        'nrc': False,  # SHOT method. ON/OFF.
        'r_nrc': 0.1, #hyperparameter for NRC
        # ======================================================================
        #                           END - NRC
        # ======================================================================


        #                       DISTRIBUTION ESTIMATION
        # ======================================================================
        # SFDE: "Source-Free Domain Adaptation via Distribution Estimation", CVPR 2022
        # https://arxiv.org/abs/2002.08546.
        'sfde': False,  # SFDE method. ON/OFF.
        'sfde_threshold': 1.0, # lambda of this term. >= 0.
        # ======================================================================
        #                     END - DISTRIBUTION ESTIMATION
        # ======================================================================
        # SFUDA losses
        # 1- CE over image global pseudo-labels.
        # WARNING: THE TRAINING OF SOME WSOL METHODS USING CE VIA IMAGE CLASS
        # PSEUDO-LABELS MAY REQUIRE DIFFERENT CE STRATEGY. E.G. SPG, ACOL,
        # CUTMIX. IF 'ce_pseudo_lb' IS ACTIVATED, WE WILL USE THE
        # CORRESPONDING CE LOSS OF THE WSOL METHOD. NOTHING CHANGES IN THE
        # WSOL, EXCEPT THAT WE FEED THE IMAGE CLASS PSEUDOLABELS INSTEAD OF
        # THE TRUE ONES. [SHOT, AdaDSA]
        'ce_pseudo_lb': False,  # cross-entropy over pseudo-labels.
        'ce_pseudo_lb_lambda': 0.3,  # lambda of this term. >= 0.
        'ce_pseudo_lb_start_ep': 0,  # epoch when to start this loss.
        'ce_pseudo_lb_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.
        'ce_pseudo_lb_smooth': 0.0,  # smoothing CE [0., 1.] for target.

        # 2- Entropy: minimize entropy over image global class prediction.
        # [SHOT, AdaDSA]
        'ent_pseudo_lb': False,  # Entropy over image class prediction.
        'ent_pseudo_lb_lambda': 0.3,  # lambda of this term. >= 0.
        'ent_pseudo_lb_start_ep': 0,  # epoch when to start this loss.
        'ent_pseudo_lb_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 3- Diversity loss over image global class prediction. [SHOT]
        'div_pseudo_lb': False,  # Diversity loss over image global class
        # prediction. Push expected prob vector over minibatch to follow
        # uniform dist. 'Discriminative Clustering by Regularized Information
        # Maximization', 2010.
        'div_pseudo_lb_lambda': 0.3,  # lambda of this term. >= 0.
        'div_pseudo_lb_start_ep': 0,  # epoch when to start this loss.
        'div_pseudo_lb_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 4- Views features consistency: [FAUST] align the features of each
        # augmented version of input image to be similar to the feature of a
        # the non-augmented feature of the same input image. Feature: dense
        # image representation. Referred to as: Aleatoric Uncertainty by
        # Augmentation (intra-space consistency) in FAUST. Number views is
        # set by: 'faust_n_views'. The alignment is done via cosine similarity.
        # Views are created via a series of standard data augmentation methods
        # over the input image: see wsol_loader.py
        'views_ft_consist': False,  # on/off this loss.
        'views_ft_consist_lambda': 0.3,  # lambda of this term. >= 0.
        'views_ft_consist_start_ep': 0,  # epoch when to start this loss.
        'views_ft_consist_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # -5 Views alignment with soft-class pseudo-labels [FAUST]: A soft
        # imaghe class pseudo-label is determined over minibtach for each
        # image. Views class probabilities are aligned to the soft-label
        # assigned to the non-augmented image. Number of views is set by
        # 'faust_n_views'.
        'ce_views_soft_pl': False,  # on/off.
        'ce_views_soft_pl_t': 1.,  # float > 0. temperature toi heatup
        # soft-labels. score/t.
        'ce_views_soft_pl_lambda': 0.3,  # lambda of this term. >= 0.
        'ce_views_soft_pl_start_ep': 0,  # epoch when to start this loss.
        'ce_views_soft_pl_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 6- Min epistemic uncertainty [FAUST]: use MC-dropout to estimate
        # the image class probability to minimize the l2-norm of its variance.
        # It is run over non-augmented input image, and it is expensive.
        # The model should have dropout. Currently, all models support 2d
        # dropout (except MAXMIN) via: model -> spatial_dropout which
        # MUST be > 0 if this is on, otherwise an error will be thrown.
        'mc_var_prob': False,  # on/off.
        'mc_var_prob_n_dout': 2,  # int >= 2. Number of times to do
        # mc-dropout. EXPENSIVE, REQUIRES MORE MEMORY depending on how many
        # times we do mc-dropout + batch size.
        'mc_var_prob_lambda': 0.3,  # lambda of this term. >= 0.
        'mc_var_prob_start_ep': 0,  # epoch when to start this loss.
        'mc_var_prob_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 7- Min probability entropy [FAUST]: Minimize the entropy of
        # image class probability over non-augmented image.
        'min_prob_entropy': False,  # on/off.
        'min_prob_entropy_lambda': 0.3,  # lambda of this term. >= 0.
        'min_prob_entropy_start_ep': 0,  # epoch when to start this loss.
        'min_prob_entropy_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 8- Adversarial loss for discriminator for SDDA method. [SDDA]
        'adv_d_sdda': False,  # MUST BE ON. Mandatory.
        'adv_d_sdda_lambda': 1.,  # lambda of this term. >= 0.
        'adv_d_sdda_start_ep': 0,  # epoch when to start this loss.
        'adv_d_sdda_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 9- Adversarial loss for the generator for SDDA method [SDDA]
        'adv_g_sdda': False,  # MUST be ON. Mandatory.
        'adv_g_sdda_lambda': 1.,  # lambda of this term. >= 0.
        'adv_g_sdda_start_ep': 0,  # epoch when to start this loss.
        'adv_g_sdda_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 10- Max likelihood p(x) over source model over generated samples. [
        # SDDA]
        'px_sdda': False,  # ON/OFF.
        'px_sdda_lambda': 0.1,  # lambda of this term. >= 0.
        'px_sdda_start_ep': 0,  # epoch when to start this loss.
        'px_sdda_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 11- Cross-entrpy of source model over generated samples. [SDDA]
        'ce_src_m_fake_sdda': False,  # ON/OFF.
        'ce_src_m_fake_sdda_lambda': 0.1,  # lambda of this term. >= 0.
        'ce_src_m_fake_sdda_start_ep': 0,  # epoch when to start this loss.
        'ce_src_m_fake_sdda_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # 12- Cross-entrpy over target classifier over generated images. [SDDA]
        'ce_trg_m_fake_sdda': False,  # ON/OFF.
        'ce_trg_m_fake_sdda_lambda': 0.1,  # lambda of this term. >= 0.
        'ce_trg_m_fake_sdda_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop. -> start epoch is set by 'adaptation_start_epoch'.

        # 13- Cross-entrpy over domain discriminator. [SDDA]
        'ce_dom_d_sdda': False,  # ON/OFF.
        'ce_dom_d_sdda_a': 10.,  # float > 0. gamma to adapt lambda in
        # reverse-layer in the paper https://arxiv.org/pdf/1409.7495.pdf.
        'ce_dom_d_sdda_end_ep': -1,  # epoch when to stop using this loss.
        # -1: never stop. -> start epoch is set by `adaptation_start_epoch`

        # 14- neighborhood loss. [NRC]
        'nrc_neighborhood': False,  # ON/OFF.
        'nrc_na': False,  # ON/OFF.
        'nrc_na_lambda': 1.0,  #float > 0.
        'nrc_ena': False,  # ON/OFF.
        'nrc_ena_lambda': 1.0,  # float > 0.
        'nrc_kl': False,  # ON/OFF.
        'nrc_kl_lambda': 1.0,  # float > 0.
        'nrc_epsilon': 0.001,  # float > 0.
        'nrc_k': 2,  # K nearest neighbors
        'nrc_kk': 2,  # KK-nearest neighbors of each neighbor
        # 15- CDD over image global class prediction using MMD
        'cdd_lambda': 0.01, # lambda of this term. >= 0.
        'cdd_pseudo_lb': False,  # CDD over image global class prediction.
        'cdd_pseudo_lb_kernel_num': [5, 5], # number of kernels for MMD.
        'cdd_pseudo_lb_kernel_mul': [2, 2],
        'cdd_pseudo_lb_num_layers': 2,  # number of layers for MMD.
        'cdd_pseudo_lb_num_classes': 2,  # number of classes for MMD.
        'cdd_pseudo_lb_intra_only': False,  # if true, we use only intra-class
        'cdd_variance':1.0, # variance of the gaussian estimation for the source features
        # 16- Contrastive loss  CDCL
        'cdcl_pseudo_lb': False,  # on/off.
        'cdcl_tau': 0.05,  # lambda of this term. >= 0.
        'cdcl_lambda': 0.01,  # lambda of this term. >= 0.
    }

    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = args['dataset']
    if dsname.startswith(pre):
        dsname = dsname.replace('{}_'.format(pre), '')

    args['data_paths'] = configure_data_paths(args, dsname)
    if args['dataset'] == constants.BREAKHIS:
        args['metadata_root'] = join(
            args['metadata_root'], args['dataset'], args['magnification'],
            f"fold-{args['fold'] + 1}")
    else:
        args['metadata_root'] = join(args['metadata_root'], args['dataset'],
                                     f"fold-{args['fold']}")

    args['mask_root'] = join(args['mask_root'], dsname)

    data_cams = join(root_dir, constants.DATA_CAMS)
    if not os.path.isdir(data_cams):
        os.makedirs(data_cams)

    return args


if __name__ == '__main__':
    args = get_config(constants.GLAS, fold=0, magnification=constants.MAG40X)
    print(args['metadata_root'])


