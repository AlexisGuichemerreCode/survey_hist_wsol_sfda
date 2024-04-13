import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from dlib.losses.jaccard import JaccardLoss
from dlib.losses.dice import DiceLoss
from dlib.losses.focal import FocalLoss
from dlib.losses.lovasz import LovaszLoss
from dlib.losses.soft_bce import SoftBCEWithLogitsLoss
from dlib.losses.soft_ce import SoftCrossEntropyLoss

from dlib.losses.core import MasterLoss
from dlib.losses.core import ClLoss
from dlib.losses.core import SpgLoss
from dlib.losses.core import AcolLoss
from dlib.losses.core import CutMixLoss
from dlib.losses.core import MaxMinLoss
from dlib.losses.core import SegLoss
from dlib.losses.core import ImgReconstruction
from dlib.losses.core import SelfLearningFcams
from dlib.losses.core import ConRanFieldFcams
from dlib.losses.core import EntropyFcams
from dlib.losses.core import MaxSizePositiveFcams

from dlib.losses.core import SelfLearningNegev
from dlib.losses.core import ConRanFieldNegev
from dlib.losses.core import JointConRanFieldNegev
from dlib.losses.core import MaxSizePositiveNegev
from dlib.losses.core import NegativeSamplesNegev

from dlib.losses.sf_uda import UdaCrossEntropyImgPseudoLabels
from dlib.losses.sf_uda import UdaTargetClassProbEntropy
from dlib.losses.sf_uda import UdaDiversityTargetClass
from dlib.losses.sf_uda import UdaCutMixLoss
from dlib.losses.sf_uda import UdaAcolLoss
from dlib.losses.sf_uda import UdaSpgLoss
from dlib.losses.sf_uda import UdaMaxMinLoss

from dlib.losses.sf_uda import UdaFeatureViewsConsistencyFaust
from dlib.losses.sf_uda import UdaClassProbsViewsSoftLabelsFaust
from dlib.losses.sf_uda import UdaMcDropoutVarMinFaust
from dlib.losses.sf_uda import UdaClassProbsEntropyFaust

from dlib.losses.sf_uda import UdaNANrc
from dlib.losses.sf_uda import UdaENANrc
from dlib.losses.sf_uda import UdaKLNrc

from dlib.losses.sf_uda_sdda import UdaSddaAdvGenerator
from dlib.losses.sf_uda_sdda import UdaSddaAdvDiscriminator
from dlib.losses.sf_uda_sdda import UdaSddaSrcModelCeFakeImage
from dlib.losses.sf_uda_sdda import UdaSddaTrgModelCeFakeImage
from dlib.losses.sf_uda_sdda import UdaSddaDomainDiscriminator
from dlib.losses.sf_uda_sdda import UdaSddaSrcModelPxLikelihood
from dlib.losses.sf_uda_sdda import SpgUdaSddaTrgModelCeFakeImage
from dlib.losses.sf_uda_sdda import AcolUdaSddaTrgModelCeFakeImage
from dlib.losses.sf_uda_sdda import CutMixUdaSddaTrgModelCeFakeImage
from dlib.losses.sf_uda_sdda import MaxMinUdaSddaTrgModelCeFakeImage

from dlib.losses.sf_uda import UdaCdd
from dlib.losses.sf_uda import UdaCdcl

from dlib.losses.core import SatLoss