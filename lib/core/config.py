from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
from packaging import version
import torch
import torch.nn as nn
from torch.nn import init
import yaml

import nn as mynn
from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Datasets to train on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, the model is trained on their union
__C.TRAIN.DATASETS = ()

# Scales to use during training
# Each scale is the pixel size of an image's shortest side
# If multiple scales are listed, then one is selected uniformly at random for
# each training image (i.e., scale jitter data augmentation)
__C.TRAIN.SCALES = (600, )

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images *per GPU* in the training minibatch
# Total images per minibatch = TRAIN.IMS_PER_BATCH * NUM_GPUS
__C.TRAIN.IMS_PER_BATCH = 2

# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
# E.g., a common configuration is: 512 * 2 * 8 = 8192
__C.TRAIN.BATCH_SIZE_PER_IM = 64

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train using these proposals
# During training, all proposals specified in the file are used (no limit is
# applied)
# Proposal files must be in correspondence with the datasets listed in
# TRAIN.DATASETS
__C.TRAIN.PROPOSAL_FILES = ()

# Snapshot (model checkpoint) period
# Divide by NUM_GPUS to determine actual period (e.g., 20000/8 => 2500 iters)
# to allow for linear training schedule scaling
__C.TRAIN.SNAPSHOT_ITERS = 10000

# Filter proposals that are inside of crowd regions by CROWD_FILTER_THRESH
# "Inside" is measured as: proposal-with-crowd intersection area divided by
# proposal area
__C.TRAIN.CROWD_FILTER_THRESH = 0

# Ignore ground-truth objects with area < this threshold
__C.TRAIN.GT_MIN_AREA = -1

# Freeze the backbone architecture during training if set to True
__C.TRAIN.FREEZE_CONV_BODY = False

# The maximum number of proposal clusters
__C.TRAIN.MAX_PC_NUM = 5

# The number of k-means clusters
__C.TRAIN.NUM_KMEANS_CLUSTER = 3

# The IoU threshold to build graph
__C.TRAIN.GRAPH_IOU_THRESHOLD = 0.4

__C.TRAIN.FG_THRESH = 0.5

__C.TRAIN.BG_THRESH = 0.1

# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = AttrDict()

# Number of Python threads to use for the data loader (warning: using too many
# threads can cause GIL-based interference with Python Ops leading to *slower*
# training; 4 seems to be the sweet spot in our experience)
__C.DATA_LOADER.NUM_THREADS = 4


# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Datasets to test on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, testing is performed on each one sequentially
__C.TEST.DATASETS = ()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALE = 600

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Test using these proposal files (must correspond with TEST.DATASETS)
__C.TEST.PROPOSAL_FILES = ()

# Limit on the number of proposals per image used during inference
__C.TEST.PROPOSAL_LIMIT = -1

# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
__C.TEST.DETECTIONS_PER_IM = 100

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
__C.TEST.SCORE_THRESH = 1e-5

# Save detection results files if True
# If false, results files are cleaned up (they can be large) after local
# evaluation
__C.TEST.COMPETITION_MODE = True

# Evaluate detections with the COCO json dataset eval code even if it's not the
# evaluation code for the dataset (e.g. evaluate PASCAL VOC results using the
# COCO API to get COCO style AP on PASCAL VOC)
__C.TEST.FORCE_JSON_DATASET_EVAL = False

# [Inferred value; do not set directly in a config]
# Indicates if precomputed proposals are used at test time
# Not set for 1-stage models and 2-stage models with RPN subnetwork enabled
__C.TEST.PRECOMPUTED_PROPOSALS = True


# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_2x.yaml for an example
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_AUG = AttrDict()

# Enable test-time augmentation for bounding box detection if True
__C.TEST.BBOX_AUG.ENABLED = False

# Heuristic used to combine predicted box scores
#   Valid options: ('ID', 'AVG', 'UNION')
__C.TEST.BBOX_AUG.SCORE_HEUR = 'AVG'

# Heuristic used to combine predicted box coordinates
#   Valid options: ('ID', 'AVG', 'UNION')
__C.TEST.BBOX_AUG.COORD_HEUR = 'ID'

# Horizontal flip at the original scale (id transform)
__C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
__C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
__C.TEST.BBOX_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
__C.TEST.BBOX_AUG.SCALE_H_FLIP = False

# Apply scaling based on object size
__C.TEST.BBOX_AUG.SCALE_SIZE_DEP = False
__C.TEST.BBOX_AUG.AREA_TH_LO = 50**2
__C.TEST.BBOX_AUG.AREA_TH_HI = 180**2

# Each aspect ratio is relative to image width
__C.TEST.BBOX_AUG.ASPECT_RATIOS = ()

# Horizontal flip at each aspect ratio
__C.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP = False

# ---------------------------------------------------------------------------- #
# Soft NMS
# ---------------------------------------------------------------------------- #
__C.TEST.SOFT_NMS = AttrDict()

# Use soft NMS instead of standard NMS if set to True
__C.TEST.SOFT_NMS.ENABLED = False
# See soft NMS paper for definition of these options
__C.TEST.SOFT_NMS.METHOD = 'linear'
__C.TEST.SOFT_NMS.SIGMA = 0.5
# For the soft NMS overlap threshold, we simply use TEST.NMS

# ---------------------------------------------------------------------------- #
# Bounding box voting (from the Multi-Region CNN paper)
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_VOTE = AttrDict()

# Use box voting if set to True
__C.TEST.BBOX_VOTE.ENABLED = False

# We use TEST.NMS threshold for the NMS step. VOTE_TH overlap threshold
# is used to select voting boxes (IoU >= VOTE_TH) for each box that survives NMS
__C.TEST.BBOX_VOTE.VOTE_TH = 0.8

# The method used to combine scores when doing bounding box voting
# Valid options include ('ID', 'AVG', 'IOU_AVG', 'GENERALIZED_AVG', 'QUASI_SUM')
__C.TEST.BBOX_VOTE.SCORING_METHOD = 'ID'

# Hyperparameter used by the scoring method (it has different meanings for
# different methods)
__C.TEST.BBOX_VOTE.SCORING_METHOD_BETA = 1.0


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The type of model to use
# The string must match a function in the modeling.model_builder module
# (e.g., 'generalized_rcnn', 'mask_rcnn', ...)
__C.MODEL.TYPE = ''

# The backbone conv body to use
__C.MODEL.CONV_BODY = ''

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = -1

# Whether to load imagenet pretrained weights
# If True, path to the weight file must be specified.
# See: __C.RESNETS.IMAGENET_PRETRAINED_WEIGHTS
__C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = True

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
#
# In older versions, the weights were set such that the regression deltas
# would have unit standard deviation on the training dataset. Presently, rather
# than computing these statistics exactly, we use a fixed set of weights
# (10., 10., 5., 5.) by default. These are approximately the weights one would
# get from COCO using the previous unit stdev heuristic.
__C.MODEL.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# With Fast R-CNN branch
__C.MODEL.WITH_FRCNN = True

# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# e.g 'SGD', 'Adam'
__C.SOLVER.TYPE = 'SGD'

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = 'step'

# Some LR Policies (by example):
# 'step'
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** (cur_iter // SOLVER.STEP_SIZE)
# 'steps_with_decay'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.GAMMA = 0.1
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** current_step
#   iters [0, 59999] are in current_step = 0, iters [60000, 79999] are in
#   current_step = 1, and so on
# 'steps_with_lrs'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.LRS = [0.02, 0.002, 0.0002]
#   lr = LRS[current_step]

# Hyperparameter used by the specified policy
# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Uniform step size for 'steps' policy
__C.SOLVER.STEP_SIZE = 30000

# Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs'
# policies
__C.SOLVER.STEPS = []

# Learning rates to use with 'steps_with_lrs' policy
__C.SOLVER.LRS = []

# Maximum number of SGD iterations
__C.SOLVER.MAX_ITER = 40000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005
# L2 regularization hyperparameter for GroupNorm's parameters
__C.SOLVER.WEIGHT_DECAY_GN = 0.0

# Whether to double the learning rate for bias
__C.SOLVER.BIAS_DOUBLE_LR = True

# Whether to have weight decay on bias as well
__C.SOLVER.BIAS_WEIGHT_DECAY = False

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0

# WARM_UP_METHOD can be either 'constant' or 'linear' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'linear'

# Scale the momentum update history by new_lr / old_lr when updating the
# learning rate (this is correct given MomentumSGDUpdateOp)
__C.SOLVER.SCALE_MOMENTUM = True
# Only apply the correction if the relative LR change exceeds this threshold
# (prevents ever change in linear warm up from scaling the momentum by a tiny
# amount; momentum scaling is only important if the LR change is large)
__C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1


# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN = AttrDict()

# The type of RoI head to use for bounding box classification and regression
# The string must match a function this is imported in modeling.model_builder
# (e.g., 'head_builder.add_roi_2mlp_head' to specify a two hidden layer MLP)
__C.FAST_RCNN.ROI_BOX_HEAD = ''

# Hidden layer dimension when using an MLP for the RoI box head
__C.FAST_RCNN.MLP_HEAD_DIM = 1024

# Hidden Conv layer dimension when using Convs for the RoI box head
__C.FAST_RCNN.CONV_HEAD_DIM = 256
# Number of stacked Conv layers in the RoI box head
__C.FAST_RCNN.NUM_STACKED_CONVS = 4

# RoI transformation function (e.g., RoIPool or RoIAlign)
# (RoIPoolF is the same as RoIPool; ignore the trailing 'F')
__C.FAST_RCNN.ROI_XFORM_METHOD = 'RoIPoolF'

# Number of grid sampling points in RoIAlign (usually use 2)
# Only applies to RoIAlign
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 0

# RoI transform output resolution
# Note: some models may have constraints on what they can use, e.g. they use
# pretrained FC layers like in VGG16, and will ignore this option
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = 14



# ---------------------------------------------------------------------------- #
# VGG options
# ---------------------------------------------------------------------------- #
__C.VGG = AttrDict()

# Freeze model weights before and including which block.
# Choices: [0, 2, 3, 4, 5]. O means not fixed. First conv and bn are defaults to
# be fixed.
__C.VGG.FREEZE_AT = 2

# Path to pretrained resnet weights on ImageNet.
# If start with '/', then it is treated as a absolute path.
# Otherwise, treat as a relative path to __C.ROOT_DIR
__C.VGG.IMAGENET_PRETRAINED_WEIGHTS = ''



# ---------------------------------------------------------------------------- #
# MISC options
# ---------------------------------------------------------------------------- #

# Numer of refinement times
__C.REFINE_TIMES = 3

# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/8 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1. / 8.

# Clip bounding box transformation predictions to prevent np.exp from
# overflowing
# Heuristic choice based on that would scale a 16 pixel anchor up to 1000 pixels
__C.BBOX_XFORM_CLIP = np.log(1000. / 8.)

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# "Fun" fact: the history of where these values comes from is lost (From Detectron lol)
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 4

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Output basedir
__C.OUTPUT_DIR = 'Outputs'

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Dump detection visualizations
__C.VIS = False

# Score threshold for visualization
__C.VIS_TH = 0.9

# Expected results should take the form of a list of expectations, each
# specified by four elements (dataset, task, metric, expected value). For
# example: [['coco_2014_minival', 'box_proposal', 'AR@1000', 0.387]]
__C.EXPECTED_RESULTS = []
# Absolute and relative tolerance to use when comparing to EXPECTED_RESULTS
__C.EXPECTED_RESULTS_RTOL = 0.1
__C.EXPECTED_RESULTS_ATOL = 0.005
# Set to send email in case of an EXPECTED_RESULTS failure
__C.EXPECTED_RESULTS_EMAIL = ''

# ------------------------------
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# [Deprecate]
__C.POOLING_MODE = 'crop'

# [Deprecate] Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

__C.CROP_RESIZE_WITH_MAX_POOL = True

# [Infered value]
__C.CUDA = False

__C.DEBUG = False

# [Infered value]
__C.PYTORCH_VERSION_LESS_THAN_040 = False


def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if __C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        assert __C.VGG.IMAGENET_PRETRAINED_WEIGHTS, \
            "Path to the weight file must not be empty to load imagenet pertrained resnets."
    if version.parse(torch.__version__) < version.parse('0.4.0'):
        __C.PYTORCH_VERSION_LESS_THAN_040 = True
        # create alias for PyTorch version less than 0.4.0
        init.uniform_ = init.uniform
        init.normal_ = init.normal
        init.constant_ = init.constant
        nn.GroupNorm = mynn.GroupNorm
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
