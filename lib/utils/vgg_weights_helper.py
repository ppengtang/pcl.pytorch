"""
Helper functions for converting resnet pretrained weights from other formats
"""
import os
import pickle

import torch

import nn as mynn
import utils.detectron_weight_helper as dwh
from core.config import cfg


def load_pretrained_imagenet_weights(model):
    """Load pretrained weights
    Args:
        model: the generalized rcnnn module
    """
    _, ext = os.path.splitext(cfg.VGG.IMAGENET_PRETRAINED_WEIGHTS)
    if ext == '.pkl':
        with open(cfg.VGG.IMAGENET_PRETRAINED_WEIGHTS, 'rb') as fp:
            src_blobs = pickle.load(fp, encoding='latin1')
        if 'blobs' in src_blobs:
            src_blobs = src_blobs['blobs']
        pretrianed_state_dict = src_blobs
    else:
        weights_file = os.path.join(cfg.ROOT_DIR, cfg.VGG.IMAGENET_PRETRAINED_WEIGHTS)
        pretrianed_state_dict = convert_state_dict(torch.load(weights_file))


    model_state_dict = model.state_dict()

    pattern = dwh.vgg_weights_name_pattern()

    name_mapping, _ = model.detectron_weight_mapping

    for k, v in name_mapping.items():
        if isinstance(v, str):  # maybe a str, None or True
            if pattern.match(v):
                pretrianed_key = k.split('.', 1)[-1]
                if ext == '.pkl':
                    model_state_dict[k].copy_(torch.Tensor(pretrianed_state_dict[v]))
                else:
                    model_state_dict[k].copy_(pretrianed_state_dict[pretrianed_key])


def convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our vgg conv_body and box_head.
    """
    dst_dict = {}
    dst_dict['conv1.0.weight'] = src_dict['features.0.weight']
    dst_dict['conv1.0.bias'] = src_dict['features.0.bias']
    dst_dict['conv1.2.weight'] = src_dict['features.2.weight']
    dst_dict['conv1.2.bias'] = src_dict['features.2.bias']
    dst_dict['conv2.0.weight'] = src_dict['features.5.weight']
    dst_dict['conv2.0.bias'] = src_dict['features.5.bias']
    dst_dict['conv2.2.weight'] = src_dict['features.7.weight']
    dst_dict['conv2.2.bias'] = src_dict['features.7.bias']
    dst_dict['conv3.0.weight'] = src_dict['features.10.weight']
    dst_dict['conv3.0.bias'] = src_dict['features.10.bias']
    dst_dict['conv3.2.weight'] = src_dict['features.12.weight']
    dst_dict['conv3.2.bias'] = src_dict['features.12.bias']
    dst_dict['conv3.4.weight'] = src_dict['features.14.weight']
    dst_dict['conv3.4.bias'] = src_dict['features.14.bias']
    dst_dict['conv4.0.weight'] = src_dict['features.17.weight']
    dst_dict['conv4.0.bias'] = src_dict['features.17.bias']
    dst_dict['conv4.2.weight'] = src_dict['features.19.weight']
    dst_dict['conv4.2.bias'] = src_dict['features.19.bias']
    dst_dict['conv4.4.weight'] = src_dict['features.21.weight']
    dst_dict['conv4.4.bias'] = src_dict['features.21.bias']
    dst_dict['conv5.0.weight'] = src_dict['features.24.weight']
    dst_dict['conv5.0.bias'] = src_dict['features.24.bias']
    dst_dict['conv5.2.weight'] = src_dict['features.26.weight']
    dst_dict['conv5.2.bias'] = src_dict['features.26.bias']
    dst_dict['conv5.4.weight'] = src_dict['features.28.weight']
    dst_dict['conv5.4.bias'] = src_dict['features.28.bias']
    dst_dict['fc1.weight'] = src_dict['classifier.0.weight']
    dst_dict['fc1.bias'] = src_dict['classifier.0.bias']
    dst_dict['fc2.weight'] = src_dict['classifier.3.weight']
    dst_dict['fc2.bias'] = src_dict['classifier.3.bias']
    return dst_dict
