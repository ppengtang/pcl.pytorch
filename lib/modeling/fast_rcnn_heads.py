import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
from model.pcl.pcl import get_proposal_clusters, _get_highest_score_proposals
import nn as mynn
import utils.net as net_utils

import numpy as np

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, dim_out)
        self.bbox_pred = nn.Linear(dim_in, dim_out * 4)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


def get_fast_rcnn_targets(boxes, refine_score, im_labels):
    cls_prob = refine_score[-1].data.cpu().numpy()

    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps

    # proposals = _get_highest_score_proposals(boxes.copy(), cls_prob.copy(), im_labels.copy())
    proposals = _get_highest_score_proposals(boxes.copy(), cls_prob.copy(), im_labels.copy())

    labels, cls_loss_weights, _, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())

    return labels.reshape(-1).astype(np.int64).copy(), \
           cls_loss_weights.reshape(-1).astype(np.float32).copy(), \
           bbox_targets.astype(np.float32).copy(), \
           bbox_inside_weights.astype(np.float32).copy(), \
           bbox_outside_weights.astype(np.float32).copy()


def fast_rcnn_losses(cls_score, bbox_pred, labels, cls_loss_weights,
                     bbox_targets, bbox_inside_weights, bbox_outside_weights):
    cls_loss = -(F.log_softmax(cls_score, dim=1)[range(cls_score.size(0)), labels].view(-1) * cls_loss_weights).mean()

    bbox_loss = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_loss_weights)

    return cls_loss, bbox_loss
