from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.boxes as box_utils
from core.config import cfg

import numpy as np
from sklearn.cluster import KMeans

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def PCL(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps

    proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(),
        im_labels.copy())

    labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights = _get_proposal_clusters(boxes.copy(),
            proposals, im_labels.copy(), cls_prob_new.copy())

    return {'labels' : labels.reshape(1, -1).astype(np.float32).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'pc_labels' : pc_labels.reshape(1, -1).astype(np.float32).copy(),
            'pc_probs' : pc_probs.reshape(1, -1).astype(np.float32).copy(),
            'pc_count' : pc_count.reshape(1, -1).astype(np.float32).copy(),
            'img_cls_loss_weights' : img_cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'im_labels_real' : np.hstack((np.array([[1]]), im_labels)).astype(np.float32).copy()}

def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=cfg.TRAIN.NUM_KMEANS_CLUSTER,
        random_state=cfg.RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index

def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = box_utils.bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))

    return (overlaps > iou_threshold).astype(np.float32)

def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]

            graph = _build_graph(boxes_tmp, cfg.TRAIN.GRAPH_IOU_THRESHOLD)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)\
                [-1:(-1 - min(len(gt_scores_tmp), cfg.TRAIN.MAX_PC_NUM)):-1]

            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores,
                gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes,
                (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals

def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)

    for i in xrange(gt_boxes.shape[0]):
        po_index = np.where(gt_assignment == i)[0]
        img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])

    return labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count, img_cls_loss_weights


class PCLLosses(torch.autograd.Function):

    def forward(ctx, pcl_probs, labels, cls_loss_weights,
                gt_assignment, pc_labels, pc_probs, pc_count,
                img_cls_loss_weights, im_labels):
        ctx.pcl_probs, ctx.labels, ctx.cls_loss_weights, \
        ctx.gt_assignment, ctx.pc_labels, ctx.pc_probs, \
        ctx.pc_count, ctx.img_cls_loss_weights, ctx.im_labels = \
        pcl_probs, labels, cls_loss_weights, gt_assignment, \
        pc_labels, pc_probs, pc_count, img_cls_loss_weights, im_labels

        batch_size, channels = pcl_probs.size()
        loss = 0
        ctx.mark_non_differentiable(labels, cls_loss_weights,
                                    gt_assignment, pc_labels, pc_probs,
                                    pc_count, img_cls_loss_weights, im_labels)

        for c in range(channels):
            if im_labels[0, c] != 0:
                if c == 0:
                    for i in range(batch_size):
                        if labels[0, i] == 0:
                            loss -= cls_loss_weights[0, i] * torch.log(pcl_probs[i, c])
                else:
                    for i in range(pc_labels.size(0)):
                        if pc_probs[0, i] == c:
                            loss -= img_cls_loss_weights[0, i] * torch.log(pc_probs[0, i])

        return loss / batch_size

    def backward(ctx, grad_output):
        pcl_probs, labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights, im_labels = \
        ctx.pcl_probs, ctx.labels, ctx.cls_loss_weights, \
        ctx.gt_assignment, ctx.pc_labels, ctx.pc_probs, \
        ctx.pc_count, ctx.img_cls_loss_weights, ctx.im_labels

        grad_input = grad_output.new(pcl_probs.size()).zero_()

        batch_size, channels = pcl_probs.size()

        for i in range(batch_size):
            for c in range(channels):
                grad_input[i, c] = 0
                if im_labels[0, c] != 0:
                    if c == 0:
                        if labels[0, i] == 0:
                            grad_input[i, c] = -cls_loss_weights[0, i] / pcl_probs[i, c]
                    else:
                        if labels[0, i] == c:
                            pc_index = int(gt_assignment[0, i].item())
                            if c != pc_labels[0, pc_index]:
                                print('labels mismatch.')
                            grad_input[i, c] = -img_cls_loss_weights[0, pc_index] / (pc_count[0, pc_index] * pc_probs[0, pc_index])

        grad_input /= batch_size

        return grad_input, grad_output.new(labels.size()).zero_(), grad_output.new(cls_loss_weights.size()).zero_(), \
               grad_output.new(gt_assignment.size()).zero_(), grad_output.new(pc_labels.size()).zero_(), \
               grad_output.new(pc_probs.size()).zero_(), grad_output.new(pc_count.size()).zero_(), \
               grad_output.new(img_cls_loss_weights.size()).zero_(), grad_output.new(im_labels.size()).zero_()
