# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from core.test import box_results_for_corloc, box_results_with_nms_and_limit
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
from utils.io import save_object
from utils.timer import Timer

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    # Generic case that handles all network types other than RPN-only nets
    # and RetinaNet
    child_func = test_net
    parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        if not (dataset_name + '.pkl') in cfg.TEST.PROPOSAL_FILES[index]:
            proposal_file = os.path.join(cfg.TEST.PROPOSAL_FILES[index],
                                         dataset_name + '.pkl')
        else:
            proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))

    roidb = dataset.get_roidb()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    final_boxes = empty_results(num_classes, num_images)
    test_corloc = 'train' in dataset_name
    for i, entry in enumerate(roidb):
        boxes = all_boxes[entry['image']]
        if test_corloc:
            _, _, cls_boxes_i = box_results_for_corloc(boxes['scores'], boxes['boxes'])
        else:
            _, _, cls_boxes_i = box_results_with_nms_and_limit(boxes['scores'],
                                                         boxes['boxes'])
        extend_results(i, final_boxes, cls_boxes_i)
    results = task_evaluation.evaluate_all(
        dataset, final_boxes, output_dir, test_corloc
    )
    return results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    tag = 'discovery' if 'train' in dataset_name else 'detection'
    outputs = subprocess_utils.process_in_parallel(
        tag, num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_boxes = {}
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_boxes.update(all_boxes_batch)
    if 'train' in dataset_name:
        det_file = os.path.join(output_dir, 'discovery.pkl')
    else:
        det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes = {}
    timers = defaultdict(Timer)
    for i, entry in enumerate(roidb):
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes']
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None

        im = cv2.imread(entry['image'])
        cls_boxes_i = im_detect_all(model, im, box_proposals, timers)

        all_boxes[entry['image']] = cls_boxes_i

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, eta
                )
            )

    cfg_yaml = yaml.dump(cfg)
    if 'train' in dataset_name:
        if ind_range is not None:
            det_name = 'discovery_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'discovery.pkl'
    else:
        if ind_range is not None:
            det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
