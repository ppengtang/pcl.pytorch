"""Perform re-evaluation on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
from six.moves import cPickle as pickle

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import empty_results, extend_results
from core.test import box_results_for_corloc, box_results_with_nms_and_limit
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--result_path',
        help='the path for result file.')
    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results.')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    return parser.parse_args()


if __name__ == '__main__':

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert os.path.exists(args.result_path)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.result_path)
        logger.info('Automatically set output directory to %s', args.output_dir)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2014":
        cfg.TEST.DATASETS = ('coco_2014_val',)
        cfg.MODEL.NUM_CLASSES = 80
    elif args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 80
    elif args.dataset == 'voc2007test':
        cfg.TEST.DATASETS = ('voc_2007_test',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2012test':
        cfg.TEST.DATASETS = ('voc_2012_test',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2007trainval':
        cfg.TEST.DATASETS = ('voc_2007_trainval',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2012trainval':
        cfg.TEST.DATASETS = ('voc_2012_trainval',)
        cfg.MODEL.NUM_CLASSES = 20
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Re-evaluation with config:')
    logger.info(pprint.pformat(cfg))

    with open(args.result_path, 'rb') as f:
        results = pickle.load(f)
        logger.info('Loading results from {}.'.format(args.result_path))
    all_boxes = results['all_boxes']

    dataset_name = cfg.TEST.DATASETS[0]
    dataset = JsonDataset(dataset_name)
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
        dataset, final_boxes, args.output_dir, test_corloc
    )
