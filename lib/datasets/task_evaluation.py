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

"""Evaluation interface for supported tasks (box detection, instance
segmentation, keypoint detection, ...).


Results are stored in an OrderedDict with the following nested structure:

<dataset>:
  <task>:
    <metric>: <val>

<dataset> is any valid dataset (e.g., 'coco_2014_minival')
<task> is in ['box', 'mask', 'keypoint', 'box_proposal']
<metric> can be ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR@1000',
                 'ARs@1000', 'ARm@1000', 'ARl@1000', ...]
<val> is a floating point number
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import logging
import os
import pprint

from core.config import cfg
from utils.logging import send_email
import datasets.cityscapes_json_dataset_evaluator as cs_json_dataset_evaluator
import datasets.json_dataset_evaluator as json_dataset_evaluator
import datasets.voc_dataset_evaluator as voc_dataset_evaluator

logger = logging.getLogger(__name__)


def evaluate_all(
    dataset, all_boxes, output_dir, test_corloc=False, use_matlab=False
):
    """Evaluate "all" tasks, where "all" includes box detection, instance
    segmentation, and keypoint detection.
    """
    all_results = evaluate_boxes(
        dataset, all_boxes, output_dir, test_corloc=test_corloc,
        use_matlab=use_matlab
    )
    logger.info('Evaluating bounding boxes is done!')
    return all_results


def evaluate_boxes(dataset, all_boxes, output_dir, test_corloc=False, use_matlab=False):
    """Evaluate bounding box detection."""
    logger.info('Evaluating detections')
    not_comp = not cfg.TEST.COMPETITION_MODE
    if _use_json_dataset_evaluator(dataset):
        coco_eval = json_dataset_evaluator.evaluate_boxes(
            dataset, all_boxes, output_dir, use_salt=not_comp, cleanup=not_comp
        )
        box_results = _coco_eval_to_box_results(coco_eval)
    elif _use_voc_evaluator(dataset):
        # For VOC, always use salt and always cleanup because results are
        # written to the shared VOCdevkit results directory
        voc_eval = voc_dataset_evaluator.evaluate_boxes(
            dataset, all_boxes, output_dir, test_corloc=test_corloc,
            use_matlab=use_matlab
        )
        box_results = _voc_eval_to_box_results(voc_eval)
    else:
        raise NotImplementedError(
            'No evaluator for dataset: {}'.format(dataset.name)
        )
    return OrderedDict([(dataset.name, box_results)])


def log_copy_paste_friendly_results(results):
    """Log results in a format that makes it easy to copy-and-paste in a
    spreadsheet. Lines are prefixed with 'copypaste: ' to make grepping easy.
    """
    for dataset in results.keys():
        logger.info('copypaste: Dataset: {}'.format(dataset))
        for task, metrics in results[dataset].items():
            logger.info('copypaste: Task: {}'.format(task))
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            logger.info('copypaste: ' + ','.join(metric_names))
            logger.info('copypaste: ' + ','.join(metric_vals))


def check_expected_results(results, atol=0.005, rtol=0.1):
    """Check actual results against expected results stored in
    cfg.EXPECTED_RESULTS. Optionally email if the match exceeds the specified
    tolerance.

    Expected results should take the form of a list of expectations, each
    specified by four elements: [dataset, task, metric, expected value]. For
    example: [['coco_2014_minival', 'box_proposal', 'AR@1000', 0.387], ...].
    """
    # cfg contains a reference set of results that we want to check against
    if len(cfg.EXPECTED_RESULTS) == 0:
        return

    for dataset, task, metric, expected_val in cfg.EXPECTED_RESULTS:
        assert dataset in results, 'Dataset {} not in results'.format(dataset)
        assert task in results[dataset], 'Task {} not in results'.format(task)
        assert metric in results[dataset][task], \
            'Metric {} not in results'.format(metric)
        actual_val = results[dataset][task][metric]
        err = abs(actual_val - expected_val)
        tol = atol + rtol * abs(expected_val)
        msg = (
            '{} > {} > {} sanity check (actual vs. expected): '
            '{:.3f} vs. {:.3f}, err={:.3f}, tol={:.3f}'
        ).format(dataset, task, metric, actual_val, expected_val, err, tol)
        if err > tol:
            msg = 'FAIL: ' + msg
            logger.error(msg)
            if cfg.EXPECTED_RESULTS_EMAIL != '':
                subject = 'Detectron end-to-end test failure'
                job_name = os.environ[
                    'DETECTRON_JOB_NAME'
                ] if 'DETECTRON_JOB_NAME' in os.environ else '<unknown>'
                job_id = os.environ[
                    'WORKFLOW_RUN_ID'
                ] if 'WORKFLOW_RUN_ID' in os.environ else '<unknown>'
                body = [
                    'Name:',
                    job_name,
                    'Run ID:',
                    job_id,
                    'Failure:',
                    msg,
                    'Config:',
                    pprint.pformat(cfg),
                    'Env:',
                    pprint.pformat(dict(os.environ)),
                ]
                send_email(
                    subject, '\n\n'.join(body), cfg.EXPECTED_RESULTS_EMAIL
                )
        else:
            msg = 'PASS: ' + msg
            logger.info(msg)


def _use_json_dataset_evaluator(dataset):
    """Check if the dataset uses the general json dataset evaluator."""
    return dataset.name.find('coco_') > -1 or cfg.TEST.FORCE_JSON_DATASET_EVAL


def _use_voc_evaluator(dataset):
    """Check if the dataset uses the PASCAL VOC dataset evaluator."""
    return dataset.name[:4] == 'voc_'


# Indices in the stats array for COCO boxes and masks
COCO_AP = 0
COCO_AP50 = 1
COCO_AP75 = 2
COCO_APS = 3
COCO_APM = 4
COCO_APL = 5


# ---------------------------------------------------------------------------- #
# Helper functions for producing properly formatted results.
# ---------------------------------------------------------------------------- #

def _coco_eval_to_box_results(coco_eval):
    res = _empty_box_results()
    if coco_eval is not None:
        s = coco_eval.stats
        res['box']['AP'] = s[COCO_AP]
        res['box']['AP50'] = s[COCO_AP50]
        res['box']['AP75'] = s[COCO_AP75]
        res['box']['APs'] = s[COCO_APS]
        res['box']['APm'] = s[COCO_APM]
        res['box']['APl'] = s[COCO_APL]
    return res


def _voc_eval_to_box_results(voc_eval):
    # Not supported (return empty results)
    return _empty_box_results()


def _empty_box_results():
    return OrderedDict({
        'box':
        OrderedDict(
            [
                ('AP', -1),
                ('AP50', -1),
                ('AP75', -1),
                ('APs', -1),
                ('APm', -1),
                ('APl', -1),
                ('CorLoc', -1)
            ]
        )
    })
