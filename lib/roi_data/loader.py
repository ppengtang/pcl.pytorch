import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from core.config import cfg
from roi_data.minibatch import get_minibatch
import utils.blob as blob_utils
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes


class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, training=True):
        self._roidb = roidb
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self._roidb)

    def __getitem__(self, index_tuple):
        index, ratio = index_tuple
        single_db = [self._roidb[index]]
        blobs, valid = get_minibatch(single_db, self._num_classes)
        #TODO: Check if minibatch is valid ? If not, abandon it.
        # Need to change _worker_loop in torch.utils.data.dataloader.py.

        # Squeeze batch dim
        # for key in blobs:
        #     if key != 'roidb':
        #         blobs[key] = blobs[key].squeeze(axis=0)
        blobs['data'] = blobs['data'].squeeze(axis=0)

        return blobs

    def __len__(self):
        return self.DATA_SIZE


def cal_minibatch_ratio(ratio_list):
    """Given the ratio_list, we want to make the RATIO same for each minibatch on each GPU.
    Note: this only work for 1) cfg.TRAIN.MAX_SIZE is ignored during `prep_im_for_blob`
    and 2) cfg.TRAIN.SCALES containing SINGLE scale.
    Since all prepared images will have same min side length of cfg.TRAIN.SCALES[0], we can
     pad and batch images base on that.
    """
    DATA_SIZE = len(ratio_list)
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / cfg.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * cfg.TRAIN.IMS_PER_BATCH
        right_idx = min((i+1) * cfg.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx+1)] = target_ratio
    return ratio_list_minibatch


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_list, ratio_index):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.num_data = len(ratio_list)

    def __iter__(self):
        rand_perm = npr.permutation(self.num_data)
        ratio_list = self.ratio_list[rand_perm]
        ratio_index = self.ratio_index[rand_perm]
        # re-calculate minibatch ratio list
        ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data


class BatchSampler(torch_sampler.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, torch_sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    lists = []
    for blobs in list_of_blobs:
        lists.append({'data' : blobs.pop('data'),
                      'rois' : blobs.pop('rois'),
                      'labels' : blobs.pop('labels')})
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = lists[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        minibatch = default_collate(mini_list)
        for key in minibatch:
            Batch[key].append(minibatch[key])

    return Batch
