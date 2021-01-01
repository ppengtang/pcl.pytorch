# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
from mmcv.ops import (RoIPool, RoIAlign, roi_pool, roi_align, nms, soft_nms)

# yapf: enable

__all__ = [
    'RoIPool', 'RoIAlign', 'roi_pool', 'roi_align', 'nms', 'soft_nms'
]
