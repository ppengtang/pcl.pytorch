from torch.nn.modules.module import Module
from ..functions.pcl_losses import PCLLosses


class _PCL_Losses(Module):

    def forward(self, pcl_prob, labels, cls_loss_weights, gt_assignment,
                pc_labels, pc_probs, pc_count, img_cls_loss_weights,
                im_labels_real):
        return PCLLosses()(pcl_prob, labels, cls_loss_weights, gt_assignment,
                           pc_labels, pc_probs, pc_count, img_cls_loss_weights,
                           im_labels_real)
