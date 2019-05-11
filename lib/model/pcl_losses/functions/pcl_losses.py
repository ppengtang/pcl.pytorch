import torch
from torch.autograd import Function
from .._ext import pcl_losses
import pdb

class PCLLosses(Function):

    def forward(ctx, pcl_probs, labels, cls_loss_weights, gt_assignment,
                pc_labels, pc_probs, pc_count, img_cls_loss_weights,
                im_labels):
        device_id = pcl_probs.get_device()
        pcl_probs = pcl_probs.data.cpu()
        ctx.save_for_backward(pcl_probs, labels, cls_loss_weights,
                              gt_assignment, pc_labels, pc_probs,
                              pc_count, img_cls_loss_weights, im_labels,
                              torch.tensor(device_id))

        output = pcl_probs.new(1, pcl_probs.shape[1]).zero_()

        pcl_losses.pcl_losses_forward(pcl_probs, labels, cls_loss_weights,
                                      pc_labels, pc_probs, img_cls_loss_weights,
                                      im_labels, output)

        return output.cuda(device_id).sum() / pcl_probs.size(0)

    def backward(ctx, grad_output):
        pcl_probs, labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights, im_labels, device_id = ctx.saved_tensors

        if grad_output.is_cuda:
            grad_output = grad_output.data.cpu()

        grad_input = grad_output.new(pcl_probs.size()).zero_()

        pcl_losses.pcl_losses_backward(pcl_probs, labels, cls_loss_weights,
                                       gt_assignment, pc_labels, pc_probs,
                                       pc_count, img_cls_loss_weights, im_labels,
                                       grad_output, grad_input)

        grad_input /= pcl_probs.size(0)

        return grad_input.cuda(device_id.item()), None, None, None, None, None, None, None, None
