int pcl_losses_forward_cuda(THCudaTensor * pcl_probs, THCudaTensor * labels, 
                            THCudaTensor * cls_loss_weights, THCudaTensor * pc_labels,
                            THCudaTensor * pc_probs, THCudaTensor * img_cls_loss_weights, 
                            THCudaTensor * im_labels, THCudaTensor * output);

int pcl_losses_backward_cuda(THCudaTensor * pcl_probs, THCudaTensor * labels, 
                             THCudaTensor * cls_loss_weights, THCudaTensor * gt_assignment,
                             THCudaTensor * pc_labels, THCudaTensor * pc_probs, 
                             THCudaTensor * pc_count, THCudaTensor * img_cls_loss_weights, 
                             THCudaTensor * im_labels, THCudaTensor * top_grad, 
                             THCudaTensor * bottom_grad);