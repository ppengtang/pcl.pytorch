int pcl_losses_forward(THFloatTensor * pcl_probs, THFloatTensor * labels, 
                       THFloatTensor * cls_loss_weights, THFloatTensor * pc_labels,
                       THFloatTensor * pc_probs, THFloatTensor * img_cls_loss_weights, 
                       THFloatTensor * im_labels, THFloatTensor * output);

int pcl_losses_backward(THFloatTensor * pcl_probs, THFloatTensor * labels, 
                        THFloatTensor * cls_loss_weights, THFloatTensor * gt_assignment,
                        THFloatTensor * pc_labels, THFloatTensor * pc_probs, 
                        THFloatTensor * pc_count, THFloatTensor * img_cls_loss_weights, 
                        THFloatTensor * im_labels, THFloatTensor * top_grad, 
                        THFloatTensor * bottom_grad);