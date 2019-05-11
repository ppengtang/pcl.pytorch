#include <THC/THC.h>
#include <math.h>
#include "pcl_losses_kernel.h"

extern THCState *state;

int pcl_losses_forward_cuda(THCudaTensor * pcl_probs, THCudaTensor * labels, 
                            THCudaTensor * cls_loss_weights, THCudaTensor * pc_labels,
                            THCudaTensor * pc_probs, THCudaTensor * img_cls_loss_weights, 
                            THCudaTensor * im_labels, THCudaTensor * output)
{
    // Grab the input tensor
    float * prob_data_flat = THCudaTensor_data(state, pcl_probs);
    float * labels_flat = THCudaTensor_data(state, labels);
    float * cls_loss_weights_flat = THCudaTensor_data(state, cls_loss_weights);
    float * pc_labels_flat = THCudaTensor_data(state, pc_labels);
    float * pc_probs_flat = THCudaTensor_data(state, pc_probs);
    float * img_cls_loss_weights_flat = THCudaTensor_data(state, img_cls_loss_weights);
    float * im_labels_flat = THCudaTensor_data(state, im_labels);

    float * output_flat = THCudaTensor_data(state, output);

    int batch_size = THCudaTensor_size(state, pcl_probs, 0);
    int channels = THCudaTensor_size(state, pcl_probs, 1);
    int num_positive = THCudaTensor_size(state, pc_labels, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    PCLLossesForwardLaucher(
        prob_data_flat, labels_flat, cls_loss_weights_flat,
        pc_labels_flat, pc_probs_flat, img_cls_loss_weights_flat, 
        im_labels_flat, batch_size, channels, num_positive, 
        output_flat, stream);

    return 1;
}

int pcl_losses_backward_cuda(THCudaTensor * pcl_probs, THCudaTensor * labels, 
                             THCudaTensor * cls_loss_weights, THCudaTensor * gt_assignment,
                             THCudaTensor * pc_labels, THCudaTensor * pc_probs, 
                             THCudaTensor * pc_count, THCudaTensor * img_cls_loss_weights, 
                             THCudaTensor * im_labels, THCudaTensor * top_grad, 
                             THCudaTensor * bottom_grad)
{
    // Grab the input tensor
    float * prob_data_flat = THCudaTensor_data(state, pcl_probs);
    float * labels_flat = THCudaTensor_data(state, labels);
    float * cls_loss_weights_flat = THCudaTensor_data(state, cls_loss_weights);
    float * gt_assignment_flat = THCudaTensor_data(state, gt_assignment);
    float * pc_labels_flat = THCudaTensor_data(state, pc_labels);
    float * pc_probs_flat = THCudaTensor_data(state, pc_probs);
    float * pc_count_flat = THCudaTensor_data(state, pc_count);
    float * img_cls_loss_weights_flat = THCudaTensor_data(state, img_cls_loss_weights);
    float * im_labels_flat = THCudaTensor_data(state, im_labels);

    float * top_grad_flat = THCudaTensor_data(state, top_grad);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    int batch_size = THCudaTensor_size(state, pcl_probs, 0);
    int channels = THCudaTensor_size(state, pcl_probs, 1);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    PCLLossesBackwardLaucher(
        top_grad_flat, prob_data_flat, labels_flat, cls_loss_weights_flat,
        gt_assignment_flat, pc_labels_flat, pc_probs_flat, pc_count_flat,
        img_cls_loss_weights_flat, im_labels_flat, batch_size, channels,
        bottom_grad_flat, stream);

    return 1;
}
