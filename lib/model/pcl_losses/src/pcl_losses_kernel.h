#ifndef _PCL_LOSSES_KERNEL
#define _PCL_LOSSES_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int PCLLossesForwardLaucher(
    const float* bottom_data, const float* labels, const float* cls_loss_weights,
    const float* pc_labels, const float* pc_probs, const float* img_cls_loss_weights, 
    const float* im_labels, const int batch_size, const int channels, 
    const int num_positive, float* top_data, cudaStream_t stream);


int PCLLossesBackwardLaucher(const float* top_diff, const float* prob_data, 
    const float* labels, const float* cls_loss_weights, const float* gt_assignment,
    const float* pc_labels, const float* pc_probs, const float* pc_count,
    const float* img_cls_loss_weights, const float* im_labels, const int batch_size, 
    const int channels, float* bottom_diff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

