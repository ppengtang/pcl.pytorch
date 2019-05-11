#include <TH/TH.h>
#include <math.h>

int pcl_losses_forward(THFloatTensor * pcl_probs, THFloatTensor * labels, 
                       THFloatTensor * cls_loss_weights, THFloatTensor * pc_labels,
                       THFloatTensor * pc_probs, THFloatTensor * img_cls_loss_weights, 
                       THFloatTensor * im_labels, THFloatTensor * output)
{
    // Grab the input tensor
    float * prob_data_flat = THFloatTensor_data(pcl_probs);
    float * labels_flat = THFloatTensor_data(labels);
    float * cls_loss_weights_flat = THFloatTensor_data(cls_loss_weights);
    float * pc_labels_flat = THFloatTensor_data(pc_labels);
    float * pc_probs_flat = THFloatTensor_data(pc_probs);
    float * img_cls_loss_weights_flat = THFloatTensor_data(img_cls_loss_weights);
    float * im_labels_flat = THFloatTensor_data(im_labels);

    float * output_flat = THFloatTensor_data(output);

    int batch_size = THFloatTensor_size(pcl_probs, 0);
    int channels = THFloatTensor_size(pcl_probs, 1);
    int num_positive = THFloatTensor_size(pc_labels, 1);

    float eps = 1e-6;

    for (int c = 0; c < channels; c++) {
        output_flat[c] = 0;
        if (im_labels_flat[c] != 0) {
            if (c == 0) {
                for (int i = 0; i < batch_size; i++) {
                    if (labels_flat[i] == 0) {
                        output_flat[c] -= cls_loss_weights_flat[i] * log(fmaxf(prob_data_flat[i * channels + c], eps));
                    }
                }
            }
            else {
                for (int i = 0; i < num_positive; i++) {
                    if (pc_labels_flat[i] == c) {
                        output_flat[c] -= img_cls_loss_weights_flat[i] * log(fmaxf(pc_probs_flat[i], eps));
                    }
                }
            }
        }
    }
    return 1;
}


int pcl_losses_backward(THFloatTensor * pcl_probs, THFloatTensor * labels, 
                        THFloatTensor * cls_loss_weights, THFloatTensor * gt_assignment,
                        THFloatTensor * pc_labels, THFloatTensor * pc_probs, 
                        THFloatTensor * pc_count, THFloatTensor * img_cls_loss_weights, 
                        THFloatTensor * im_labels, THFloatTensor * top_grad, 
                        THFloatTensor * bottom_grad)
{
    // Grab the input tensor
    float * prob_data_flat = THFloatTensor_data(pcl_probs);
    float * labels_flat = THFloatTensor_data(labels);
    float * cls_loss_weights_flat = THFloatTensor_data(cls_loss_weights);
    float * gt_assignment_flat = THFloatTensor_data(gt_assignment);
    float * pc_labels_flat = THFloatTensor_data(pc_labels);
    float * pc_probs_flat = THFloatTensor_data(pc_probs);
    float * pc_count_flat = THFloatTensor_data(pc_count);
    float * img_cls_loss_weights_flat = THFloatTensor_data(img_cls_loss_weights);
    float * im_labels_flat = THFloatTensor_data(im_labels);

    float * bottom_grad_flat = THFloatTensor_data(bottom_grad);

    int batch_size = THFloatTensor_size(pcl_probs, 0);
    int channels = THFloatTensor_size(pcl_probs, 1);

    float eps = 1e-5;

    for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < channels; c++) {
            bottom_grad_flat[i * channels + c] = 0;
            if (im_labels_flat[c] != 0) {
                if (c == 0) {
                    if (labels_flat[i] == 0) {
                        bottom_grad_flat[i * channels + c] = -cls_loss_weights_flat[i] 
                            / fmaxf(prob_data_flat[i * channels + c], eps);
                    }
                }
                else {
                    if (labels_flat[i] == c) {
                        int pc_index = gt_assignment_flat[i];
                        if (c != pc_labels_flat[pc_index]) {
                            printf("labels mismatch.\n");
                        }
                        bottom_grad_flat[i * channels + c] = -img_cls_loss_weights_flat[pc_index] 
                            / fmaxf(pc_count_flat[pc_index] * pc_probs_flat[pc_index], eps);
                    }
                }
            }
        }
    }
    return 1;
}