// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>
#include "pcl_losses_kernel.h"


#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void PCLLossesForward(const int nthreads, const float* bottom_data, 
    const float* labels, const float* cls_loss_weights, const float* pc_labels,
    const float* pc_probs, const float* img_cls_loss_weights, 
    const float* im_labels, const int batch_size, const int num_positive, float* top_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        top_data[index] = 0;
        if (im_labels[index] != 0) {
            if (index == 0) {
                for (int i = 0; i < batch_size; i++) {
                    if (labels[i] == 0) {
                        top_data[index] -= cls_loss_weights[i] * log(bottom_data[i * nthreads + index]);
                    }
                }
            }
            else {
                for (int i = 0; i < num_positive; i++) {
                    if (pc_labels[i] == index) {
                        top_data[index] -= img_cls_loss_weights[i] * log(pc_probs[i]);
                    }
                }
            }
        }
    }
}

int PCLLossesForwardLaucher(
    const float* bottom_data, const float* labels, const float* cls_loss_weights,
    const float* pc_labels, const float* pc_probs, const float* img_cls_loss_weights,
    const float* im_labels, const int batch_size, const int channels, 
    const int num_positive, float* top_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 4;
    cudaError_t err;

    PCLLossesForward<<<(channels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      channels, bottom_data, labels, cls_loss_weights, pc_labels, pc_probs, img_cls_loss_weights,
      im_labels, batch_size, num_positive, top_data);

    // dim3 blocks(DIVUP(output_size, kThreadsPerBlock),
    //             DIVUP(output_size, kThreadsPerBlock));
    // dim3 threads(kThreadsPerBlock);
    //
    // ROIPoolForward<<<blocks, threads, 0, stream>>>(
    //   output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
    //   pooled_width, bottom_rois, top_data, argmax_data);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void PCLLossesBackward(const int nthreads, const float* prob_data, 
    const float* labels, const float* cls_loss_weights, const float* gt_assignment,
    const float* pc_labels, const float* pc_probs, const float* pc_count,
    const float* img_cls_loss_weights, const float* im_labels, const int channels, 
    float* bottom_diff) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        int i = index / channels;
        int c = index % channels;
        bottom_diff[index] = 0;

        if (im_labels[c] != 0) {
            if (c == 0) {
                if (labels[i] == 0) {
                    bottom_diff[index] = -cls_loss_weights[i] / prob_data[index];
                }
            }
            else {
                if (labels[i] == c) {
                    int pc_index = gt_assignment[i];
                    if (c != pc_labels[pc_index]) {
                        printf("labels mismatch.\n");
                    }
                    bottom_diff[index] = -img_cls_loss_weights[pc_index]
                        / (pc_count[pc_index] * pc_probs[pc_index]);
                }
            }
        }
    }
}

int PCLLossesBackwardLaucher(const float* top_diff, const float* prob_data, 
    const float* labels, const float* cls_loss_weights, const float* gt_assignment,
    const float* pc_labels, const float* pc_probs, const float* pc_count,
    const float* img_cls_loss_weights, const float* im_labels, const int batch_size, 
    const int channels, float* bottom_diff, cudaStream_t stream)
{
    const int kThreadsPerBlock = 16;
    int output_size = batch_size * channels;
    cudaError_t err;

    PCLLossesBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, prob_data, labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count,
      img_cls_loss_weights, im_labels, channels, bottom_diff);

    // dim3 blocks(DIVUP(output_size, kThreadsPerBlock),
    //             DIVUP(output_size, kThreadsPerBlock));
    // dim3 threads(kThreadsPerBlock);
    //
    // ROIPoolBackward<<<blocks, threads, 0, stream>>>(
    //   output_size, top_diff, argmax_data, num_rois, spatial_scale, height, width, channels, pooled_height,
    //   pooled_width, bottom_diff, bottom_rois);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


// #ifdef __cplusplus
// }
// #endif
