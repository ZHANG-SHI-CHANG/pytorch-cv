#pragma once
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

at::Tensor ROIAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width);

at::Tensor ROIPool_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width);

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);


// deformable convolution
at::Tensor dcn_v2_cuda_forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &offset,
    const at::Tensor &mask,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group);


std::vector<at::Tensor> dcn_v2_cuda_backward(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &offset,
    const at::Tensor &mask,
    const at::Tensor &grad_output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int deformable_group);


std::tuple<at::Tensor, at::Tensor> dcn_v2_psroi_pooling_cuda_forward(
    const at::Tensor &input,
    const at::Tensor &bbox,
    const at::Tensor &trans,
    const int no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std);


std::tuple<at::Tensor, at::Tensor> dcn_v2_psroi_pooling_cuda_backward(
    const at::Tensor &out_grad,
    const at::Tensor &input,
    const at::Tensor &bbox,
    const at::Tensor &trans,
    const at::Tensor &top_count,
    const int no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std);


// criss-cross module
at::Tensor ca_forward_cuda(const at::Tensor &t, const at::Tensor &f);
std::tuple<at::Tensor, at::Tensor> ca_backward_cuda(
    const at::Tensor &dw, const at::Tensor &t, const at::Tensor &f);

at::Tensor ca_map_forward_cuda(const at::Tensor &weight, const at::Tensor &g);
std::tuple<at::Tensor, at::Tensor> ca_map_backward_cuda(
    const at::Tensor &dout, const at::Tensor &weight, const at::Tensor &g);