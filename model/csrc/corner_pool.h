#pragma once
#include "cpu/vision.h"

std::vector<at::Tensor> top_pool_forward(at::Tensor input);
std::vector<at::Tensor> top_pool_backward(at::Tensor input, at::Tensor grad_output);

std::vector<at::Tensor> bottom_pool_forward(at::Tensor input);
std::vector<at::Tensor> bottom_pool_backward(at::Tensor input, at::Tensor grad_output);

std::vector<at::Tensor> left_pool_forward(at::Tensor input);
std::vector<at::Tensor> left_pool_backward(at::Tensor input, at::Tensor grad_output);

std::vector<at::Tensor> right_pool_forward(at::Tensor input);
std::vector<at::Tensor> right_pool_backward(at::Tensor input, at::Tensor grad_output);