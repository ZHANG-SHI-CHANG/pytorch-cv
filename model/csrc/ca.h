#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor ca_forward(const at::Tensor &t, const at::Tensor &f) {
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
      return ca_forward_cuda(t, f);
#else
      AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> ca_backward(
    const at::Tensor &dw, const at::Tensor &t, const at::Tensor &f) {
    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
      return ca_backward_cuda(dw, t, f);
#else
      AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor ca_map_forward(const at::Tensor &weight, const at::Tensor &g) {
    if (g.device().is_cuda()) {
#ifdef WITH_CUDA
      return ca_map_forward_cuda(weight, g);
#else
      AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> ca_map_backward(
    const at::Tensor &dout, const at::Tensor &weight, const at::Tensor &g) {
    if (g.device().is_cuda()) {
#ifdef WITH_CUDA
      return ca_map_backward_cuda(dout, weight, g);
#else
      AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}