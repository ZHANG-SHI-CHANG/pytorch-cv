#include "cpu/vision.h"

at::Tensor ca_forward_cpu(const at::Tensor &t, const at::Tensor &f) {
    AT_ERROR("Not implement on cpu");
}

std::tuple<at::Tensor, at::Tensor> ca_backward_cpu(
    const at::Tensor &dw, const at::Tensor &t, const at::Tensor &f) {
    AT_ERROR("Not implement on cpu");
}


at::Tensor ca_map_forward_cpu(const at::Tensor &weight, const at::Tensor &g) {
    AT_ERROR("Not implement on cpu");
}

std::tuple<at::Tensor, at::Tensor> ca_map_backward_cpu(
    const at::Tensor &dout, const at::Tensor &weight, const at::Tensor &g) {
    AT_ERROR("Not implement on cpu");
}