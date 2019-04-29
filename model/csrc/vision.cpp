#include "roialign.h"
#include "roipool.h"
#include "dcn_v2.h"
#include "nms.h"
#include "ca.h"
//#include "corner_pool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  // deformable convolution
  m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
  m.def("dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward, "dcn_v2_psroi_pooling_forward");
  m.def("dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward, "dcn_v2_psroi_pooling_backward");
  // criss-cross module
  m.def("ca_forward", &ca_forward, "ca forward");
  m.def("ca_backward", &ca_backward, "ca backward");
  m.def("ca_map_forward", &ca_map_forward, "ca map forward");
  m.def("ca_map_backward", &ca_map_backward, "ca map backward");

//  // corner pool
//  m.def("top_pool_forward", &top_pool_forward, "top pool forward");
//  m.def("top_pool_backward", &top_pool_backward, "top pool backward");
//  m.def("bottom_pool_forward", &bottom_pool_forward, "bottom pool forward");
//  m.def("bottom_pool_backward", &bottom_pool_backward, "bottom pool backward");
//  m.def("left_pool_forward", &left_pool_forward, "left pool forward");
//  m.def("left_pool_backward", &left_pool_backward, "left pool backward");
//  m.def("right_pool_forward", &right_pool_forward, "right pool forward");
//  m.def("right_pool_backward", &right_pool_backward, "right pool backward");
}