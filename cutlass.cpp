// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "cutlass_kernel.h"

#include "ATen/ATen.h" // @manual
#include "torch/extension.h" // @manual

at::Tensor gemm(at::Tensor a, at::Tensor b) {
  auto c = a.new_empty({a.size(0), b.size(1)});
  gemm_kernel(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

at::Tensor wgmma(at::Tensor a, at::Tensor b) {
  // a row-major (m x k), b col-major (n x k)
  auto c = a.new_zeros({a.size(0), b.size(0)});
  wgmma_kernel(
      a.data_ptr(),
      b.data_ptr(),
      c.data_ptr(),
      a.size(0),
      b.size(0),
      a.size(1));
  return c;
}

TORCH_LIBRARY(cutlass, m) {
  m.def("gemm", &gemm);
  m.def("wgmma", &wgmma);
}
