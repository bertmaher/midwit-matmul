// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "wgmma_kernel.h"

#include "ATen/ATen.h" // @manual
#include "torch/extension.h" // @manual

at::Tensor wgmma(at::Tensor a, at::Tensor b) {
  // a (m x k), b (k x n)
  auto c = a.new_empty({a.size(0), b.size(1)});
  wgmma_kernel(
      a.data_ptr(),
      b.data_ptr(),
      c.data_ptr(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

TORCH_LIBRARY(wgmma, m) {
  m.def("wgmma", &wgmma);
}
