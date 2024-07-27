// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_bf16.h>

void gemm_kernel(float* a, float* b, float* c, int m, int n, int k);
void wgmma_kernel(void* a, void* b, void* c, int m, int n, int k);
