// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_bf16.h>

void wgmma_kernel(void* a, void* b, void* c, int m, int n, int k);
