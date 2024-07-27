# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import math
import os

import torch

from torch.profiler import profile
from triton.testing import do_bench  # @manual

from triton_kernel import matmul as triton_matmul

def to_core(x):
    m, n = x.shape
    return x.view(m // 8, 8, n // 8, 8).transpose(1, 2).contiguous().view(m, n)

def main():
    torch.set_printoptions(profile="full", sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    try:
        torch.ops.load_library("cutlass.so")
    except Exception:
        torch.ops.load_library("//scripts/bertrand/tf32_gemm:cutlass")

    m, n, k = 64, 256, 16

    a = torch.eye(m, k, device="cuda").to(torch.bfloat16)
    b = torch.arange(n * k, device="cuda").reshape(n, k).bfloat16()

    #c_wgmma = torch.ops.cutlass.wgmma(a, b)
    c_torch = a @ b.T
    c_wgmma = torch.ops.cutlass.wgmma(to_core(a), to_core(b))

    #print(c_wgmma)
    #print(c_torch)

    print("allclose?", torch.allclose(c_wgmma, c_torch, atol=1e-3, rtol=1e-3))

    tflops = 2 * m * n * k / 1e9

    print("torch (cublas):", tflops / do_bench(lambda: torch.mm(a, b.T)))
    print("wgmma:", tflops / do_bench(lambda: torch.ops.cutlass.wgmma(a, b)))

    if args.profile:
        with profile() as p:
            torch.zeros(1)
            for _ in range(3):
                torch.mm(a, b)
            torch.cuda.synchronize()
            for _ in range(3):
                torch.ops.cutlass.gemm(a, b)
            torch.cuda.synchronize()
            for _ in range(3):
                triton_matmul(a, b)
            torch.cuda.synchronize()
        p.export_chrome_trace("gemm.json.gz")
        os.system(
            "manifold put --overwrite --threads 20 gemm.json.gz gpu_traces/tree/traces/bertrand/gemm.json.gz"
        )
        print(
            "https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/bertrand/gemm.json.gz&bucket=gpu_traces"
        )


if __name__ == "__main__":
    main()
