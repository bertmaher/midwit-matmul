# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="wgmma",
    ext_modules=[
        CUDAExtension(
            "wgmma",
            [
                "wgmma.cpp",
                "wgmma_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
