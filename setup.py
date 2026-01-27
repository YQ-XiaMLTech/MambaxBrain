from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
    ],
}

setup(
    name="selective_scan_cuda",
    ext_modules=[
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_fwd_fp32.cu",
                "csrc/selective_scan/selective_scan_fwd_fp16.cu",
                "csrc/selective_scan/selective_scan_fwd_bf16.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
            ],
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
