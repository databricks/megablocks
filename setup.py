from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        "megablocks_ops",
        ["csrc/ops.cu", "csrc/binned_scatter.cu"],
        include_dirs = ["csrc"],
        extra_compile_args={
            "cxx": ["-fopenmp"],
            "nvcc": [
                "--ptxas-options=-v",
                "--optimize=2",
                "--generate-code=arch=compute_80,code=sm_80"
            ]
        })
]

setup(
    name="megablocks_ops",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
