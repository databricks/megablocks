from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


_dc = torch.cuda.get_device_capability()
_dc = f"{_dc[0]}{_dc[1]}"
ext_modules = [
    CUDAExtension(
        "megablocks_ops",
        ["csrc/ops.cu"],
        include_dirs = ["csrc"],
        extra_compile_args={
            "cxx": ["-fopenmp"],
            "nvcc": [
                "--ptxas-options=-v",
                "--optimize=2",
                f"--generate-code=arch=compute_{_dc},code=sm_{_dc}"
            ]
        })
]

install_requires=[
    "triton==2.1.0",
    "stanford-stk>=0.0.6",
]

extra_deps = {}

extra_deps["gg"] = [
    "grouped_gemm",
]

extra_deps["quant"] = [
    "mosaicml-turbo==0.0.4",
]

extra_deps["dev"] = [
    "absl-py",
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name="megablocks",
    version="0.5.0",
    author="Trevor Gale",
    author_email="tgale@stanford.edu",
    description="MegaBlocks",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/stanford-futuredata/megablocks",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=install_requires,
    extras_require=extra_deps,
)
