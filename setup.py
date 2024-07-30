import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if os.environ.get("TORCH_CUDA_ARCH_LIST"):
    # Let PyTorch builder to choose device to target for.
    device_capability = ""
else:
    device_capability = torch.cuda.get_device_capability()
    device_capability = f"{device_capability[0]}{device_capability[1]}"

nvcc_flags = [
    "--ptxas-options=-v",
    "--optimize=2",
]
if device_capability:
    nvcc_flags.append(
        f"--generate-code=arch=compute_{device_capability},code=sm_{device_capability}"
    )

ext_modules = [
    CUDAExtension(
        "megablocks_ops",
        ["csrc/ops.cu"],
        include_dirs=["csrc"],
        extra_compile_args={"cxx": ["-fopenmp"], "nvcc": nvcc_flags},
    )
]

install_requires = [
    'numpy>=1.21.5,<2.1.0',
    'torch>=2.3.0,<2.4',
    'triton>=2.1.0',
    'stanford-stk @ git+https://git@github.com/stanford-futuredata/stk.git@a1ddf98466730b88a2988860a9d8000fd1833301',
    'packaging>=21.3.0,<24.2',
]

extra_deps = {}

extra_deps["gg"] = [
    'grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@66c7195e35e8c4f22fa6a014037ef511bfa397cb',
]

extra_deps['dev'] = [
    'absl-py',
    'coverage[toml]==7.4.4',
    'pytest_codeblocks>=0.16.1,<0.17',
    'pytest-cov>=4,<5',
    'pytest>=7.2.1,<8',
    'pre-commit>=3.4.0,<4',
]

extra_deps['testing'] = [
    'mosaicml>=0.22.0',
]

extra_deps['all'] = list({
    dep for key, deps in extra_deps.items() for dep in deps
    if key not in {'testing'}
})

setup(
    name="megablocks",
    version="0.5.1",
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
