from setuptools import setup, find_packages
import os

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    torch_available = True
except ImportError:
    torch_available = False

# defaults
nvcc_flags = [
    "--ptxas-options=-v",
    "--optimize=2",
]
cmdclass = {}
ext_modules = []

# update defaults if torch is available
if torch_available:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        # Let PyTorch builder to choose device to target for.
        device_capability = ""
    else:
        device_capability = torch.cuda.get_device_capability()
        device_capability = f"{device_capability[0]}{device_capability[1]}"

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

    cmdclass = {"build_ext": BuildExtension}

install_requires=[
    "torch",
    "triton>=2.1.0",
    "stanford-stk==0.7.0",
]

extra_deps = {}

extra_deps["gg"] = [
    "grouped_gemm==0.1.4",
]

extra_deps["dev"] = [
    "absl-py",
    'pytest_codeblocks>=0.16.1,<0.17',
    'coverage[toml]==7.4.4',
    'pytest-cov>=4,<5',
    'pre-commit>=3.4.0,<4',
    'pytest>=7.2.1,<8',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

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
    cmdclass=cmdclass,
    install_requires=install_requires,
    extras_require=extra_deps,
)
