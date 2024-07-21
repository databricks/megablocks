# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

import os

from setuptools import find_packages, setup

# We require torch in setup.py to build cpp extensions "ahead of time"
# More info here: # https://pytorch.org/tutorials/advanced/cpp_extension.html
is_torch_installed = False
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    is_torch_installed = True
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "No module named 'torch'. Torch is required to install this repo."
    ) from e

# default values
nvcc_flags = ['--ptxas-options=-v', '--optimize=2']
ext_modules = []
cmdclass = {}

# if torch is installed update default values
if is_torch_installed:

    if os.environ.get('TORCH_CUDA_ARCH_LIST'):
        # Let PyTorch builder to choose device to target for.
        device_capability = ''
    else:
        device_capability = torch.cuda.get_device_capability()
        device_capability = f'{device_capability[0]}{device_capability[1]}'

    if device_capability:
        nvcc_flags.append(
            f'--generate-code=arch=compute_{device_capability},code=sm_{device_capability}'
        )

    ext_modules = [
        CUDAExtension(
            'megablocks_ops',
            ['csrc/ops.cu'],
            include_dirs=['csrc'],
            extra_compile_args={
                'cxx': ['-fopenmp'],
                'nvcc': nvcc_flags
            },
        )
    ]

    cmdclass = {'build_ext': BuildExtension}

install_requires = [
    'triton>=2.1.0',
    'stanford-stk==0.7.0',
]

extra_deps = {}

extra_deps['gg'] = [
    'grouped_gemm==0.1.4',
]

extra_deps['dev'] = [
    'absl-py',
    'pytest_codeblocks>=0.16.1,<0.17',
    'coverage[toml]==7.4.4',
    'pytest-cov>=4,<5',
    'pre-commit>=3.4.0,<4',
    'pytest>=7.2.1,<8',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

# convert README to long description on PyPI, optionally skipping certain
# marked sections if present (e.g., for coverage / code quality badges)
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
while True:
    start_tag = '<!-- LONG_DESCRIPTION_SKIP_START -->'
    end_tag = '<!-- LONG_DESCRIPTION_SKIP_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'Skipped section starts and ends imbalanced'
        break
    else:
        assert end != -1, 'Skipped section starts and ends imbalanced'
        long_description = long_description[:start] + long_description[
            end + len(end_tag):]

setup(
    name='megablocks',
    version='0.5.1',
    author='Trevor Gale',
    author_email='tgale@stanford.edu',
    description='MegaBlocks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stanford-futuredata/megablocks',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
    ],
    packages=find_packages(exclude=['tests*']),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    extras_require=extra_deps,
)
