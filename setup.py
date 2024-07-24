# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

"""MegaBlocks package setup."""

import os
import warnings

from setuptools import find_packages, setup

# We require torch in setup.py to build cpp extensions "ahead of time"
# More info here: # https://pytorch.org/tutorials/advanced/cpp_extension.html
try:
    import torch
    from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension,
                                           CUDAExtension,)
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "No module named 'torch'. `torch` is required to install this repo."
    ) from e


_PACKAGE_NAME = 'megablocks'
_PACKAGE_DIR = 'megablocks'
_REPO_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
_PACKAGE_REAL_PATH = os.path.join(_REPO_REAL_PATH, _PACKAGE_DIR)

# Read the package version
# We can't use `.__version__` from the library since it's not installed yet
with open(os.path.join(_PACKAGE_REAL_PATH, '_version.py'), encoding='utf-8') as f:
    version_globals = {}
    version_locals = {}
    exec(f.read(), version_globals, version_locals)
    repo_version = version_locals['__version__']

install_requires = [
    'numpy>=1.21.5,<2.1.0',
    'torch>=2.3.0,<2.4',
    'triton>=2.1.0',
    # 'stanford-stk==0.7.0',
    'stanford-stk @ git+https://git@github.com/eitanturok/stk.git'
]

extra_deps = {}

extra_deps['gg'] = [
    'grouped_gemm==0.1.4',
]

extra_deps['dev'] = [
    'absl-py',
    'coverage[toml]==7.4.4',
    'pytest_codeblocks>=0.16.1,<0.17',
    'pytest-cov>=4,<5',
    'pytest>=7.2.1,<8',
    'pre-commit>=3.4.0,<4',
]

extra_deps['all'] = list(
    set(dep for deps in extra_deps.values() for dep in deps))


cmdclass = {}
ext_modules = []

# Only install CUDA extensions if available
if 'cu' in torch.__version__ and CUDA_HOME is not None:

    cmdclass = {'build_ext': BuildExtension}
    nvcc_flags = ['--ptxas-options=-v', '--optimize=2']

    if os.environ.get('TORCH_CUDA_ARCH_LIST'):
        # Let PyTorch builder to choose device to target for.
        device_capability = ''
    else:
        device_capability_tuple = torch.cuda.get_device_capability()
        device_capability = f'{device_capability_tuple[0]}{device_capability_tuple[1]}'

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
elif CUDA_HOME is None:
    warnings.warn(
        'Attempted to install CUDA extensions, but CUDA_HOME was None. ' +
        'Please install CUDA and ensure that the CUDA_HOME environment ' +
        'variable points to the installation location.')
else:
    warnings.warn('Warning: No CUDA devices; cuda code will not be compiled.')


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
    name=_PACKAGE_NAME,
    version=repo_version,
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
