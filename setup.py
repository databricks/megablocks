# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

"""MegaBlocks package setup."""

import os
import warnings
from typing import Any, Dict, Mapping

from setuptools import find_packages, setup

# We require torch in setup.py to build cpp extensions "ahead of time"
# More info here: # https://pytorch.org/tutorials/advanced/cpp_extension.html
try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'torch'. `torch` is required to install `MegaBlocks`.",) from e

_PACKAGE_NAME = 'megablocks'
_PACKAGE_DIR = 'megablocks'
_REPO_REAL_PATH = os.path.dirname(os.path.realpath(__file__))
_PACKAGE_REAL_PATH = os.path.join(_REPO_REAL_PATH, _PACKAGE_DIR)

# Read the package version
# We can't use `.__version__` from the library since it's not installed yet
version_path = os.path.join(_PACKAGE_REAL_PATH, '_version.py')
with open(version_path, encoding='utf-8') as f:
    version_globals: Dict[str, Any] = {}
    version_locals: Mapping[str, object] = {}
    content = f.read()
    exec(content, version_globals, version_locals)
    repo_version = version_locals['__version__']

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        long_description = long_description[:start] + \
            long_description[end + len(end_tag):]

classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'License :: OSI Approved :: BSD License',
    'Operating System :: Unix',
]

install_requires = [
    'numpy>=1.21.5,<2.1.0',
    'packaging>=21.3.0,<24.2',
    'torch>=2.6.0,<2.6.1',
    'triton>=3.2.0,<3.3.0',
    'stanford-stk==0.7.1',
]

extra_deps = {}

extra_deps['gg'] = [
    'grouped_gemm==0.1.6',
]

extra_deps['dev'] = [
    'absl-py',  # TODO: delete when finish removing all absl tests
    'coverage[toml]==7.4.4',
    'pytest_codeblocks>=0.16.1,<0.17',
    'pytest-cov>=4,<5',
    'pytest>=7.2.1,<8',
    'pre-commit>=3.4.0,<4',
]

extra_deps['testing'] = [
    'mosaicml>=0.24.1',
]

extra_deps['all'] = list({dep for key, deps in extra_deps.items() for dep in deps if key not in {'testing'}})

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
        nvcc_flags.append(f'--generate-code=arch=compute_{device_capability},code=sm_{device_capability}',)

    ext_modules = [
        CUDAExtension(
            'megablocks_ops',
            ['csrc/ops.cu'],
            include_dirs=['csrc'],
            extra_compile_args={
                'cxx': ['-fopenmp'],
                'nvcc': nvcc_flags,
            },
        ),
    ]
elif CUDA_HOME is None:
    warnings.warn(
        'Attempted to install CUDA extensions, but CUDA_HOME was None. ' +
        'Please install CUDA and ensure that the CUDA_HOME environment ' +
        'variable points to the installation location.',
    )
else:
    warnings.warn('Warning: No CUDA devices; cuda code will not be compiled.')

setup(
    name=_PACKAGE_NAME,
    version=repo_version,
    author='Trevor Gale',
    author_email='tgale@stanford.edu',
    description='MegaBlocks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/databricks/megablocks',
    classifiers=classifiers,
    packages=find_packages(exclude=['tests*', 'third_party*', 'yamls*', 'exp*', '.github*']),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.9',
    package_data={_PACKAGE_NAME: ['py.typed']},
)
