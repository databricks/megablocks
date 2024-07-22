# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import warnings
from typing import List, Optional

from packaging.version import Version, parse
from setuptools import find_packages, setup

# We require torch in setup.py to build cpp extensions "ahead of time"
# More info here: # https://pytorch.org/tutorials/advanced/cpp_extension.html
is_torch_installed = False
try:
    import torch
    from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension,
                                           CUDAExtension)
    is_torch_installed = True
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "No module named 'torch'. Torch is required to install this repo."
    ) from e

###############################################################################
# Requirements
###############################################################################

install_requires = [
    'torch>=2.3.0,<2.4',
    'triton>=2.1.0',
    # 'stanford-stk==0.7.0',
    'git+https://github.com/eitanturok/stk.git'
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

###############################################################################
# Extension Modules
###############################################################################


def package_files(prefix: str, directory: str, extension: str):
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, _, filenames) in os.walk(os.path.join(prefix, directory)):
        for filename in filenames:
            if filename.endswith(extension):
                paths.append(
                    os.path.relpath(os.path.join(path, filename), prefix))
    return paths


def get_cuda_bare_metal_version(cuda_dir: Optional[str]):
    raw_output = subprocess.check_output(
        [cuda_dir + '/bin/nvcc', '-V'],  # type: ignore
        universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index('release') + 1
    bare_metal_version = parse(output[release_idx].split(',')[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args: List[str]):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version('11.2'):
        return nvcc_extra_args + ['--threads', '4']
    return nvcc_extra_args


def check_cuda_torch_binary_vs_bare_metal(cuda_dir: Optional[str]):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)  # type: ignore

    print('\nCompiling cuda extensions with')
    print(raw_output + 'from ' + cuda_dir + '/bin\n')  # type: ignore

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            'Cuda extensions are being compiled with a version of Cuda that does '
            + 'not match the version used to compile Pytorch binaries.  ' +
            f'Pytorch binaries were compiled with Cuda {torch.version.cuda}'  # type: ignore
            + f'while the installed Cuda version is {bare_metal_version}. ' +
            'In some cases, a minor-version mismatch will not cause later errors:  '
            +
            'https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  ' +
            'You can try commenting out this check (at your own risk).')


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

cmdclass = {}
ext_modules = []

# Only install CUDA extensions if available
if 'cu' in torch.__version__ and CUDA_HOME is not None:
    check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]  # type: ignore
    if os.path.exists(
            os.path.join(torch_dir, 'include', 'ATen', 'CUDAGeneratorImpl.h')):
        generator_flag = ['-DOLD_GENERATOR_PATH']

    # generate code for the latest CUDA versions we can
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    cc_flag.append('-gencode')
    cc_flag.append('arch=compute_70,code=sm_70')
    cc_flag.append('-gencode')
    cc_flag.append('arch=compute_80,code=sm_80')
    cc_flag.append('-gencode')
    cc_flag.append('arch=compute_80,code=compute_80')
    if bare_metal_version >= Version('11.1'):
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_86,code=sm_86')
    if bare_metal_version >= Version('11.8'):
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_90,code=sm_90')

    ext_modules.append(
        CUDAExtension(
            name='megablocks_ops',
            sources=['csrc/ops.cu'],  # public API via pybind
            extra_compile_args={
                # 'cxx': ['-O3'] + generator_flag,
                'cxx': ['-fopenmp'],
                'nvcc':
                    append_nvcc_threads([
                        '-O3',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda',
                        '--use_fast_math',
                        # uncomment these if you hit errors resulting from
                        # PyTorch and CUDA independently implementing slightly
                        # different [b]f16 support
                        # EDIT: yes, these seem to be necessary for lion
                        # for some reason...
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                        '-U__CUDA_NO_BFLOAT162_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
                    ] + generator_flag + cc_flag),
            },
            include_dirs=[os.path.join(this_dir, 'csrc')],
        ))
    # Trevor's original ext_modules
    # ext_modules = [
    #     CUDAExtension(
    #         'megablocks_ops',
    #         ['csrc/ops.cu'],
    #         include_dirs=['csrc'],
    #         extra_compile_args={
    #             'cxx': ['-fopenmp'],
    #             'nvcc': nvcc_flags
    #         },
    #     )
    # ]

    cmdclass = {'build_ext': BuildExtension}
elif CUDA_HOME is None:
    warnings.warn(
        'Attempted to install CUDA extensions, but CUDA_HOME was None. ' +
        'Please install CUDA and ensure that the CUDA_HOME environment ' +
        'variable points to the installation location.')
else:
    warnings.warn('Warning: No CUDA devices; cuda code will not be compiled.')

# # default values
# nvcc_flags = ['--ptxas-options=-v', '--optimize=2']
# ext_modules = []
# cmdclass = {}

# # if torch is installed update default values
# if is_torch_installed:

#     if os.environ.get('TORCH_CUDA_ARCH_LIST'):
#         # Let PyTorch builder to choose device to target for.
#         device_capability = ''
#     else:
#         device_capability = torch.cuda.get_device_capability()
#         device_capability = f'{device_capability[0]}{device_capability[1]}'

#     if device_capability:
#         nvcc_flags.append(
#             f'--generate-code=arch=compute_{device_capability},code=sm_{device_capability}'
#         )

#     ext_modules = [
#         CUDAExtension(
#             'megablocks_ops',
#             ['csrc/ops.cu'],
#             include_dirs=['csrc'],
#             extra_compile_args={
#                 'cxx': ['-fopenmp'],
#                 'nvcc': nvcc_flags
#             },
#         )
#     ]

#     cmdclass = {'build_ext': BuildExtension}

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
