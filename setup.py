from setuptools import setup, find_packages
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

install_requires=[
    'stanford-stk @ git+https://github.com/vchiley/stk.git@setup_deps',
]

extra_deps = {}

extra_deps['dev'] = [
    'absl-py',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name="megablocks",
    version="0.0.3",
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
