import os
import glob
from setuptools import setup

import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

requirements = ["torch"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            '._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
        )
    ]

    return ext_modules


setup(
    name="pytorch-cv",
    version="0.1",
    author="fmassa",
    description="pytorch cpp extension for cv",
    # packages=find_packages(exclude=("configs", "examples", "test",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
