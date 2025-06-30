from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='swin_window_process',
    ext_modules=[
        CUDAExtension('swin_window_process', [
            'swin_window_process.cpp',
            'swin_window_process_kernel.cu',
        ],
        extra_compile_args={
            'nvcc': [
                '-gencode', 'arch=compute_80,code=sm_80',  # A100
            ]
        })
    ],
    cmdclass={'build_ext': BuildExtension})
