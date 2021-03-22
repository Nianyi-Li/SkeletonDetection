import sys
sys.path
sys.path.append('/home/user/.local/lib/python3.8/site-packages')

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules=[
    CUDAExtension('dapalib', [
        'association.cpp',
        'gpu/nmsBase.cu',
        'gpu/bodyPartConnectorBase.cu',
        'gpu/cuda_cal.cu',
        ], 
        include_dirs=['/usr/local/cuda-11.2/include', '/usr/local/lib', '/home/user/.local/lib/python3.8/site-packages', '/usr/local/cuda-11.2/bin/nvcc'] ,   # '/usr/include/eigen3'
    ),         
]

setup(
    name='dapalib',
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
