from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension("cudss_wrapp",
              sources=["ccuDSS.pyx"],
              include_dirs=['/usr/local/cuda-12.2/include',"/home/mdossena/miniconda3/envs/testCUDSS/lib/python3.13/site-packages/nvidia/cu12/include"],
              libraries=['cudart'],
              extra_link_args = ["-l:libcudss.so.0"],
              library_dirs=['/usr/local/cuda-12.2/lib64',"/home/mdossena/miniconda3/envs/testCUDSS/lib/python3.13/site-packages/nvidia/cu12/lib"],
              language="c++",
              extra_compile_args=["-std=c++11"],
              runtime_library_dirs = ["/home/mdossena/miniconda3/envs/testCUDSS/lib/python3.13/site-packages/nvidia/cu12/lib","/usr/local/cuda-12.2/lib64",'.'],
              )
]

setup(name="ccuDSS",
      ext_modules=cythonize(ext_modules))