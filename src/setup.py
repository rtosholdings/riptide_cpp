import setuptools
import numpy as np
import sys

sm_module = None
if sys.platform != 'win32':
    sm_module = setuptools.Extension(
        'riptide_cpp', 
        sources = ['RipTide.cpp','Recycler.cpp','BasicMath.cpp','HashFunctions.cpp','HashLinear.cpp', 'MathThreads.cpp', 'Sort.cpp',
                    'Compare.cpp', 'MultiKey.cpp', 'MathWorker.cpp', 'BitCount.cpp', 'GroupBy.cpp','UnaryOps.cpp','Ema.cpp','Reduce.cpp',
                    'Merge.cpp','CRC32.cpp','Convert.cpp', 'SDSFile.cpp', 'SDSFilePython.cpp', 'TimeWindow.cpp', 'Compress.cpp', 'SharedMemory.cpp', 
                    'Bins.cpp', 'DateTime.cpp', 'strptime5.cpp', 'Hook.cpp', 'Array.cpp', 'TileRepeat.cpp',
                    'zstd/compress/fse_compress.c',
                    'zstd/compress/hist.c',
                    'zstd/compress/huf_compress.c',
                    'zstd/compress/zstdmt_compress.c',
                    'zstd/compress/zstd_compress.c',
                    'zstd/compress/zstd_compress_literals.c',
                    'zstd/compress/zstd_compress_sequences.c',
                    'zstd/compress/zstd_compress_superblock.c',
                    'zstd/compress/zstd_double_fast.c',
                    'zstd/compress/zstd_fast.c',
                    'zstd/compress/zstd_lazy.c',
                    'zstd/compress/zstd_ldm.c',
                    'zstd/compress/zstd_opt.c',
                    'zstd/decompress/zstd_decompress.c',
                    'zstd/decompress/huf_decompress.c',
                    'zstd/decompress/zstd_ddict.c',
                    'zstd/decompress/zstd_decompress_block.c',
                    'zstd/common/fse_decompress.c',
                    'zstd/common/entropy_common.c',
                    'zstd/common/zstd_common.c',
                    'zstd/common/xxhash.c',
                    'zstd/common/error_private.c',
                    'zstd/common/pool.c'],
                  
        include_dirs = ['zstd', 'zstd/common', 'zstd/compress', 'zstd/decompress',],
        extra_compile_args = ['-mavx2', '-mbmi2', '-fpermissive','-Wno-unused-variable','-std=c++11','-pthread','-falign-functions=32','-falign-loops=32'],
        #libraries = [''],
        )

if sys.platform == 'win32':
    sm_module = setuptools.Extension(
        'riptide_cpp', 
        sources = ['RipTide.cpp','Recycler.cpp','BasicMath.cpp','HashFunctions.cpp','HashLinear.cpp', 'MathThreads.cpp', 'Sort.cpp',
                    'Compare.cpp', 'MultiKey.cpp', 'MathWorker.cpp', 'BitCount.cpp', 'GroupBy.cpp','UnaryOps.cpp','Ema.cpp','Reduce.cpp',
                    'Merge.cpp','CRC32.cpp','Convert.cpp', 'SDSFile.cpp', 'SDSFilePython.cpp', 'TimeWindow.cpp', 'Compress.cpp', 'SharedMemory.cpp',
                    'Bins.cpp', 'DateTime.cpp', 'strptime5.cpp', 'Hook.cpp', 'Array.cpp', 'TileRepeat.cpp',
                    'zstd/compress/fse_compress.c',
                    'zstd/compress/hist.c',
                    'zstd/compress/huf_compress.c',
                    'zstd/compress/zstdmt_compress.c',
                    'zstd/compress/zstd_compress.c',
                    'zstd/compress/zstd_compress_literals.c',
                    'zstd/compress/zstd_compress_sequences.c',
                    'zstd/compress/zstd_compress_superblock.c',
                    'zstd/compress/zstd_double_fast.c',
                    'zstd/compress/zstd_fast.c',
                    'zstd/compress/zstd_lazy.c',
                    'zstd/compress/zstd_ldm.c',
                    'zstd/compress/zstd_opt.c',
                    'zstd/decompress/zstd_decompress.c',
                    'zstd/decompress/huf_decompress.c',
                    'zstd/decompress/zstd_ddict.c',
                    'zstd/decompress/zstd_decompress_block.c',
                    'zstd/common/fse_decompress.c',
                    'zstd/common/entropy_common.c',
                    'zstd/common/zstd_common.c',
                    'zstd/common/xxhash.c',
                    'zstd/common/error_private.c',
                    'zstd/common/pool.c'],

        include_dirs = ['zstd', 'zstd/common', 'zstd/compress', 'zstd/decompress',]
        )

setuptools.setup(
    name = 'riptide_cpp', 
    version = '1.3',
    description = 'Python Package with fast math util functions',
    author = 'RTOS Holdings',
    author_email = 'nobody@rtosholdings.com',
    ext_modules = [sm_module],
    long_description= 'Python Package with fast math util functions',
    long_description_content_type= 'text/markdown',
    url="https://github.com/rtosholdings/riptide_cpp",
    #packages=setuptools.find_packages(),
    packages=['riptide_cpp'],
    package_dir={'riptide_cpp' : '.'},
    classifiers=[
         "Development Status :: 4 - Beta",
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
    ],
    include_dirs=[np.get_include()]
    )
