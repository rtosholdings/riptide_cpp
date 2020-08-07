
from distutils.core import setup, Extension, DEBUG
import numpy as np
import sys

sm_module = None
if sys.platform == 'linux':
    sm_module = Extension(
        'riptide_cpp', 
        sources = ['RipTide.cpp','Recycler.cpp','BasicMath.cpp','HashFunctions.cpp','HashLinear.cpp', 'MathThreads.cpp', 'Sort.cpp',
                    'Compare.cpp', 'MultiKey.cpp', 'MathWorker.cpp', 'BitCount.cpp', 'GroupBy.cpp','UnaryOps.cpp','Ema.cpp','Reduce.cpp',
                    'Merge.cpp','CRC32.cpp','Convert.cpp', 'SDSFile.cpp', 'SDSFilePython.cpp', 'TimeWindow.cpp', 'Compress.cpp', 'SharedMemory.cpp', 
                    'Bins.cpp', 'DateTime.cpp', 'strptime5.cpp', 'Hook.cpp', 'Array.cpp', 'TileRepeat.cpp',
                    'zstd/lib/compress/zstd_compress.c',
                    'zstd/lib/compress/zstdmt_compress.c',
                    'zstd/lib/compress/zstd_fast.c',
                    'zstd/lib/compress/zstd_double_fast.c',
                    'zstd/lib/compress/zstd_lazy.c',
                    'zstd/lib/compress/zstd_opt.c',
                    'zstd/lib/compress/zstd_ldm.c',
                    'zstd/lib/compress/fse_compress.c',
                    'zstd/lib/compress/huf_compress.c',
                    'zstd/lib/compress/hist.c',
                    'zstd/lib/decompress/zstd_decompress.c',
                    'zstd/lib/decompress/huf_decompress.c',
                    'zstd/lib/decompress/zstd_ddict.c',
                    'zstd/lib/decompress/zstd_decompress_block.c',
                    'zstd/lib/common/fse_decompress.c',
                    'zstd/lib/common/entropy_common.c',
                    'zstd/lib/common/zstd_common.c',
                    'zstd/lib/common/xxhash.c',
                    'zstd/lib/common/error_private.c',
                    'zstd/lib/common/pool.c'],
                  
        include_dirs = ['zstd/lib', 'zstd/lib/common', 'zstd/lib/compress', 'zstd/lib/decompress',],
        extra_compile_args = ['-mavx2', '-mbmi2', '-fpermissive','-Wno-unused-variable','-std=c++11','-pthread','-falign-functions=32','-falign-loops=32'],
        #libraries = [''],
        )

if sys.platform == 'win32':
    sm_module = Extension(
        'riptide_cpp', 
        sources = ['RipTide.cpp','Recycler.cpp','BasicMath.cpp','HashFunctions.cpp','HashLinear.cpp', 'MathThreads.cpp', 'Sort.cpp',
                    'Compare.cpp', 'MultiKey.cpp', 'MathWorker.cpp', 'BitCount.cpp', 'GroupBy.cpp','UnaryOps.cpp','Ema.cpp','Reduce.cpp',
                    'Merge.cpp','CRC32.cpp','Convert.cpp', 'SDSFile.cpp', 'SDSFilePython.cpp', 'TimeWindow.cpp', 'Compress.cpp', 'SharedMemory.cpp',
                    'Bins.cpp', 'DateTime.cpp', 'strptime5.cpp', 'Hook.cpp', 'Array.cpp', 'TileRepeat.cpp',
                    'zstd/lib/compress/zstd_compress.c',
                    'zstd/lib/compress/zstdmt_compress.c',
                    'zstd/lib/compress/zstd_fast.c',
                    'zstd/lib/compress/zstd_double_fast.c',
                    'zstd/lib/compress/zstd_lazy.c',
                    'zstd/lib/compress/zstd_opt.c',
                    'zstd/lib/compress/zstd_ldm.c',
                    'zstd/lib/compress/fse_compress.c',
                    'zstd/lib/compress/huf_compress.c',
                    'zstd/lib/compress/hist.c',
                    'zstd/lib/decompress/zstd_decompress.c',
                    'zstd/lib/decompress/huf_decompress.c',
                    'zstd/lib/decompress/zstd_ddict.c',
                    'zstd/lib/decompress/zstd_decompress_block.c',
                    'zstd/lib/common/fse_decompress.c',
                    'zstd/lib/common/entropy_common.c',
                    'zstd/lib/common/zstd_common.c',
                    'zstd/lib/common/xxhash.c',
                    'zstd/lib/common/error_private.c',
                    'zstd/lib/common/pool.c'],

        include_dirs = ['zstd/lib', 'zstd/lib/common', 'zstd/lib/compress', 'zstd/lib/decompress',]
        )

setup(name = 'riptide_cpp', 
    version = '1.0',
    description = 'Python Package with fast math util functions',
    author = 'SIG LLC',
    ext_modules = [sm_module],
    include_dirs=[np.get_include()]
    )