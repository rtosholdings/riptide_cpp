import setuptools
try:
    import numpy as np
except:
    # readthedocs does not install numpy
    # another was is to use pip.__path__ and remove the pip and replace with numpy/core/include
    import pip
    package='riptide_cpp'
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])
    import numpy as np
import sys

package_name = 'riptide_cpp'
rc_module = None
sources_cpp=['RipTide.cpp','Recycler.cpp','BasicMath.cpp','HashFunctions.cpp','HashLinear.cpp', 'MathThreads.cpp', 'Sort.cpp',
            'Compare.cpp', 'MultiKey.cpp', 'MathWorker.cpp', 'BitCount.cpp', 'GroupBy.cpp','UnaryOps.cpp','Ema.cpp','Reduce.cpp',
            'Merge.cpp','CRC32.cpp','Convert.cpp', 'SDSFile.cpp', 'SDSFilePython.cpp', 'TimeWindow.cpp', 'Compress.cpp', 'SharedMemory.cpp',
            'Bins.cpp', 'DateTime.cpp', 'strptime5.cpp', 'Hook.cpp', 'Array.cpp', 'TileRepeat.cpp']

sources_zstd=['zstd/compress/fse_compress.c',
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
            'zstd/common/pool.c']

if sys.platform == 'linux':
    rc_module = setuptools.Extension(
        package_name,
        sources = sources_cpp + sources_zstd,

        include_dirs = ['zstd', 'zstd/common', 'zstd/compress', 'zstd/decompress',],
        extra_compile_args = ['-mavx2', '-mbmi2', '-fpermissive','-Wno-unused-variable','-std=c++11','-pthread','-falign-functions=32','-falign-loops=32'],
        extra_link_args = ['-lrt'],
        #libraries = [''],
        )

if sys.platform == 'darwin':
    rc_module = setuptools.Extension(
        package_name,
        sources = sources_cpp,
        include_dirs = ['zstd'],
        extra_link_args = ['lib/libzstd.a'],
        #libraries = ['libzstd.a'],
        #library_dirs = ['lib'],
        extra_compile_args = ['-mavx2', '-mbmi2', '-fpermissive','-Wno-unused-variable','-std=c++11','-pthread','-falign-functions=32'],
        )


if sys.platform == 'win32':
    rc_module = setuptools.Extension(
        package_name,
        sources = sources_cpp + sources_zstd,
        include_dirs = ['zstd', 'zstd/common', 'zstd/compress', 'zstd/decompress',],
        #extra_compile_args = ['/MT /Ox /Ob2 /Oi /Ot'],
        # For MSVC windows compiler 2019 it has the new __CxxFrameHandler4 which is found in vcrntime140_1.dll which is not on all systems
        # We use /dsFH4- to disable this frame handler
        extra_compile_args = ['/Ox','/Ob2','/Oi','/Ot','/d2FH4-'],
        )

setuptools.setup(
    name = package_name,
    use_scm_version = {
        'root': '..',
        'version_scheme': 'post-release',
    },
    setup_requires=['setuptools_scm'],
    description = 'Python Package with fast math util functions',
    author = 'RTOS Holdings',
    author_email = 'thomasdimitri@gmail.com',
    ext_modules = [rc_module],
    long_description= 'Python Package with fast math util functions',
    long_description_content_type= 'text/markdown',
    url="https://github.com/rtosholdings/riptide_cpp",
    #packages=setuptools.find_packages(),
    packages=[package_name],
    install_requires=['numpy'],
    package_dir={package_name : '.'},
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
