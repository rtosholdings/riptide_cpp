import sys

from skbuild import setup

package_name = 'riptide_cpp'

setup(
    name = package_name,
    use_scm_version = {
        'root': '.',
        'version_scheme': 'post-release',
    },
    cmake_args = ['-DBENCHMARK_ENABLE_GTEST_TESTS=off','-DRIPTIDE_PYTHON_VER=3.9.9'],
    cmake_install_dir='src',
    setup_requires=['setuptools_scm'],
    description = 'Python Package with fast math util functions',
    author = 'RTOS Holdings',
    author_email = 'thomasdimitri@gmail.com',
    long_description= 'Python Package with fast math util functions',
    long_description_content_type= 'text/markdown',
    url="https://github.com/rtosholdings/riptide_cpp",
    packages=[package_name],
    install_requires=['numpy'],
    package_dir={'riptide_cpp' : 'src'},
    classifiers=[
         "Development Status :: 4 - Beta",
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
    ]
    )
