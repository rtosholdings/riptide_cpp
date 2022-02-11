from skbuild import setup
import platform

package_name = 'riptide_cpp'

riptide_python_ver_pieces = [ "-DRIPTIDE_PYTHON_VER=", platform.python_version()]
riptide_python_ver = "".join(riptide_python_ver_pieces)

cmake_args = [
    '-DBENCHMARK_ENABLE_GTEST_TESTS=off',
     riptide_python_ver
     ]

if platform.system() == 'Windows':
    cmake_args += [
        '-GVisual Studio 16 2019',
        '-Tv142'
        ]
elif platform.system() == 'Linux':
    cmake_args += [
        '-DCMAKE_C_COMPILER=gcc',
        '-DCMAKE_CXX_COMPILER=g++'
        ]

setup(
    name = package_name,
    use_scm_version = {
        'root': '.',
        'version_scheme': 'post-release',
    },
    cmake_args = cmake_args,
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
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
    ]
    )
