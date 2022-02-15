from skbuild import setup
import platform

package_name = 'riptide_cpp'

cmake_args = [
     "-DRIPTIDE_PYTHON_VER=" + platform.python_version()
     ]

if platform.system() == 'Windows':
    cmake_args += [
        '-GVisual Studio 16 2019'
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
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
    ]
    )
