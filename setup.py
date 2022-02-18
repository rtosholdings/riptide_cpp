import os
import platform
import subprocess
from pprint import pprint
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from distutils.spawn import find_executable

package_name = 'riptide_cpp'

# CMake-driven setuptools build extension inspired by https://martinopilia.com/posts/2018/09/15/building-python-extension.html
class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class CMakeBuild(build_ext):
    def build_extensions(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        cmake_args = [
            "-DRIPTIDE_PYTHON_VER=" + platform.python_version()
            ]

        if platform.system() == 'Windows':
            cmake_args += [
                '-GVisual Studio 16 2019'
                ]
        elif platform.system() == 'Linux':
            if find_executable('ninja'):
                cmake_args += [
                    '-GNinja'
                    ]

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = 'Release'

            cmake_args += [
                '-DSETUPBUILD=ON',
                '-DCMAKE_BUILD_TYPE=%s' % cfg,
            ]

            if platform.system() == 'Windows':
                plat = ('x64' if platform.architecture()[0] == '64bit' else 'Win32')
                cmake_args += [
                    #'-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                ]

            print("Running cmake on " + ext.cmake_lists_dir + " into " + extdir + " from " + self.build_temp);
            pprint(cmake_args)

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config and build the extension
            subprocess.check_call(['cmake', ext.cmake_lists_dir,
                                    '--install-prefix', extdir,
                                    ] + cmake_args,
                                  cwd=self.build_temp)
            subprocess.check_call(['cmake',
                                    '--build', '.',
                                    '--config', cfg,
                                    '--target', 'install'
                                    ],
                                  cwd=self.build_temp)

# git-driven setuptools sdist extension inspired by scikit-build
class CMakeSdist(sdist):
    _MANIFEST_IN = 'MANIFEST.in'

    def run(self):
        try:
            self._generate_template()
            super().run()
        finally:
            os.remove(CMakeSdist._MANIFEST_IN)

    def _generate_template(self):
        with open(CMakeSdist._MANIFEST_IN, 'wb') as manifest_in_file:
            # Since Git < 2.11 does not support --recurse-submodules option, fallback to
            # regular listing.
            try:
                cmd_out = subprocess.check_output(['git', 'ls-files', '--recurse-submodules'])
            except subprocess.CalledProcessError:
                cmd_out = subprocess.check_output(['git', 'ls-files'])
            git_files = [git_file.strip() for git_file in cmd_out.split(b'\n')]
            manifest_text = b'\n'.join([
                b'include %s' % git_file.strip()
                for git_file in git_files
                if git_file
            ])
            manifest_text += b'\nexclude MANIFEST.in'
            manifest_in_file.write(manifest_text)


setup(
    name = package_name,
    use_scm_version = {
        'root': '.',
        'version_scheme': 'post-release',
    },
    setup_requires=['setuptools_scm'],
    description = 'Python Package with fast math util functions',
    author = 'RTOS Holdings',
    author_email = 'thomasdimitri@gmail.com',
    long_description= 'Python Package with fast math util functions',
    long_description_content_type= 'text/markdown',
    url="https://github.com/rtosholdings/riptide_cpp",
    ext_modules=[CMakeExtension(package_name)],
    cmdclass={
        'build_ext': CMakeBuild,
        'sdist': CMakeSdist,
        },
    install_requires=['numpy'],
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
