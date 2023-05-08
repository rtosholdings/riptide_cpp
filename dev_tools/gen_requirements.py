# Generates requirements for riptide_cpp

import argparse
import platform
import sys


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_python(major: int, minor: int) -> bool:
    ver = sys.version_info
    return ver.major == major and ver.minor == minor


_ABSEIL_REQ = "abseil-cpp==20220623.*"
_BENCHMARK_REQ = "benchmark>=1.7,<1.8"
_CMAKE_REQ = "cmake>=3.18"
_NUMPY_REQ = "numpy>=1.22,<1.24"
_TBB_VER = "==2021.6.*"
_TBB_REQ = "tbb" + _TBB_VER
_TBB_DEVEL_REQ = "tbb-devel" + _TBB_VER
_ZSTD_REQ = "zstd>=1.5.2,<1.6"

# Host toolchain requirements to build riptide_cpp.
toolchain_reqs = []
if is_linux():
    toolchain_reqs += [
        "binutils",
        "binutils_linux-64",
        "gcc==8.*",
        "gxx==8.*",
        "ninja",
    ]

# Conda-build requirements.
# Most everything else will be specified in meta.yaml.
conda_reqs = [
    "conda-build",
    "setuptools_scm",  # Needed to construct BUILD_VERSION for meta.yaml
] + toolchain_reqs

# PyPI setup build requirements.
# Most everything else *should* be in pyproject.toml, but since we run
# setup.py directly we need to set up build environment manually here.
pypi_reqs = [
    _ABSEIL_REQ,  # PyPI package doesn't exist
    _BENCHMARK_REQ,  # PyPI package doesn't exist
    _CMAKE_REQ,
    _TBB_DEVEL_REQ,  # needed because PyPI tbb-devel pkg doesn't contain CMake files yet
    "wheel",
    _ZSTD_REQ,  # PyPI package doesn't exist
] + toolchain_reqs

# Native CMake build requirements.
native_reqs = [
    _ABSEIL_REQ,
    _BENCHMARK_REQ,
    _CMAKE_REQ,
    _NUMPY_REQ,
    _TBB_DEVEL_REQ,
    _ZSTD_REQ,
] + toolchain_reqs

# Extras build requirements (same as native build requirements).
extras_reqs = native_reqs

# Runtime requirements for riptide_cpp.
# Replicates runtime requirements in meta.yaml and setup.py.
runtime_reqs = [
    # Disable abseil-cpp requirements to avoid pip install failures (no PyPI package exists)
    # Assume it's conda install'ed as part of build-level requirements.
    # _ABSEIL_REQ,
    _NUMPY_REQ,
    _TBB_REQ,
    _ZSTD_REQ,
]

# Complete test requirements for riptide_cpp tests.
# Needed to engage all features and their tests (including riptable/python tests).
tests_reqs = [
    # Disable benchmark requirement to avoid pip install failures (PyPI pkg is 'google-benchmark')
    # Assume it's conda install'ed as part of build-level requirements.
    # TODO: Can extend this script to support PyPI vs Conda requirements to map correct names
    # _BENCHMARK_REQ,
]
# Add riptable tests requirements
tests_reqs += [
    "arrow",
    "bokeh",
    "bottleneck",
    "flake8",
    "hypothesis",
    "ipykernel",
    "ipython<8.13" if is_python(3, 8) else "ipython",
    "matplotlib",
    "nose",
    "pyarrow",
    "pytest",
]
# Add riptable runtime requirements
tests_reqs += [
    "ansi2html>=1.5.2",
    "numba>=0.56.2",
    _NUMPY_REQ,
    "pandas>=0.24,<2.0",
    "python-dateutil",
]

# Complete developer requirements.
# Union of all above requuirements plus those needed for code contributions.
developer_reqs = (
    [
        "black",
        "clang-format",
        "setuptools_scm",
    ]
    + conda_reqs
    + pypi_reqs
    + runtime_reqs
    + tests_reqs
    + toolchain_reqs
)
if is_linux():
    developer_reqs += [
        "gdb",
    ]

target_reqs = {
    "conda": conda_reqs,
    "developer": developer_reqs,
    "extras": extras_reqs,
    "native": native_reqs,
    "pypi": pypi_reqs,
    "runtime": runtime_reqs,
    "tests": tests_reqs,
    "toolchain": toolchain_reqs,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "targets", help="requirement targets", choices=target_reqs.keys(), nargs="+"
)
parser.add_argument("--out", help="output file", type=str)
args = parser.parse_args()

reqs = list({r for t in args.targets for r in target_reqs[t]})
reqs.sort()

# Emit plain list to enable usage like: conda install $(gen_requirements.py developer)
out = open(args.out, "w") if args.out else sys.stdout
try:
    for req in reqs:
        print(req, file=out)
finally:
    if args.out:
        out.close()
