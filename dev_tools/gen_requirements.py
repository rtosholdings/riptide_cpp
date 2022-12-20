# Generates requirements for riptide_cpp

import argparse
import platform
import sys


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return platform.system() == "Windows"


_CMAKE_REQ = "cmake>=3.18"
_NUMPY_REQ = "numpy>=1.22"
_TBB_DEVEL_REQ = "tbb-devel==2021.6.*"

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
# Most everything else will be specified in setup.py.
pypi_reqs = [
    _CMAKE_REQ,  # TODO: Remove this once setup.py sdist properly requires cmake!
    _TBB_DEVEL_REQ,  # needed because PyPI tbb-devel pkg doesn't contain CMake files yet
    "wheel",
] + toolchain_reqs

# Native CMake build requirements.
native_reqs = [
    _CMAKE_REQ,
    _NUMPY_REQ,
    _TBB_DEVEL_REQ,
] + toolchain_reqs

# Extras build requirements (same as native build requirements).
extras_reqs = native_reqs

# Runtime requirements for riptable and riptide_cpp.
# Replicates runtime requirements in meta.yaml and setup.py.
runtime_reqs = [
    "ansi2html>=1.5.2",
    "numba>=0.56.2",
    _NUMPY_REQ,
    "pandas>=0.24,<2.0",
    "python-dateutil",
    "tbb==2021.6.*",
]

# Complete test requirements for riptable tests.
# Needed to engage all features and their tests.
tests_reqs = [
    "arrow",
    "bokeh",
    "bottleneck",
    "flake8",
    "hypothesis",
    "ipykernel",
    "ipython",
    "nose",
    "pyarrow",
    "pytest",
]

# Complete developer requirements.
# Union of all above requuirements plus those needed for code contributions.
developer_reqs = (
    ["clang-format", "setuptools_scm"]
    + conda_reqs
    + pypi_reqs
    + runtime_reqs
    + tests_reqs
    + toolchain_reqs
)

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
