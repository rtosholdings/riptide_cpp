find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

################################################################################
# Determine the filename suffix and extension to use; the compiled extension (library)
# will have this appended to the name so it's recognized as a CPython extension.
################################################################################
execute_process(
  COMMAND "python" -c "import importlib.machinery; print(importlib.machinery.EXTENSION_SUFFIXES[0], end='')"
    RESULT_VARIABLE _PYTHON_EXTENSION_SUFFIX_RESULT
    OUTPUT_VARIABLE _PYTHON_EXTENSION_SUFFIX
    ERROR_QUIET)
if(NOT _PYTHON_EXTENSION_SUFFIX_RESULT EQUAL "0")
  message(WARNING "Cannot deduce the Python C extension file suffix, fall back to default, set PYTHON_EXT_LIB_SUFFIX to override")
  if(WIN32)
    set(_PYTHON_EXTENSION_SUFFIX ".lib")
  else(WIN32)
    set(_PYTHON_EXTENSION_SUFFIX ".so")
  endif(WIN32)
endif(NOT _PYTHON_EXTENSION_SUFFIX_RESULT EQUAL "0")

get_filename_component(PYTHON_EXE_DIR ${Python3_EXECUTABLE} DIRECTORY )

set(PYTHON_EXT_LIB_SUFFIX ${_PYTHON_EXTENSION_SUFFIX} CACHE STRING "The suffix of the C extension library")
message(STATUS "Using Python C extension file suffix: ${PYTHON_EXT_LIB_SUFFIX}")
