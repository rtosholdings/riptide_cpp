find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

message(DEBUG "Python3_VERSION = ${Python3_VERSION}")
message(DEBUG "Python3_VERSION_MAJOR = ${Python3_VERSION_MAJOR}")
message(DEBUG "Python3_VERSION_MINOR = ${Python3_VERSION_MINOR}")
message(DEBUG "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(DEBUG "Python3_STDLIB = ${Python3_STDLIB}")
message(DEBUG "Python3_SITELIB = ${Python3_SITELIB}")
message(DEBUG "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(DEBUG "Python3_LINK_OPTIONS = ${Python3_LINK_OPTIONS}")
message(DEBUG "Python3_LIBRARIES = ${Python3_LIBRARIES}")
message(DEBUG "Python3_LIBRARY_DIRS = ${Python3_LIBRARY_DIRS}")
message(DEBUG "Python3_RUNTIME_LIBRARY_DIRS = ${Python3_RUNTIME_LIBRARY_DIRS}")


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

set(PYTHON_EXT_LIB_SUFFIX ${_PYTHON_EXTENSION_SUFFIX} CACHE STRING "The suffix of the C extension library")
message(DEBUG "Using Python C extension file suffix: ${PYTHON_EXT_LIB_SUFFIX}")