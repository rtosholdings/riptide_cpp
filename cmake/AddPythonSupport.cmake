set(Python_FIND_STRATEGY "LOCATION")
set(Python3_FIND_VIRTUALENV "FIRST")
find_package(Python3 "${RIPTIDE_PYTHON_VER}" EXACT COMPONENTS Interpreter Development REQUIRED)
set(Python3_NumPy_INCLUDE_DIR "${Python3_SITELIB}/numpy/core/include" )

message(NOTICE "Python3_VERSION = ${Python3_VERSION}")
message(NOTICE "Python3_VERSION_MAJOR = ${Python3_VERSION_MAJOR}")
message(NOTICE "Python3_VERSION_MINOR = ${Python3_VERSION_MINOR}")
message(NOTICE "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(NOTICE "Python3_STDLIB = ${Python3_STDLIB}")
message(NOTICE "Python3_SITELIB = ${Python3_SITELIB}")
message(NOTICE "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(NOTICE "Python3_LINK_OPTIONS = ${Python3_LINK_OPTIONS}")
message(NOTICE "Python3_LIBRARIES = ${Python3_LIBRARIES}")
message(NOTICE "Python3_LIBRARY_DIRS = ${Python3_LIBRARY_DIRS}")
message(NOTICE "Python3_RUNTIME_LIBRARY_DIRS = ${Python3_RUNTIME_LIBRARY_DIRS}")
message(NOTICE "Python3_NumPy_INCLUDE_DIR hint = ${Python3_NumPy_INCLUDE_DIR}")
message(NOTICE "Python_FIND_STRATEGY = ${Python_FIND_STRATEGY}")

find_package(Python3 "${RIPTIDE_PYTHON_VER}" COMPONENTS Interpreter Development NumPy)

message(NOTICE "Python3_VERSION = ${Python3_VERSION}")
message(NOTICE "Python3_VERSION_MAJOR = ${Python3_VERSION_MAJOR}")
message(NOTICE "Python3_VERSION_MINOR = ${Python3_VERSION_MINOR}")
message(NOTICE "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(NOTICE "Python3_STDLIB = ${Python3_STDLIB}")
message(NOTICE "Python3_SITELIB = ${Python3_SITELIB}")
message(NOTICE "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(NOTICE "Python3_LINK_OPTIONS = ${Python3_LINK_OPTIONS}")
message(NOTICE "Python3_LIBRARIES = ${Python3_LIBRARIES}")
message(NOTICE "Python3_LIBRARY_DIRS = ${Python3_LIBRARY_DIRS}")
message(NOTICE "Python3_RUNTIME_LIBRARY_DIRS = ${Python3_RUNTIME_LIBRARY_DIRS}")
message(NOTICE "Python3_NumPy_INCLUDE_DIRS = ${Python3_NumPy_INCLUDE_DIRS}")

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
message(NOTICE "Using Python C extension file suffix: ${PYTHON_EXT_LIB_SUFFIX}")