#ifndef RIPTIDE_PYTHON_TEST_H
#define RIPTIDE_PYTHON_TEST_H

#ifdef _DEBUG
#undef _DEBUG
#include "Python.h"
#define _DEBUG
#else
#include "Python.h"
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL riptide_python_test_global
#ifndef PYTHON_TEST_MAIN
#define NO_IMPORT_ARRAY (1)
#else
#undef NO_IMPORT
#undef NO_IMPORT_ARRAY
#endif

#include "numpy/arrayobject.h"
#include "numpy_traits.h"

namespace riptide_python_test::internal
{
    extern PyObject * get_named_function( PyObject * module_p, char const * name_p );
}

extern PyObject * riptide_module_p;

#endif
