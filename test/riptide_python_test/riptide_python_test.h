#ifndef RIPTIDE_PYTHON_TEST_H
#define RIPTIDE_PYTHON_TEST_H

// Undo the damage we're about to do by undefining a reserved macro name
#if defined(_MSC_VER) && defined(_DEBUG) && _MSC_VER >= 1930
    #include <corecrt.h>
#endif

// Hack because debug builds force python36_d.lib
#define MS_NO_COREDLL    // don't add import libs by default
#define Py_ENABLE_SHARED // but do enable shared libs
#include <pyconfig.h>
#undef Py_DEBUG // don't use debug Python APIs

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//#define PY_ARRAY_UNIQUE_SYMBOL riptide_python_test_global
#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API
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
    extern PyObject * get_named_function(PyObject * module_p, char const * name_p);
    extern void pyobject_printer(PyObject * printable);
}

extern PyObject * riptide_module_p;
extern PyObject * riptable_module_p;

enum struct hash_choice_t
{
    hash_linear,
    tbb,
};

inline hash_choice_t runtime_hash_choice;
#endif
