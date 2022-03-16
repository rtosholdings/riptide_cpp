#include "riptide_python_test.h"

PyObject * riptide_module_p{ nullptr };
PyObject * riptable_module_p{ nullptr };

namespace riptide_python_test::internal
{
    PyObject * get_named_function(PyObject * module_p, char const * name_p)
    {
        PyObject * dictionary{ PyModule_GetDict(module_p) };
        PyObject * function_name{ Py_BuildValue("s", name_p) };
        PyObject * function_object{ PyDict_GetItem(dictionary, function_name) };

        return function_object;
    }

    void pyobject_printer(PyObject * object_p)
    {
        PyObject * globals{ Py_BuildValue("{sO}", "printable", object_p) };
        PyObject * locals{ Py_BuildValue("{}") };
        PyRun_String("print(printable)", Py_single_input, globals, locals);
    }
}
