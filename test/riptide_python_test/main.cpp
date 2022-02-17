#define PYTHON_TEST_MAIN (1)
#include "riptide_python_test.h"

#ifdef WIN32
#include <Windows.h>
#endif

extern "C"
{
    extern PyObject * PyInit_riptide_cpp();
}

PyObject * riptide_module_p{nullptr};

namespace riptide_python_test::internal
{
    PyObject * get_named_function( PyObject * module_p, char const * name_p )
    {
        PyObject * dictionary{ PyModule_GetDict( module_p ) };
        PyObject * function_name{ Py_BuildValue("s", "TestNumpy") };
        PyObject * function_object{ PyDict_GetItem(dictionary, function_name) };

        return function_object;
    }
}

int main(int argc, char const ** argv)
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    
    if (PyImport_AppendInittab("riptide_cpp", PyInit_riptide_cpp) == -1)
    {
        fprintf(stderr, "Error: Could not extend the built-in modules table\n");
        exit(2);
    }

/* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Start the interpreter, load numpy */
    Py_Initialize();
    import_array();
    
    /* Load our module */
    riptide_module_p = PyImport_ImportModule("riptide_cpp");
    if ( riptide_module_p == nullptr )
    {
        PyErr_Print();
        fprintf(stderr, "Error: Could not import module 'riptide_cpp'\n");
    }
}
