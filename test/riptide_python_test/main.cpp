#define SHAREDATA_MAIN_C_FILE
#include "RipTide.h"

#ifdef WIN32
#include <Windows.h>
#endif

extern "C"
{
    extern PyObject * PyInit_riptide_cpp();
}

PyObject * riptide_module_p{nullptr};

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
