#define SHAREDATA_MAIN_C_FILE
#include "RipTide.h"

#ifdef WIN32
#include <Windows.h>
#endif

extern "C"
{
    extern PyObject * PyInit_riptide_cpp();
}

int main(int argc, char const ** argv)
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    Py_Initialize();
    import_array();
    PyInit_riptide_cpp();
}
