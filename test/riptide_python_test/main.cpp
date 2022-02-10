#define SHAREDATA_MAIN_C_FILE
#include "RipTide.h"

#ifdef WIN32
#include <Windows.h>
#endif

extern "C"
{
    extern PyObject * PyInit_riptide_cpp();
}

int main()
{
#ifdef WIN32
    SetEnvironmentVariable("PYTHONPATH",".");
#endif
    Py_Initialize();
    import_array();
    PyInit_riptide_cpp();
}
