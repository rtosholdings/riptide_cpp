#define PYTHON_TEST_MAIN (1)
#include "riptide_python_test.h"

#include "ut_core.h"

#ifdef WIN32
    #include <Windows.h>
#endif

extern "C"
{
    extern PyObject * PyInit_riptide_cpp();
}

namespace
{
    int start_python(int const argc, char const ** const argv)
    {
        wchar_t * program = Py_DecodeLocale(argv[0], NULL);
        if (program == NULL)
        {
            fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
            return -1;
        }

        if (PyImport_AppendInittab("riptide_cpp", PyInit_riptide_cpp) == -1)
        {
            fprintf(stderr, "Error: Could not extend the built-in modules table\n");
            return -1;
        }

        PyStatus status;

        PyConfig config;
        PyConfig_InitPythonConfig(&config);

        /*
        Set the program name (for the Python interpreter).
        This implicitly pre-initializes Python:
            https://docs.python.org/3.11/c-api/init_config.html#initialization-with-pyconfig
        */
        status = PyConfig_SetString(&config, &config.program_name, program);
        if (PyStatus_Exception(status))
        {
            fprintf(stderr, "Failed to set the Python program name.\n");
            return -1;
        }

        /* Start the interpreter, load numpy */
        Py_Initialize();

#ifdef WIN32
        // Workaround for NumPy loading failure in Python 3.8+.
        // Need to explicitly set the DLL search directory in order to find the MKL DLLs
        // located in $PYTHONHOME/Library/bin.
        // See https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew for details.

        wchar_t dll_directory[_MAX_PATH + 1]{ 0 };
        GetEnvironmentVariableW(L"PYTHONHOME", dll_directory, sizeof(dll_directory));
        if (! dll_directory[0])
        {
            fprintf(stderr, "Cannot obtain the environment variable PYTHONHOME\n");
            return -1;
        }
        wcscat_s(dll_directory, L"\\Library\\bin");

        if (! SetDllDirectoryW(dll_directory))
        {
            auto const error{ GetLastError() };
            fprintf(stderr, "Cannot set DLL search directory to \"%ws\": error=%d\n", dll_directory, error);
            return -1;
        }
#endif

        /* Load our module */
        riptide_module_p = PyImport_ImportModule("riptide_cpp");
        if (riptide_module_p == nullptr)
        {
            PyErr_Print();
            fprintf(stderr, "Error: Could not import module 'riptide_cpp'\n");
            return -1;
        }

        if (PY_ARRAY_UNIQUE_SYMBOL == nullptr)
        {
            import_array1(-1);
        }

        riptable_module_p = PyImport_ImportModule("riptable");
        if (riptable_module_p == nullptr)
        {
            PyErr_Print();
            fprintf(stderr, "Error: Could not import module 'riptable'\n");
            return -1;
        }

        return 0;
    }

    void stop_python()
    {
        Py_Finalize();
    }
}

int main(int argc, char const ** argv)
{
    auto const ut_options{ riptide_utility::ut::parse_options(argc, argv) };

    boost::ut::cfg<boost::ut::override> = ut_options;

    if (! ut_options.dry_run)
    {
        if (start_python(argc, argv) != 0)
        {
            exit(1);
        }
    }

    auto result{ boost::ut::cfg<boost::ut::override>.run() };

    if (! ut_options.dry_run)
    {
        stop_python();
    }

    return result;
}
