#define PYTHON_TEST_MAIN (1)
#include "riptide_python_test.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#ifdef WIN32
    #include <Windows.h>
#endif

extern "C"
{
    extern PyObject * PyInit_riptide_cpp();
}

PyObject * riptide_module_p{ nullptr };

namespace riptide_python_test::internal
{
    PyObject * get_named_function(PyObject * module_p, char const * name_p)
    {
        PyObject * dictionary{ PyModule_GetDict(module_p) };
        PyObject * function_name{ Py_BuildValue("s", name_p) };
        PyObject * function_object{ PyDict_GetItem(dictionary, function_name) };

        return function_object;
    }
}

namespace
{
    struct options
    {
        bool list_tests{false};
        std::string test_filter{};
    };

    options parse_options(int argc, char const ** argv);

    int start_python(int argc, char const ** argv);
}

int main(int argc, char const ** argv)
{
    auto const options{parse_options(argc, argv)};

    auto & runner{boost::ut::cfg<>};

    boost::ut::options ut_options;
    if (options.list_tests)
    {
        ut_options.dry_run = true;
    }
    if (!options.test_filter.empty())
    {
        ut_options.filter = options.test_filter;
    }
    runner = ut_options;

    if (start_python(argc, argv) != 0)
    {
        exit(1);
    }

    auto result{runner.run()};

    return result;
}

namespace
{
    options parse_options(int const argc, char const ** const argv)
    {
        options options;

        for (int ac{1}; ac < argc; ++ac)
        {
            std::string_view const arg{argv[ac]};

            if (arg == "--ut_list_tests")
            {
                options.list_tests = true;
            }

            else if (arg == "--ut_filter")
            {
                if (ac == argc - 1)
                {
                    fprintf(stderr, "Fatal error: missing filter argument");
                    exit(1);
                }

                options.test_filter = argv[++ac];
            }

            else
            {
                fprintf(stderr, "Fatal error: unrecognized option, %s\n", std::string(arg).c_str());
                exit(1);
            }
        }

        return options;
    }

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

        /* Pass argv[0] to the Python interpreter */
        Py_SetProgramName(program);

        /* Start the interpreter, load numpy */
        Py_Initialize();

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

        return 0;
    }
}
