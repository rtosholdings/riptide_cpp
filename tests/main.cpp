// Hack because debug builds force python36_d.lib
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#include <gtest/gtest.h>

namespace
{
    class PythonInterp
    {
    public:
        PythonInterp(char const * name)
        {
            program_ = Py_DecodeLocale(name, nullptr);
            Py_SetProgramName(program_);
            Py_Initialize();
        }

        ~PythonInterp()
        {
            if (Py_FinalizeEx() >= 0)
            {
                PyMem_RawFree(program_);
            }
        }

    private:
        wchar_t * program_{};
    };
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // This lets c++ exceptions and windows SEH (i.e. null pointer/segv)
    // ripple up so we can write tests to document when we expect this
    // to happen.
    ::testing::GTEST_FLAG(catch_exceptions) = false;

    PythonInterp const interp{argv[0]};

    auto const ret{RUN_ALL_TESTS()};

    return ret;
}