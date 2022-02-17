#include "RipTide.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

using namespace boost::ut;
using boost::ut::suite;

namespace riptide_python_test::internal
{
    extern PyObject * get_named_function( PyObject * module_p, char const * name_p );
}

extern PyObject * riptide_module_p;

namespace
{
    suite riptide_ops = []
    {
        "test_numpy_float"_test = [&]
        {
            PyObject * function_object = riptide_python_test::internal::get_named_function( riptide_module_p, "TestNumpy" );
            PyObject * retval = PyObject_CallFunction(function_object, "d", 3.142);

            expect(retval != nullptr);
        };
    };
}
