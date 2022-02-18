#include "riptide_python_test.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

using namespace boost::ut;
using boost::ut::suite;

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
