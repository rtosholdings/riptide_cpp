#include "riptide_python_test.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <array>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    suite reduce_ops = []
    {
        "reduce_nanmean_"_test = [&]
        {
            std::array<char,6> test_values{'a','a','a','b','c'};
            PyObject * function_object = riptide_python_test::internal::get_named_function( riptide_module_p, "Reduce" );
            PyObject * retval = PyObject_CallFunction(function_object, "d", 3.142);

            expect(retval != nullptr);
        };
    };
}
