#include "riptide_python_test.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <array>

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

            expect(function_object != nullptr);
            expect(retval != nullptr);
        };
    };

    suite reduce_ops = []
    {
        "reduce_nanmean_"_test = [&]
        {
            std::array<char,6> test_values{'a','a','a','b','c'};
            npy_intp one_dim{1};
            PyObject * array{ PyArray_SimpleNewFromData( 5, &one_dim, NPY_BYTE, test_values.data() ) };
            PyObject * function_object = riptide_python_test::internal::get_named_function( riptide_module_p, "Reduce" );

            expect(array != nullptr);
            expect(function_object != nullptr);
        };
    };
}
