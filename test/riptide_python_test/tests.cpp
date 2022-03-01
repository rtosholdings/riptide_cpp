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
            std::array<char,5> test_values{'a','a','a','b','c'};
            npy_intp const dim_len{5};
            PyObject * array{ PyArray_SimpleNew(1, &dim_len, NPY_BYTE ) };
            PyObject * function_object = riptide_python_test::internal::get_named_function( riptide_module_p, "Reduce" );
            PyObject * reduce_fn_num{ Py_BuildValue( "i", 103 ) };
            PyObject * arg_tuple{ PyTuple_Pack( 2, array, reduce_fn_num ) };
            PyObject * retval = PyObject_CallObject(function_object, arg_tuple);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(arg_tuple != nullptr);
            expect(retval != nullptr);
        };
    };
}
