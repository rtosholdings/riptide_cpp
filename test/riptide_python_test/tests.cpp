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
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "TestNumpy");
            PyObject * retval = PyObject_CallFunction(function_object, "d", 3.142);

            expect(function_object != nullptr);
            expect(retval != nullptr);
            expect(retval == Py_None);
        };
    };

    suite reduce_ops = []
    {
        "reduce_nanmean_char"_test = [&]
        {
            std::array<char, 5> test_values{ 'a', 'a', 'a', 'b', 'c' };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_BYTE, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 103) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
        };

        "reduce_mean_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, 2.0f, 0.0f, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 102) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 2.0_f);
            expect(is_float_type == 1_i);
        };

        "reduce_nanmean_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, 2.0f, NAN, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 103) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 2.5_f);
            expect(is_float_type == 1_i);
        };

        "reduce_nanmax"_test = [&]
        {
            std::array<double, 5> test_values{ 3.0, NAN, 2.0, 3.142, -5.0 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_DOUBLE, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 203) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            double ret_val = PyFloat_AsDouble(retval);
            expect(ret_val == 3.142_d);
            expect(is_float_type == 1_i);
        };
    };
}
