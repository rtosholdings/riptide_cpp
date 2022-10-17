#include "riptide_python_test.h"

#include "boost/ut.hpp"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    std::random_device dev{};

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

        "reduce_min"_test = [&]
        {
            std::array<double, 5> test_values{ 3.0, -42.5, 2.0, 3.142, -5.0 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_DOUBLE, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 200) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            double ret_val = PyFloat_AsDouble(retval);
            expect(ret_val == -42.5_d);
            expect(is_float_type == 1_i);
        };

        "reduce_nanmin"_test = [&]
        {
            std::array<double, 5> test_values{ 3.0, NAN, 2.0, 3.142, -5.0 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_DOUBLE, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 201) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            double ret_val = PyFloat_AsDouble(retval);
            expect(ret_val == -5.0_d);
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

        "reduce_sum"_test = [&]
        {
            std::array<int32_t, 5> test_values{ 1, 2, 42, 3, 4 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_INT32, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 0) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int32_t ret_val = PyLong_AsLong(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 52_i);
            expect(is_float_type == 0_i);
        };

        "reduce_sum_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, 2.0f, 42.5, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 0) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 52.5_f);
            expect(is_float_type == 1_i);
        };

        "reduce_nansum_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, NAN, 42.5, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 1) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 50.5_f);
            expect(is_float_type == 1_i);
        };

        "reduce_float_long"_test = [&]
        {
            std::vector<float> test_data(65536);
            std::iota(std::begin(test_data), std::end(test_data), -10000.0f);
            std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
            npy_intp const dim_len{ 65536 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_data.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_sum{ Py_BuildValue("i", 0) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_sum, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_sum != nullptr);
            expect(retval != nullptr);
            float x = 1492090880.0f;
            expect(ret_val == x) << "We got " << ret_val << " but wanted " << x << "\n";
            expect(is_float_type == 1_i);

            PyObject * reduce_fn_mean{ Py_BuildValue("i", 102) };
            retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_mean, NULL);
            double calc_avg = PyFloat_AsDouble(retval);
            double actual_avg = x / test_data.size();
            is_float_type = PyFloat_Check(retval);

            expect(reduce_fn_mean != nullptr);
            expect(retval != nullptr);
            expect(calc_avg == actual_avg);
            expect(is_float_type == 1_i);

            PyObject * reduce_fn_var{ Py_BuildValue("i", 106) };
            retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_var, NULL);
            ret_val = PyFloat_AsDouble(retval);
            is_float_type = PyFloat_Check(retval);

            expect(reduce_fn_var != nullptr);
            expect(retval != nullptr);
            double y = 357919402.0 + 2.0 / 3.0;
            expect(ret_val == y) << "Calculated variance [" << std::setprecision(21) << ret_val << "] expected was [" << y
                                 << "]\n";
            ;
            expect(is_float_type == 1_i);

            PyObject * reduce_fn_std{ Py_BuildValue("i", 108) };
            retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_std, NULL);
            ret_val = PyFloat_AsDouble(retval);
            is_float_type = PyFloat_Check(retval);

            expect(reduce_fn_var != nullptr);
            expect(retval != nullptr);
            y = std::sqrt(y);
            expect(ret_val == y) << "Calculated population standard deviation [" << std::setprecision(21) << ret_val
                                 << "] expected was [" << y << "]\n";
            ;
            expect(is_float_type == 1_i);
        };

        "reduce_sum_int32_long"_test = [&]
        {
            std::vector<int32_t> test_data(65535);
            std::iota(std::begin(test_data), std::end(test_data), -10000);
            std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
            npy_intp const dim_len{ 65535 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_INT32, test_data.data()) };
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", 0) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int32_t ret_val = PyLong_AsLong(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == std::accumulate(std::begin(test_data), std::end(test_data), 0));
            expect(is_float_type == 0_i);
        };
    };
}
