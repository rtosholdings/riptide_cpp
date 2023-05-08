#include "CommonInc.h"

#include "riptide_python_test.h"

#include "boost/ut.hpp"

#include <array>

using namespace boost::ut;
using boost::ut::suite;

using riptide_python_test::internal::get_named_function;

namespace
{
    suite riptide_ops = []
    {
        "test_basic_math_ok"_test = [&]
        {
            PyObject * const function_object{ get_named_function(riptide_module_p, "BasicMathTwoInputs") };
            // (double+double->?): OK -> returns double array
            PyObject * const retval{ PyObject_CallFunction(function_object, "(dd)ii", 3.14, 2.78, MATH_OPERATION::ADD,
                                                           NPY_DOUBLE) };

            expect(! PyErr_Occurred());
            expect(retval != nullptr);
            expect(PyArray_Check(retval));

            Py_XDECREF(retval);
        };

        "test_basic_math_unsupported"_test = [&]
        {
            PyObject * const function_object{ get_named_function(riptide_module_p, "BasicMathTwoInputs") };
            // (double+str->?): unsupported -> returns None
            PyObject * const retval{ PyObject_CallFunction(function_object, "(ds)ii", 3.14, "Hi", MATH_OPERATION::ADD,
                                                           NPY_DATETIME) };

            expect(! PyErr_Occurred());
            expect(retval == Py_None);
        };

        "test_basic_math_error"_test = [&]
        {
            PyObject * const function_object{ get_named_function(riptide_module_p, "BasicMathTwoInputs") };
            // (double+double->int): illegal -> raises exception
            PyObject * const retval{ PyObject_CallFunction(function_object, "(ddi)ii", 3.14, 2.78, -1, MATH_OPERATION::ADD,
                                                           NPY_DOUBLE) };

            expect(PyErr_Occurred() == PyExc_ValueError);
            expect(retval == nullptr);

            PyErr_Clear();
        };
    };
}
