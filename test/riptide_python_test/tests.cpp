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
        "test_numpy_float"_test = [&]
        {
            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "TestNumpy");
            PyObject * retval = PyObject_CallFunction(function_object, "d", 3.142);

            expect(function_object != nullptr);
            expect(retval != nullptr);
            expect(retval == Py_None);
        };

        "test_recycling"_test = [&]
        {
            PyObject * empty_fn{ get_named_function(riptide_module_p, "Empty") };

            PyObject * array_ob{ PyObject_CallFunction(empty_fn, "[i]iiO", 1, 0, 0, Py_False) };
            Py_DECREF(array_ob);

            PyObject * recycled_array_ob{ PyObject_CallFunction(empty_fn, "[i]iiO", 1, 0, 0, Py_False) };
            expect(recycled_array_ob == array_ob);
        };
    };
}
