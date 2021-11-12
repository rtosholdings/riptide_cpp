#include "two_inputs.h"
#include "operation_traits.h"
#include "two_inputs_impl.h"

#include "CommonInc.h"

#include <variant>
#include <utility>
#include <optional>
#include <type_traits>

namespace riptable_cpp
{
    inline namespace implementation
    {
        struct data_objects
        {
            bool scalar{};
            int32_t numpy_datatype;
            char const * data_ptr{};
        };

        using chosen_traits_t = std::pair<std::optional<data_type_t>, std::optional<data_type_t>>;

        template <typename operation_var, typename data_type_var>
        char * perform_calculation(data_objects const input1, data_objects const input2, char * output)
        {
            return nullptr;
        }
    }

    PyObject * specific_calculate_two_inputs(PyObject * input1, PyObject * input2, PyObject * output, int64_t requested_op)
    {
        arguments[0].scalar = PyArray_IsAnyScalar(input1);
        arguments[0].numpy_datatype = PyArray_TYPE(input1);
        arguments[1].scalar = PyArray_IsAnyScalar(input2);
        arguments[1].numpy_datatype = PyArray_TYPE(input2);

        auto [operation, data_type] = set_multiarg_traits(arguments, requested_op);

        if (operation && data_type)
        {
            data_objects arguments[3];

            // Call perform_calculation, but we need to handle the possible mix of array and scalar.
            return nullptr;
        }

        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject * calculate_two_inputs(PyObject * self, PyObject * args)
    {
        if (Py_SIZE(args) != 3)
        {
            PyErr_Format(PyExc_ValueError,
                         "calculate_two_inputs requires three inputs in the second parameter object: tuple, long, long");
            return nullptr;
        }

        if (not PyTuple_CheckExact(PyTuple_GET_ITEM(args, 0)))
        {
            PyErr_Format(PyExc_ValueError, "calculate_two_inputs takes a tuple");
            return nullptr;
        }

        return specific_calculate_two_inputs(PyTuple_GET_ITEM(PyTuple_GET_ITEM(args, 0), 0),
                                             PyTuple_GET_ITEM(PyTuple_GET_ITEM(args, 0), 1), nullptr,
                                             PyLong_AsLongLong(PyTuple_GET_ITEM(args, 1)));
    }

    PyObject * filtered_calculate(PyObject * self, PyObject * args)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }
}
