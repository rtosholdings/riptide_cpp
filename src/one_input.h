#ifndef RIPTABLECPP_ONE_INPUT_H
#define RIPTABLECPP_ONE_INPUT_H

#include "RipTide.h"
#include "MathWorker.h"
#include "ndarray.h"
#include "operation_traits.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <variant>

extern "C"
{
    PyObject * process_one_input(PyArrayObject const * in_array, PyArrayObject * out_object_1, int32_t function_num,
                                 int32_t numpy_intype, int32_t numpy_outtype = -1);
}

namespace riptable_cpp
{
    std::optional<riptable_cpp::array_content_t> get_type_trait(int32_t const numpy_intype);
    std::optional<riptable_cpp::operation_t> get_op_trait(int32_t const function_num);
} // namespace riptable_cpp
#endif
