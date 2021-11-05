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
    using chosen_traits_t = std::pair<std::optional<operation_t>, std::optional<data_type_t>>;

    chosen_traits_t set_traits(int32_t const function_num, int32_t const numpy_intype);
} // namespace riptable_cpp
#endif
