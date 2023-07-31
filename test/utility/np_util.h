#pragma once

#include <numpy/arrayobject.h>

namespace riptide_utility::internal
{
    template <NPY_TYPES TypeCode>
    using typecode_to_type = std::integral_constant<NPY_TYPES, TypeCode>;
}