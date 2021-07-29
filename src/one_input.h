#ifndef RIPTABLECPP_ONE_INPUT_H
#define RIPTABLECPP_ONE_INPUT_H

#include "RipTide.h"
#include "MathWorker.h"
#include "operation_traits.h"
#include "ndarray.h"

#include <variant>
#include <optional>
#include <utility>
#include <cstddef>

extern "C"
{
   PyObject * process_one_input( PyArrayObject const* in_array, PyArrayObject * out_object_1, int32_t function_num, int32_t numpy_intype, int32_t numpy_outtype = -1 );
}


namespace internal
{
   using chosen_traits_t = std::pair< std::optional< operation_t >, std::optional< data_type_t > >;

   void calculate( char * out_ptr, char const * in_ptr, npy_intp const len, int64_t const stride, operation_t const requested_op, data_type_t const in_type );
   chosen_traits_t set_traits( int32_t const function_num, int32_t const numpy_intype );
}
#endif
