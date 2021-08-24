#ifndef RIPTABLE_OPERATION_TRAITS_H
#define RIPTABLE_OPERATION_TRAITS_H

#include "RipTide.h"

#include <type_traits>
#include <variant>

namespace internal
{
   // implementaion idea:
   // Have a two variant values that are parsed / set from a top-level switch.
   // One provides an instance the trait class for the inbound data type (which includes the AVX256 version of that type)
   // The other provides the trait class for the requested operation (probably just an empty struct, classic trait class)
   // Then, rather than having XXX_OP functions, have "perform_operation" that specializes on those two.
   // That way, the entire flow is compile time without any pointer-to-function stuff getting in the way
   // of letting the compiler see through the control flow to optimize deeper.

   template< typename arithmetic_concept, typename Enable = void >
   struct data_type_traits
   {
   };

   template< typename  arithmetic_concept > 
   struct data_type_traits< arithmetic_concept, std::enable_if_t< std::is_arithmetic_v< arithmetic_concept > > >
   {
      using data_type = arithmetic_concept;
      using calculation_type = __m256i;      
   };

   using data_type_t = std::variant<
       data_type_traits< int8_t >
      , data_type_traits< int16_t >
      , data_type_traits< int32_t >
      , data_type_traits< int64_t >
      , data_type_traits< uint8_t >
      , data_type_traits< uint16_t >
      , data_type_traits< uint32_t >
      , data_type_traits< uint64_t >
      , data_type_traits< float >
      , data_type_traits< double >
      >;

   struct value_return {};
   struct bool_return {};
   
   struct abs_op { using simd_implementation = std::true_type; };
   struct fabs_op { using simd_implementation = std::false_type; };
   struct sign_op { using simd_implementation = std::false_type; };
   struct floatsign_op { using simd_implementation = std::false_type; };
   struct neg_op { using simd_implementation = std::false_type; };
   struct bitwise_not_op { using simd_implementation = std::false_type; };
   struct not_op { using simd_implementation = std::false_type; };
   struct isnotnan_op { using simd_implementation = std::true_type; };
   struct isnan_op { using simd_implementation = std::true_type; };
   struct isfinite_op { using simd_implementation = std::false_type; };
   struct isnotfinite_op { using simd_implementation = std::false_type; };
   struct isinf_op { using simd_implementation = std::false_type; };
   struct isnotinf_op { using simd_implementation = std::false_type; };
   struct isnormal_op { using simd_implementation = std::false_type; };
   struct isnotnormal_op { using simd_implementation = std::false_type; };
   struct isnanorzero_op { using simd_implementation = std::false_type; };
   struct round_op { using simd_implementation = std::true_type; };
   struct floor_op { using simd_implementation = std::true_type; };
   struct trunc_op { using simd_implementation = std::true_type; };
   struct ceil_op { using simd_implementation = std::true_type; };
   struct sqrt_op { using simd_implementation = std::true_type; };
   struct log_op { using simd_implementation = std::true_type; };
   struct log2_op { using simd_implementation = std::true_type; };
   struct log10_op { using simd_implementation = std::true_type; };
   struct exp_op { using simd_implementation = std::true_type; };
   struct exp2_op { using simd_implementation = std::true_type; };
   struct cbrt_op { using simd_implementation = std::true_type; };
   struct tan_op { using simd_implementation = std::true_type; };
   struct cos_op { using simd_implementation = std::true_type; };
   struct sin_op { using simd_implementation = std::true_type; };
   struct signbit_op { using simd_implementation = std::false_type; };

   using operation_t = std::variant
   <
      abs_op
      ,fabs_op
      ,sign_op
      ,floatsign_op
      ,neg_op
      ,bitwise_not_op
      ,not_op
      ,isnotnan_op
      ,isnan_op
      ,isfinite_op
      ,isnotfinite_op
      ,isinf_op
      ,isnotinf_op
      ,isnormal_op
      ,isnotnormal_op
      ,isnanorzero_op
      ,round_op
      ,floor_op
      ,trunc_op
      ,ceil_op
      ,sqrt_op
      ,log_op
      ,log2_op
      ,log10_op
      ,exp_op
      ,exp2_op
      ,cbrt_op
      ,tan_op
      ,cos_op
      ,sin_op
      ,signbit_op
   >;      

}

#endif
