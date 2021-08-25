#ifndef RIPTABLE_OPERATION_TRAITS_H
#define RIPTABLE_OPERATION_TRAITS_H

#include "RipTide.h"

#include <type_traits>
#include <variant>

namespace internal
{
   template< typename arithmetic_concept, typename simd_concept, typename Enable = void >
   struct data_type_traits
   {
   };

   template< typename  arithmetic_concept, typename simd_concept > 
   struct data_type_traits< arithmetic_concept, typename simd_concept, std::enable_if_t< std::is_arithmetic_v< arithmetic_concept >, void > >
   {
      using data_type = arithmetic_concept;
      using calculation_type = simd_concept;
   };

   using int8_traits = data_type_traits< int8_t, __m256i >;
   using int16_traits = data_type_traits< int16_t, __m256i >;
   using int32_traits = data_type_traits< int32_t, __m256i >;
   using int64_traits = data_type_traits< int64_t, __m256i >;
   using uint8_traits = data_type_traits< uint8_t, __m256i >;
   using uint16_traits = data_type_traits< uint16_t, __m256i >;
   using uint32_traits = data_type_traits< uint32_t, __m256i >;
   using uint64_traits = data_type_traits< uint64_t, __m256i >;
   using float_traits = data_type_traits< float, __m256 >;
   using double_traits = data_type_traits< double, __m256d >;
   
   using data_type_t = std::variant<
      int8_traits
      ,int16_traits
      ,int32_traits
      ,int64_traits
      ,uint8_traits
      ,uint16_traits
      ,uint32_traits
      ,uint64_traits
      ,float_traits
      ,double_traits
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
