#ifndef RIPTABLE_OPERATION_TRAITS_H
#define RIPTABLE_OPERATION_TRAITS_H

#include "RipTide.h"

#include <type_traits>
#include <variant>
#include <optional>

namespace riptable_cpp
{
    template <typename arithmetic_concept, typename simd_concept, typename bitmask_concept, typename Enable = void>
    struct data_type_traits
    {
    };

    template <typename arithmetic_concept, typename simd_concept, typename bitmask_concept>
    struct data_type_traits<arithmetic_concept, simd_concept, bitmask_concept,
                            std::enable_if_t<std::is_arithmetic_v<arithmetic_concept>>>
    {
        using data_type = arithmetic_concept;
        using calculation_type = simd_concept;
        using bitmask_type = bitmask_concept;
    };

    using int8_traits = data_type_traits<int8_t, __m256i, uint8_t>;
    using int16_traits = data_type_traits<int16_t, __m256i, uint16_t>;
    using int32_traits = data_type_traits<int32_t, __m256i, uint32_t>;
    using int64_traits = data_type_traits<int64_t, __m256i, uint64_t>;
    using uint8_traits = data_type_traits<uint8_t, __m256i, uint8_t>;
    using uint16_traits = data_type_traits<uint16_t, __m256i, uint16_t>;
    using uint32_traits = data_type_traits<uint32_t, __m256i, uint32_t>;
    using uint64_traits = data_type_traits<uint64_t, __m256i, uint64_t>;
    using float_traits = data_type_traits<float, __m256, uint32_t>;
    using double_traits = data_type_traits<double, __m256d, uint64_t>;

    using data_type_t = std::variant<int8_traits, int16_traits, int32_traits, int64_traits, uint8_traits, uint16_traits,
                                     uint32_traits, uint64_traits, float_traits, double_traits>;

    struct abs_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct fabs_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::false_type;
    };
    struct sign_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::false_type;
    };
    struct floatsign_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::false_type;
    };
    struct neg_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::false_type;
    };
    struct bitwise_not_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::false_type;
    };
    struct not_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct isnotnan_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::true_type;
    };
    struct isnan_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::true_type;
    };
    struct isfinite_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct isnotfinite_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct isinf_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct isnotinf_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct isnormal_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct isnotnormal_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct isnanorzero_op
    {
        static constexpr bool value_return = false;
        using simd_implementation = std::false_type;
    };
    struct round_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct floor_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct trunc_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct ceil_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct sqrt_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct log_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct log2_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct log10_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct exp_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct exp2_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct cbrt_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct tan_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct cos_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct sin_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::true_type;
    };
    struct signbit_op
    {
        static constexpr bool value_return = true;
        using simd_implementation = std::false_type;
    };

    using single_operation_t = std::variant<abs_op, fabs_op, sign_op, floatsign_op, neg_op, bitwise_not_op, not_op, isnotnan_op, isnan_op,
                                     isfinite_op, isnotfinite_op, isinf_op, isnotinf_op, isnormal_op, isnotnormal_op,
                                     isnanorzero_op, round_op, floor_op, trunc_op, ceil_op, sqrt_op, log_op, log2_op, log10_op,
                                     exp_op, exp2_op, cbrt_op, tan_op, cos_op, sin_op, signbit_op>;

    struct add_op{};
    struct sub_op{};
    struct mul_op{};
    struct div_op{};
    struct subdatetime_op{};
    struct subdates_op{};
    struct floor_div{};
    struct mod_op{};
    struct power_op{};
    struct remainder_op{};
    struct fmod_op{};

    using multiarg_operation_t = std::variant< add_op, sub_op, mul_op, div_op, subdatetime_op, subdates_op, floor_div, mod_op, power_op, remainder_op, fmod_op >;

    inline namespace implementation
    {
        std::optional<data_type_t> data_type_trait( int32_t numpy_type)
        {
            std::optional<data_type_t> retval{};

            switch (numpy_type)
            {
            case NPY_INT8:
                retval = int8_traits{};
                break;
            case NPY_INT16:
                retval = int16_traits{};
                break;
#if RT_COMPILER_MSVC
            case NPY_INT:
#endif
            case NPY_INT32:
                retval = int32_traits{};
                break;
#if (RT_COMPILER_CLANG || RT_COMPILER_GCC)
            case NPY_LONGLONG:
#endif
            case NPY_INT64:
                retval = int64_traits{};
                break;
            case NPY_UINT8:
                retval = uint8_traits{};
                break;
            case NPY_UINT16:
                retval = uint16_traits{};
                break;
#if RT_COMPILER_MSVC
            case NPY_UINT:
#endif
            case NPY_UINT32:
                retval = uint32_traits{};
                break;
#if (RT_COMPILER_CLANG || RT_COMPILER_GCC)
            case NPY_ULONGLONG:
#endif
            case NPY_UINT64:
                retval = uint64_traits{};
                break;
            case NPY_FLOAT:
                retval = float_traits{};
                break;
            case NPY_DOUBLE:
                retval = double_traits{};
                break;
            };
            
            return retval;
        }
        
    }

} // namespace riptable_cpp

#endif
