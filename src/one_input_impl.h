#ifndef RIPTABLE_CPP_ONE_INPUT_IMPL_H
#define RIPTABLE_CPP_ONE_INPUT_IMPL_H
#include "overloaded.h"

#include "MathWorker.h"
#include "RipTide.h"
#include "basic_ops.h"
#include "ndarray.h"
#include "operation_traits.h"

#include "simd/avx2.h"

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

namespace riptable_cpp
{
    inline namespace implementation
    {
        using no_simd_type = typename ::riptide::simd::avx2::template vec256<void>;

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, abs_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };
            using wide_t = typename calculation_t::calculation_type;
            [[maybe_unused]] wide_t const * wide_value_p(reinterpret_cast<wide_t const *>(in_p));

            if constexpr (std::is_unsigned_v<T> == true)
            {
                return T{ value };
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    wide_t temp_wide{ wide_ops.load_unaligned(wide_value_p) };

                    return wide_ops.abs(temp_wide);
                }
                else
                {
                    return T(std::abs(value));
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, fabs_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return T{ value };
            }
            else
            {
                return value < T{} ? T(-value) : T(value);
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, sign_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };

            if constexpr (std::is_unsigned_v<T> == true)
            {
                return T(value) > T{} ? T(1) : T{};
            }
            else
            {
                return value > T{} ? T(1) : T(value) < T{} ? T(-1) : T{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, floatsign_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return T{};
            }
            else
            {
                return value > T{} ? T(1.0) : (value < T{} ? T(-1.0) : value == value ? T{} : T(value));
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, neg_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };

            if constexpr (std::is_unsigned_v<T> == true)
            {
                return T(value);
            }
            else
            {
                return T(-value);
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, bitwise_not_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };

            if constexpr (std::is_floating_point_v<T> == true)
            {
                return T(NAN);
            }
            else
            {
                return T(~value);
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, round_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return T(std::round(value));
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    using simd_t = typename calculation_t::calculation_type;
                    simd_t const wide_value(wide_ops_t::load_unaligned(reinterpret_cast<simd_t const *>(in_p)));
                    return wide_ops.round(wide_value);
                }
                else
                {
                    return T(std::round(value));
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, floor_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };
            using wide_t = typename calculation_t::calculation_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return T(std::floor(value));
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    wide_t const wide_value(wide_ops_t::load_unaligned(reinterpret_cast<wide_t const *>(in_p)));
                    return wide_ops.floor(wide_value);
                }
                else
                {
                    return T(std::floor(value));
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, trunc_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };
            using wide_t = typename calculation_t::calculation_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return T(std::trunc(value));
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    wide_t const wide_value(wide_ops_t::load_unaligned(reinterpret_cast<wide_t const *>(in_p)));
                    return wide_ops.trunc(wide_value);
                }
                else
                {
                    return T(std::trunc(value));
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, ceil_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };
            using wide_t = typename calculation_t::calculation_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return T(std::ceil(value));
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    wide_t const wide_value(wide_ops_t::load_unaligned(reinterpret_cast<wide_t const *>(in_p)));
                    return wide_ops.ceil(wide_value);
                }
                else
                {
                    return T(std::ceil(value));
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, sqrt_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            [[maybe_unused]] T const value{ *reinterpret_cast<T const *>(in_p) };
            using wide_t = typename calculation_t::calculation_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return T(std::sqrt(value));
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    wide_t const wide_value(wide_ops_t::load_unaligned(reinterpret_cast<wide_t const *>(in_p)));
                    return wide_ops.sqrt(wide_value);
                }
                else
                {
                    return T(std::sqrt(value));
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, log_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(log(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, log2_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(log2(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, log10_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(log10(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, exp_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(exp(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, exp2_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(exp2(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, cbrt_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(cbrt(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, tan_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(tan(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, cos_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(cos(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, sin_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return T(sin(value));
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, signbit_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return (std::is_signed_v<T> && T(value) < T{}) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
            else
            {
                return std::signbit(T(value)) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, not_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };

            return (not not (T(value) == T{})) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isnotnan_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type;
            using wide_t = typename calculation_t::calculation_type;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    wide_t const wide_value(wide_ops_t::load_unaligned(reinterpret_cast<wide_t const *>(in_p)));
                    return wide_ops.isnotnan(wide_value);
                }
                else
                {
                    T const value{ *reinterpret_cast<T const *>(in_p) };
                    return (not std::isnan(value)) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isnan_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using wide_t = typename calculation_t::calculation_type;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                if constexpr (wide_ops.simd_implemented_v)
                {
                    wide_t const wide_value(wide_ops_t::load_unaligned(reinterpret_cast<wide_t const *>(in_p)));
                    return wide_ops.isnan(wide_value);
                }
                else
                {
                    T const value{ *reinterpret_cast<T const *>(in_p) };
                    return std::isnan(value) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
                }
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isfinite_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                T const value{ *reinterpret_cast<T const *>(in_p) };
                return std::isfinite(value) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isnotfinite_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                T const value{ *reinterpret_cast<T const *>(in_p) };
                return (not std::isfinite(value)) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isinf_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                T const value{ *reinterpret_cast<T const *>(in_p) };
                return std::isinf(value) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isnotinf_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                T const value{ *reinterpret_cast<T const *>(in_p) };
                return (not std::isinf(value)) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isnormal_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                T const value{ *reinterpret_cast<T const *>(in_p) };
                return std::isnormal(value) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isnotnormal_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return bitmask_t{};
            }
            else
            {
                T const value{ *reinterpret_cast<T const *>(in_p) };
                return (not std::isnormal(value)) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        template <typename calculation_t, typename wide_ops_t>
        decltype(auto) calculate(char const * in_p, isnanorzero_op const * requested_op, calculation_t const * in_type,
                                 wide_ops_t wide_ops)
        {
            using T = typename calculation_t::data_type const;
            T const value{ *reinterpret_cast<T const *>(in_p) };
            using bitmask_t = typename calculation_t::bitmask_type;

            if constexpr (not std::is_floating_point_v<T> == true)
            {
                return (not not (T(value) == T{})) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
            else
            {
                return (not not (T(value) == T{}) || std::isnan(value)) ? std::numeric_limits<bitmask_t>::max() : bitmask_t{};
            }
        }

        // numpy standard is to treat stride as bytes
        template <typename operation_t, typename data_t>
        void perform_operation(char const * in_p, char * out_p, ptrdiff_t & starting_element, int64_t const in_array_stride,
                               size_t contig_elems, operation_t * op_p, data_t * data_type_p, int64_t out_stride_as_items = 1)
        {
            auto calc = [&](auto vectorization_object)
            {
                auto x = calculate(in_p, op_p, data_type_p, vectorization_object);

                if constexpr (sizeof(x) > sizeof(size_t))
                {
                    using vector_t = decltype(vectorization_object);
                    vectorization_object.store_unaligned(reinterpret_cast<typename vector_t::reg_type *>(out_p), x);
                }
                else
                {
                    *reinterpret_cast<decltype(x) *>(out_p) = x;
                }

                starting_element += sizeof(decltype(x)) / sizeof(typename data_t::data_type);
            };

            if (op_p)
            {
                if constexpr (operation_t::simd_implementation::value)
                {
                    using wide_sct = typename riptide::simd::avx2::template vec256<typename data_t::data_type>;
                    constexpr size_t wide_size{ sizeof(typename wide_sct::reg_type) };
                    constexpr size_t input_size{ sizeof(typename data_t::data_type) };
                    if (in_array_stride == sizeof(typename data_t::data_type) && out_stride_as_items == 1 &&
                        (wide_size / input_size) + starting_element < contig_elems)
                    {
                        calc(wide_sct{});
                    }
                    else
                    {
                        calc(no_simd_type{});
                    }
                }
                else
                {
                    calc(no_simd_type{});
                }
            }
        }

        template <typename operation_variant, typename data_type, size_t... Is>
        void calculate_for_active_operation(char const * in_p, char * out_p, ptrdiff_t & starting_element,
                                            int64_t const in_array_stride, size_t contig_elems,
                                            operation_variant const & requested_op, data_type const * type_p,
                                            std::index_sequence<Is...>)
        {
            if (type_p)
            {
                (perform_operation(in_p, out_p, starting_element, in_array_stride, contig_elems, std::get_if<Is>(&requested_op),
                                   type_p),
                 ...);
            }
        }

        template <typename type_variant, size_t... Is>
        void calculate_for_active_data_type(char const * in_p, char * out_p, ptrdiff_t & starting_element,
                                            int64_t const in_array_stride, size_t contig_elems, operation_t const & requested_op,
                                            type_variant const & in_type, std::index_sequence<Is...>)
        {
            (calculate_for_active_operation(in_p, out_p, starting_element, in_array_stride, contig_elems, requested_op,
                                            std::get_if<Is>(&in_type),
                                            std::make_index_sequence<std::variant_size_v<operation_t>>{}),
             ...);
        }

        template <typename operation_variant, size_t... Is>
        bool get_active_value_return(operation_variant v, std::index_sequence<Is...>)
        {
            return ((std::get_if<Is>(&v) ? std::get_if<Is>(&v)->value_return : false) || ...);
        }

        template <typename operation_trait, typename type_trait>
        void walk_data_array(ptrdiff_t inner_len, size_t outer_len, ptrdiff_t outer_stride, ptrdiff_t stride_out,
                             char const * in_p, char * out_p, operation_trait const & requested_op, type_trait const & in_type)
        {
            ptrdiff_t offset{};
            while (std::make_unsigned_t<ptrdiff_t>(offset) < outer_len)
            {
                calculate_for_active_data_type(in_p + (offset * outer_stride), out_p + (offset * inner_len * stride_out), offset,
                                               outer_stride, outer_len, requested_op, in_type,
                                               std::make_index_sequence<std::variant_size_v<data_type_t>>{});
            }
        }

        template <typename operation_trait, typename type_trait>
        void walk_row_major(char const * in_p, char * out_p, int32_t ndim, PyArrayObject const * in_array,
                            int64_t const stride_out, operation_trait const & requested_op, type_trait const & in_type)
        {
            ptrdiff_t inner_len{ 1 };
            for (int32_t i{ 1 }; i < ndim; ++i) // Is this loop really right? One-based but bounded by < ndim???
            {
                inner_len *= PyArray_DIM(in_array, i);
            }

            ptrdiff_t const outer_len = PyArray_DIM(in_array, 0);
            ptrdiff_t const outer_stride = PyArray_STRIDE(in_array, 0);

            walk_data_array(inner_len, outer_len, outer_stride, stride_out, in_p, out_p, requested_op, in_type);
        }

        template <typename operation_trait, typename type_trait>
        void walk_column_major(char const * in_p, char * out_p, int32_t ndim, PyArrayObject const * in_array,
                               int64_t const stride_out, operation_trait const & requested_op, type_trait const & in_type)
        {
            ptrdiff_t inner_len{ PyArray_DIM(in_array, 0) * PyArray_DIM(in_array, 1) };

            /*   This loop from UnaryOps.cpp is optimized to the above,
                 I'm unsure about it since it only looks at 2 dimensions,
                 and then we utilize at ndim below instead.
                 for( int32_t i{0}; i < 1; ++i )
                 {
                 inner_len *= PyArray_DIM( in_array, i );
                 }
            */

            ptrdiff_t const outer_len{ PyArray_DIM(in_array, (ndim - 1)) };
            ptrdiff_t const outer_stride{ PyArray_DIM(in_array, (ndim - 1)) };

            walk_data_array(inner_len, outer_len, outer_stride, stride_out, in_p, out_p, requested_op, in_type);
        }
    } // namespace implementation
} // namespace riptable_cpp
#endif
