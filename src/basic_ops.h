#ifndef RIPTIDECPP_BASIC_OPS_H
#define RIPTIDECPP_BASIC_OPS_H

#include "operation_traits.h"

#include <array>

// Some missing instrinsics
#define _mm_roundme_ps(val) _mm256_round_ps((val), _MM_FROUND_NINT)
#define _mm_roundme_pd(val) _mm256_round_pd((val), _MM_FROUND_NINT)
#define _mm_truncme_ps(val) _mm256_round_ps((val), _MM_FROUND_TRUNC)
#define _mm_truncme_pd(val) _mm256_round_pd((val), _MM_FROUND_TRUNC)

namespace internal
{
    // Rewrite this if pattern matching ever becomes a thing.
    auto LOADU = [](auto const * x) -> std::remove_cv_t<std::remove_reference_t<std::remove_pointer<decltype(x)>>>
    {
        using underlying_t = std::remove_cv_t<std::remove_reference_t<std::remove_pointer<decltype(x)>>>;

        if constexpr (std::is_same_v<__m256d, underlying_t>)
        {
            return _mm256_loadu_pd(reinterpret_cast<double const *>(x));
        }

        if constexpr (std::is_same_v<__m256, underlying_t>)
        {
            return _mm256_loadu_ps(reinterpret_cast<float const *>(x));
        }

        if constexpr (std::is_same_v<__m256i, underlying_t>)
        {
            return _mm256_loadu_si256(x);
        }

        throw(std::runtime_error("Attempt to load an illegal unaligned SIMD type"));
    };

    auto STOREU = [](auto const * x, auto const y) -> void
    {
        if constexpr (std::is_same_v<__m256d const, y>)
        {
            _mm256_storeu_pd((double *)x, y);
        }
        if constexpr (std::is_same_v<__m256 const, y>)
        {
            _mm256_storeu_ps((float *)x, y);
        }
        if constexpr (std::is_same_v<__m256i const, y>)
        {
            _mm256_storeu_si256(x, y);
        }

        throw(std::runtime_error("Attempt to store an illegal unaligned SIMD type"));
    };

    //// Examples of how to store a constant in vector math
    // SFW_ALIGN(64)
    //__m256  __ones_constant32f = _mm256_set1_ps(1.0f);
    //__m256d __ones_constant64f = _mm256_set1_pd(1.0);
    //__m256i __ones_constant64i = _mm256_set1_epi64x(1);

    // This bit mask will remove the sign bit from an IEEE floating point and is how
    // ABS values are done
    SFW_ALIGN(64)
    constexpr union
    {
        int32_t i[8];
        float f[8];
        __m256 m;
    } __f32vec8_abs_mask = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

    SFW_ALIGN(64)
    constexpr union
    {
        int64_t i[4];
        double d[4];
        __m256d m;
    } __f64vec4_abs_mask = { 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff };

    SFW_ALIGN(64)
    constexpr union
    {
        int32_t i[8];
        float f[8];
        __m256 m;
        // all 1 bits in exponent must be 1 (8 bits after sign)
        // and fraction must not be 0
    } __f32vec8_finite_compare = {
        0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000
    };

    SFW_ALIGN(64)
    constexpr union
    {
        int32_t i[8];
        float f[8];
        __m256 m;
        // all 1 bits in exponent must be 1 (8 bits after sign)
        // and fraction must not be 0
    } __f32vec8_finite_mask = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

    SFW_ALIGN(64)
    constexpr union
    {
        int32_t i[8];
        float f[8];
        __m256 m;
        // all 1 bits in exponent must be 1 (8 bits after sign)
        // and fraction must not be 0
    } __f32vec8_inf_mask = { 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff };

    SFW_ALIGN(64)
    constexpr union
    {
        int64_t i[4];
        double d[4];
        __m256d m;
        // all 1 bits in exponent must be 1 (11 bits after sign)
        // and fraction must not be 0
    } __f64vec4_finite_mask = { 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff };

    SFW_ALIGN(64)
    constexpr union
    {
        int64_t i[4];
        double d[4];
        __m256d m;
        // all 1 bits in exponent must be 1 (11 bits after sign)
        // and fraction must not be 0
    } __f64vec4_finite_compare = { 0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000, 0x7ff0000000000000 };

    SFW_ALIGN(64)
    constexpr union
    {
        int64_t i[4];
        double d[4];
        __m256d m;
        // all 1 bits in exponent must be 1 (11 bits after sign)
        // and fraction must not be 0
    } __f64vec4_inf_mask = { 0x000fffffffffffff, 0x000fffffffffffff, 0x000fffffffffffff, 0x000fffffffffffff };

    // This is used to multiply the strides
    SFW_ALIGN(64)
    constexpr union
    {
        int32_t i[8];
        __m256i m;
    } __vec8_strides = { 0, 1, 2, 3, 4, 5, 6, 7 };

    // This is used to multiply the strides
    SFW_ALIGN(64)
    constexpr union
    {
        int64_t i[8];
        __m256i m;
    } __vec4_strides = { 0, 1, 2, 3 };

    //// IEEE Mask
    //// NOTE: Check NAN mask -- if not then return number, else return 0.0 or +INF
    /// or -INF / For IEEE 754, MSB is the sign bit, then next section is the
    /// exponent.  If the exponent is all 1111s, it is some kind of NAN
    //#define NAN_TO_NUM_F32(x) ((((*(uint32_t*)&x)  & 0x7f800000) != 0x7f800000) ?
    // x :  (((*(uint32_t*)&x)  & 0x007fffff) != 0) ? 0.0f : (((*(uint32_t*)&x)  &
    // 0x80000000) == 0) ?  FLT_MAX : -FLT_MAX) #define NAN_TO_NUM_F64(x)
    // ((((*(uint64_t*)&x)  & 0x7ff0000000000000) != 0x7ff0000000000000) ?  x :
    // (((*(uint64_t*)&x)  & 0x000fffffffffffff) != 0) ? 0.0 : (((*(uint64_t*)&x) &
    // 0x8000000000000000) == 0) ?  DBL_MAX : -DBL_MAX )
    //
    //#define NAN_TO_ZERO_F32(x) ((((*(uint32_t*)&x)  & 0x7f800000) != 0x7f800000) ?
    // x :   0.0f ) #define NAN_TO_ZERO_F64(x) ((((*(uint64_t*)&x)  &
    // 0x7ff0000000000000) != 0x7ff0000000000000) ?  x : 0.0 )
    //
} // namespace internal
#endif
