#pragma once
#include <cstdint>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <immintrin.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define RT_FORCEINLINE __forceinline
#else
#define RT_FORCEINLINE inline __attribute__((always_inline))
#endif   // defined(_MSC_VER) && !defined(__clang__)

/*
TODO
====
- More functions from <cmath>:
  https://en.cppreference.com/w/cpp/numeric/math
- More functions from <algorithm:
  https://en.cppreference.com/w/cpp/algorithm
- signbit()
- sgn()
- Floating-point classification functions (for float/double)
    - isfinite()
    - isinf()
    - isnormal()
- typical math operators
    - mul (though we may want overloads here for various overflow handling)
    - div
    - fmod
    - remainder
    - https://en.cppreference.com/w/c/language/operator_alternative
    - rcp
        - rcp_approx -- define this, but depending on the target architecture and element type,
          may just end up doing the real division instruction instead. Some newer hardware has
          a built-in instruction for doing an approximate reciprocal.
    - rsqrt
        - rsqrt_approx -- define this, but depending on the target architecture and element type,
          may just end up doing the real division instruction instead. Some newer hardware has
          a built-in instruction for doing an approximate reciprocal sqrt.

- putmask -- use _mm256_maskstore_pd/ps/epi32/epi64
- "as" and "to"/"convert" functions
    - "as" -- reinterpret the bytes (zero-cost) -- e.g. _mm256_castpd_si256()
    - "to"/"convert" -- actual conversion functions
        - e.g. int16 -> int8 can just use _mm256_unpacklo_epi8(_mm256_setzero_si256(), input)
        - e.g. _mm256_cvtepu16_epi32()
*/

namespace riptide
{
   namespace math
   {
      /**
       * @brief Function like @c std::min but which always propagates NaN values (for floating-point types).
       *
       * @tparam T The element type.
       * @param x The left element.
       * @param y The right element.
       * @return T const& The result of the operation.
       */
      template<typename T>
      static T const& min_with_nan_passthru(T const& x, T const& y)
      {
         return (std::min)(x, y);
      }

      template<>
      static float const& min_with_nan_passthru(float const& x, float const& y)
      {
         const auto blended = (x != x) ? x : y;
         return x < blended ? x : blended;
      }

      template<>
      static double const& min_with_nan_passthru(double const& x, double const& y)
      {
         const auto blended = (x != x) ? x : y;
         return x < blended ? x : blended;
      }

      /**
       * @brief Function like @c std::max but which always propagates NaN values (for floating-point types).
       *
       * @tparam T The element type.
       * @param x The left element.
       * @param y The right element.
       * @return T const& The result of the operation.
       */
      template<typename T>
      static T const& max_with_nan_passthru(T const& x, T const& y)
      {
         return (std::max)(x, y);
      }

      template<>
      static float const& max_with_nan_passthru(float const& x, float const& y)
      {
         const auto blended = (x != x) ? x : y;
         return x > blended ? x : blended;
      }

      template<>
      static double const& max_with_nan_passthru(double const& x, double const& y)
      {
         const auto blended = (x != x) ? x : y;
         return x > blended ? x : blended;
      }
   }

   namespace simd
   {
      namespace avx2
      {
         /**
          * @brief 256-bit vector operations templated on the element type.
          *
          * @tparam T A C++ primitive type.
          */
         template <typename T>
         struct vec256
         { };

         template <>
         struct vec256<bool>
         {
            using element_type = bool;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi8(static_cast<int8_t>(value));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Return a vector with all elements set to zero.
             *
             * @return reg_type The zeroed vector.
             */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }
         };

         template <>
         struct vec256<int8_t>
         {
            using element_type = int8_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type x)
            {
               return _mm256_abs_epi8(x);
            }

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type* const src)
            {
               return _mm256_abs_epi8(*src);
            }

            /**
             * @brief Pairwise addition of packed 8-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               return _mm256_add_epi8(x, y);
            }

            /**
             * @brief Pairwise addition of packed 8-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               return _mm256_add_epi8(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi8(value);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmpeq_epi8(x, y);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmpeq_epi8(x, *y);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type y)
            {
               return _mm256_cmpgt_epi8(x, y);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmpgt_epi8(x, *y);
            }

            ///**
            // * @brief Vectorized form of @c isgreaterequal.
            // *
            // * @param x The first vector.
            // * @param y The second vector.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type y)
            //{
            //   return _mm256_cmp_ph(x, y, _CMP_GE_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isgreaterequal.
            // *
            // * @param x The first vector.
            // * @param y A memory location to load the second vector from.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type* const y)
            //{
            //   return _mm256_cmp_ph(x, *y, _CMP_GE_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isless.
            // *
            // * @param x The first vector.
            // * @param y The second vector.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type y)
            //{
            //   return _mm256_cmp_ph(x, y, _CMP_LT_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isless.
            // *
            // * @param x The first vector.
            // * @param y A memory location to load the second vector from.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type* const y)
            //{
            //   return _mm256_cmp_ph(x, *y, _CMP_LT_OQ);
            //}

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmpgt_epi8(y, x);
            }

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x A memory location to load the first vector from.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type* const x, const reg_type y)
            {
               return _mm256_cmpgt_epi8(y, *x);
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type y)
            {
               // There's no "not equal" intrinsic for i8, so compare for equality, then compare the
               // comparison result to zero (effectively inverting the logic).
               return _mm256_cmpeq_epi8(
                  _mm256_setzero_si256(),
                  _mm256_cmpeq_epi8(x, y)
               );
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type* const y)
            {
               // There's no "not equal" intrinsic for i8, so compare for equality, then compare the
               // comparison result to zero (effectively inverting the logic).
               return _mm256_cmpeq_epi8(
                  _mm256_setzero_si256(),
                  _mm256_cmpeq_epi8(x, *y)
               );
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epi8(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epi8(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epi8(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epi8(x, *y);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type x)
            {
               return _mm256_sub_epi8(zeros(), x);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type* const src)
            {
               return _mm256_sub_epi8(zeros(), *src);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 8-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               return _mm256_sub_epi8(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 8-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               return _mm256_sub_epi8(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }

            /**
             * @brief Functions for getting a vector pre-populated with bitmasks needed for implementing certain functions.
             */
            struct bitmask
            {
               /**
                * @brief Return a vector where each component contains a bitmask which will
                *   remove the sign bit from an 8-bit signed integer value when ANDed together.
                *
                * @return reg_type The vector containing the constant 0x7f in each component.
                */
               static RT_FORCEINLINE reg_type abs()
               {
                  return _mm256_set1_epi8(0x7f);
               }
            };
         };

         template <>
         struct vec256<uint8_t>
         {
            using element_type = uint8_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Pairwise addition of packed 8-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi8(x, y);
            }

            /**
             * @brief Pairwise addition of packed 8-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi8(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi8(static_cast<int8_t>(value));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epu8(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epu8(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epu8(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epu8(x, *y);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 8-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi8(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 8-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi8(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }
         };

         template <>
         struct vec256<int16_t>
         {
            using element_type = int16_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type x)
            {
               return _mm256_abs_epi16(x);
            }

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type* const src)
            {
               return _mm256_abs_epi16(*src);
            }

            /**
             * @brief Pairwise addition of packed 16-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               return _mm256_add_epi16(x, y);
            }

            /**
             * @brief Pairwise addition of packed 16-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               return _mm256_add_epi16(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return __m256i The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi16(value);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmpeq_epi16(x, y);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmpeq_epi16(x, *y);
            }

            /**
             * @brief Get a vector of packed 16-bit integers representing the 0-based SIMD lane indices.
             *
             * @param x The input vector.
             * @return reg_type A vector containing packed integers {0, 1, ... 15}.
             * @remark Similar to the APL and C++ iota() function.
             */
            static RT_FORCEINLINE reg_type iota()
            {
               return _mm256_set_epi16(
                  15, 14, 13, 12,
                  11, 10, 9, 8,
                  7, 6, 5, 4,
                  3, 2, 1, 0);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type y)
            {
               return _mm256_cmpgt_epi16(x, y);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmpgt_epi16(x, *y);
            }

            ///**
            // * @brief Vectorized form of @c isgreaterequal.
            // *
            // * @param x The first vector.
            // * @param y The second vector.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type y)
            //{
            //   return _mm256_cmp_ph(x, y, _CMP_GE_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isgreaterequal.
            // *
            // * @param x The first vector.
            // * @param y A memory location to load the second vector from.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type* const y)
            //{
            //   return _mm256_cmp_ph(x, *y, _CMP_GE_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isless.
            // *
            // * @param x The first vector.
            // * @param y The second vector.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type y)
            //{
            //   return _mm256_cmp_ph(x, y, _CMP_LT_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isless.
            // *
            // * @param x The first vector.
            // * @param y A memory location to load the second vector from.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type* const y)
            //{
            //   return _mm256_cmp_ph(x, *y, _CMP_LT_OQ);
            //}

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmpgt_epi16(y, x);
            }

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x A memory location to load the first vector from.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type* const x, const reg_type y)
            {
               return _mm256_cmpgt_epi16(y, *x);
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type y)
            {
               // There's no "not equal" intrinsic for i16, so compare for equality, then compare the
               // comparison result to zero (effectively inverting the logic).
               return _mm256_cmpeq_epi16(
                  _mm256_setzero_si256(),
                  _mm256_cmpeq_epi16(x, y)
               );
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type* const y)
            {
               // There's no "not equal" intrinsic for i16, so compare for equality, then compare the
               // comparison result to zero (effectively inverting the logic).
               return _mm256_cmpeq_epi16(
                  _mm256_setzero_si256(),
                  _mm256_cmpeq_epi16(x, *y)
               );
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epi16(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epi16(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epi16(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epi16(x, *y);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type x)
            {
               return _mm256_sub_epi16(zeros(), x);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type* const src)
            {
               return _mm256_sub_epi16(zeros(), *src);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 16-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               return _mm256_sub_epi16(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 16-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               return _mm256_sub_epi16(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }

            /**
             * @brief Functions for getting a vector pre-populated with bitmasks needed for implementing certain functions.
             */
            struct bitmask
            {
               /**
                * @brief Return a vector where each component contains a bitmask which will
                *   remove the sign bit from a 16-bit signed integer value when ANDed together.
                *
                * @return reg_type The vector containing the constant 0x7fff in each component.
                */
               static RT_FORCEINLINE reg_type abs()
               {
                  // In all but a few cases it's faster to synthesize a vector with the constant we need here
                  // rather than loading it from memory (even if it's available in L1 cache), and this is also
                  // slightly faster than using the 'vbroadcastq' instruction (since you'd need to first get the
                  // constant into an __m128 first).
                  // The code below executes in 2-3 cycles.
                  // N.B. In a sense, this is trading space in the L1D cache for space in the L1I cache;
                  // but these instructions will be consecutive (in memory) with the code calling it so
                  // we're less likely to have to pay the cost of any cache misses (compared to loading the constant from memory).
                  const reg_type zeros = _mm256_setzero_si256();
                  const reg_type all_bits_set = _mm256_cmpeq_epi16(zeros, zeros);
                  return _mm256_srli_epi16(all_bits_set, 1);
               }
            };
         };

         template <>
         struct vec256<uint16_t>
         {
            using element_type = uint16_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Pairwise addition of packed 16-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi16(x, y);
            }

            /**
             * @brief Pairwise addition of packed 16-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi16(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi16(static_cast<int16_t>(value));
            }

            /**
             * @brief Get a vector of packed 16-bit integers representing the 0-based SIMD lane indices.
             *
             * @param x The input vector.
             * @return reg_type A vector containing packed integers {0, 1, ... 15}.
             * @remark Similar to the APL and C++ iota() function.
             */
            static RT_FORCEINLINE reg_type iota()
            {
               return _mm256_set_epi16(
                  15, 14, 13, 12,
                  11, 10, 9, 8,
                  7, 6, 5, 4,
                  3, 2, 1, 0);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epu16(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epu16(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epu16(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epu16(x, *y);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 16-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi16(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 16-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi16(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }
         };

         template <>
         struct vec256<int32_t>
         {
            using element_type = int32_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type x)
            {
               return _mm256_abs_epi32(x);
            }

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type* const src)
            {
               return _mm256_abs_epi32(*src);
            }

            /**
             * @brief Pairwise addition of packed 32-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               return _mm256_add_epi32(x, y);
            }

            /**
             * @brief Pairwise addition of packed 32-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               return _mm256_add_epi32(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi32(value);
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed, signed 32-bit integers whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type x, const reg_type y)
            {
               // There's no _mm256_blendv_epi32() intrinsic; however, since VBLENDVPS just looks at the sign bit
               // of each condition value (component of `cond`) and that's the same semantics we want here,
               // just use it to achieve what we're looking for.
               // NOTE: If this is subject to "data bypass delay" due to mixing integer and floating-point operations,
               // an alternative implementation that only costs 1 additional cycle (but no bypass delay, so it may be faster) is:
               //    return _mm256_blend_epi8(y, x, _mm256_cmpgt_epi32(cond, _mm256_setzero_pd()));

               // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPS are that the
               // the component from the second/right operand is taken when the condition is true, rather than
               // the first/left component as with a C-style ternary operator.
               return _mm256_castps_si256(_mm256_blendv_ps(
                  _mm256_castsi256_ps(y),
                  _mm256_castsi256_ps(x),
                  _mm256_castsi256_ps(cond)
               ));
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed, signed 32-bit integers whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The memory location of a vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type* const x, const reg_type y)
            {
               // There's no _mm256_blendv_epi32() intrinsic; however, since VBLENDVPS just looks at the sign bit
               // of each condition value (component of `cond`) and that's the same semantics we want here,
               // just use it to achieve what we're looking for.
               // NOTE: If this is subject to "data bypass delay" due to mixing integer and floating-point operations,
               // an alternative implementation that only costs 1 additional cycle (but no bypass delay, so it may be faster) is:
               //    return _mm256_blend_epi8(y, x, _mm256_cmpgt_epi32(cond, _mm256_setzero_pd()));

               // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPS are that the
               // the component from the second/right operand is taken when the condition is true, rather than
               // the first/left component as with a C-style ternary operator.
               return _mm256_castps_si256(_mm256_blendv_ps(
                  _mm256_castsi256_ps(y),
                  _mm256_castsi256_ps(*x),
                  _mm256_castsi256_ps(cond)
               ));
            }

            /**
             * @brief Get a vector of packed 32-bit integers representing the 0-based SIMD lane indices.
             *
             * @param x The input vector.
             * @return reg_type A vector containing packed integers {0, 1, ... 7}.
             * @remark Similar to the APL and C++ iota() function.
             */
            static RT_FORCEINLINE reg_type iota()
            {
               return _mm256_set_epi32(
                  7, 6, 5, 4,
                  3, 2, 1, 0);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmpeq_epi32(x, y);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmpeq_epi32(x, *y);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type y)
            {
               return _mm256_cmpgt_epi32(x, y);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmpgt_epi32(x, *y);
            }

            ///**
            // * @brief Vectorized form of @c isgreaterequal.
            // *
            // * @param x The first vector.
            // * @param y The second vector.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type y)
            //{
            //   return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isgreaterequal.
            // *
            // * @param x The first vector.
            // * @param y A memory location to load the second vector from.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type* const y)
            //{
            //   return _mm256_cmp_ps(x, *y, _CMP_GE_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isless.
            // *
            // * @param x The first vector.
            // * @param y The second vector.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type y)
            //{
            //   return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
            //}

            ///**
            // * @brief Vectorized form of @c isless.
            // *
            // * @param x The first vector.
            // * @param y A memory location to load the second vector from.
            // * @return reg_type A vector containing the results of the pairwise comparisons.
            // */
            //static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type* const y)
            //{
            //   return _mm256_cmp_ps(x, *y, _CMP_LT_OQ);
            //}

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmpgt_epi32(y, x);
            }

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x A memory location to load the first vector from.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type* const x, const reg_type y)
            {
               return _mm256_cmpgt_epi32(y, *x);
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type y)
            {
               // There's no "not equal" intrinsic for i32, so compare for equality, then compare the
               // comparison result to zero (effectively inverting the logic).
               return _mm256_cmpeq_epi32(
                  _mm256_setzero_si256(),
                  _mm256_cmpeq_epi32(x, y)
               );
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type* const y)
            {
               // There's no "not equal" intrinsic for i32, so compare for equality, then compare the
               // comparison result to zero (effectively inverting the logic).
               return _mm256_cmpeq_epi32(
                  _mm256_setzero_si256(),
                  _mm256_cmpeq_epi32(x, *y)
               );
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epi32(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epi32(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epi32(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epi32(x, *y);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type x)
            {
               return _mm256_sub_epi32(zeros(), x);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type* const src)
            {
               return _mm256_sub_epi32(zeros(), *src);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 32-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               return _mm256_sub_epi32(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 32-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               return _mm256_sub_epi32(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }

            /**
             * @brief Functions for getting a vector pre-populated with bitmasks needed for implementing certain functions.
             */
            struct bitmask
            {
               /**
                * @brief Return a vector where each component contains a bitmask which will
                *   remove the sign bit from a 32-bit signed integer value when ANDed together.
                *
                * @return reg_type The vector containing the constant 0x7fffffff in each component.
                */
               static RT_FORCEINLINE reg_type abs()
               {
                  // In all but a few cases it's faster to synthesize a vector with the constant we need here
                  // rather than loading it from memory (even if it's available in L1 cache), and this is also
                  // slightly faster than using the 'vbroadcastq' instruction (since you'd need to first get the
                  // constant into an __m128 first).
                  // The code below executes in 2-3 cycles.
                  // N.B. In a sense, this is trading space in the L1D cache for space in the L1I cache;
                  // but these instructions will be consecutive (in memory) with the code calling it so
                  // we're less likely to have to pay the cost of any cache misses (compared to loading the constant from memory).
                  const reg_type zeros = _mm256_setzero_si256();
                  const reg_type all_bits_set = _mm256_cmpeq_epi32(zeros, zeros);
                  return _mm256_srli_epi32(all_bits_set, 1);
               }
            };
         };

         template <>
         struct vec256<uint32_t>
         {
            using element_type = uint32_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Pairwise addition of packed 32-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi32(x, y);
            }

            /**
             * @brief Pairwise addition of packed 32-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi32(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitxor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi32(static_cast<int32_t>(value));
            }

            /**
             * @brief Get a vector of packed 32-bit integers representing the 0-based SIMD lane indices.
             *
             * @param x The input vector.
             * @return reg_type A vector containing packed integers {0, 1, ... 7}.
             * @remark Similar to the APL and C++ iota() function.
             */
            static RT_FORCEINLINE reg_type iota()
            {
               return _mm256_set_epi32(
                  7, 6, 5, 4,
                  3, 2, 1, 0);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epu32(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epu32(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epu32(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epu32(x, *y);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 32-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi32(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 32-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi32(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }
         };

         template <>
         struct vec256<int64_t>
         {
            using element_type = int64_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type x)
            {
               // _mm256_abs_epi64() only available when targeting AVX512, and it requires both
               // AVX512VL + AVX512F. If these aren't available on the target hardware, synthesize
               // the operation instead.
#if defined(__AVX512VL__) && defined(__AVX512F__)
               return _mm256_abs_epi64(x);
#else
               return bitwise_and(bitmask::abs(), x);
#endif
            }

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type* const src)
            {
               // _mm256_abs_epi64() only available when targeting AVX512, and it requires both
               // AVX512VL + AVX512F. If these aren't available on the target hardware, synthesize
               // the operation instead.
#if defined(__AVX512VL__) && defined(__AVX512F__)
               return _mm256_abs_epi64(*src);
#else
               return bitwise_and(bitmask::abs(), *src);
#endif
            }

            /**
             * @brief Pairwise addition of packed 64-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               return _mm256_add_epi64(x, y);
            }

            /**
             * @brief Pairwise addition of packed 64-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               return _mm256_add_epi64(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_xor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_xor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi64x(value);
            }

            /**
             * @brief Get a vector of packed 64-bit integers representing the 0-based SIMD lane indices.
             *
             * @param x The input vector.
             * @return reg_type A vector containing packed integers {0, 1, 2, 3}.
             * @remark Similar to the APL and C++ iota() function.
             */
            static RT_FORCEINLINE reg_type iota()
            {
               return _mm256_set_epi64x(
                  3, 2, 1, 0);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epi64(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epi64(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epi64(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epi64(x, *y);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type x)
            {
               return _mm256_sub_epi64(zeros(), x);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type* const src)
            {
               return _mm256_sub_epi64(zeros(), *src);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 64-bit signed integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               return _mm256_sub_epi64(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 64-bit signed integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               return _mm256_sub_epi64(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }

            /**
             * @brief Functions for getting a vector pre-populated with bitmasks needed for implementing certain functions.
             */
            struct bitmask
            {
               /**
                * @brief Return a vector where each component contains a bitmask which will
                *   remove the sign bit from a 64-bit signed integer value when ANDed together.
                *
                * @return reg_type The vector containing the constant 0x7fffffffffffffff in each component.
                */
               static RT_FORCEINLINE reg_type abs()
               {
                  // In all but a few cases it's faster to synthesize a vector with the constant we need here
                  // rather than loading it from memory (even if it's available in L1 cache), and this is also
                  // slightly faster than using the 'vbroadcastq' instruction (since you'd need to first get the
                  // constant into an __m128 first).
                  // The code below executes in 2-3 cycles.
                  // N.B. In a sense, this is trading space in the L1D cache for space in the L1I cache;
                  // but these instructions will be consecutive (in memory) with the code calling it so
                  // we're less likely to have to pay the cost of any cache misses (compared to loading the constant from memory).
                  const reg_type zeros = _mm256_setzero_si256();
                  const reg_type all_bits_set = _mm256_cmpeq_epi64(zeros, zeros);
                  return _mm256_srli_epi64(all_bits_set, 1);
               }
            };
         };

         template <>
         struct vec256<uint64_t>
         {
            using element_type = uint64_t;

            using reg_type = __m256i;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Pairwise addition of packed 64-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi64(x, y);
            }

            /**
             * @brief Pairwise addition of packed 64-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_add_epi64(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type y)
            {
               return _mm256_and_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-AND values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-AND values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_and(const reg_type x, const reg_type* const y)
            {
               return _mm256_and_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type y)
            {
               return _mm256_or_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-OR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-OR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_or(const reg_type x, const reg_type* const y)
            {
               return _mm256_or_si256(x, *y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_xor(const reg_type x, const reg_type y)
            {
               return _mm256_xor_si256(x, y);
            }

            /**
             * @brief Compute the pairwise bitwise-XOR values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise bitwise-XOR values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type bitwise_xor(const reg_type x, const reg_type* const y)
            {
               return _mm256_xor_si256(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return __m256i The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(const element_type value) {
               return _mm256_set1_epi64x(static_cast<int64_t>(value));
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed, signed 64-bit integers whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type x, const reg_type y)
            {
               // There's no _mm256_blendv_epi64() intrinsic; however, since VBLENDVPD just looks at the sign bit
               // of each condition value (component of `cond`) and that's the same semantics we want here,
               // just use it to achieve what we're looking for.
               // NOTE: If this is subject to "data bypass delay" due to mixing integer and floating-point operations,
               // an alternative implementation that only costs 1 additional cycle (but no bypass delay, so it may be faster) is:
               //    return _mm256_blend_epi8(y, x, _mm256_cmpgt_epi64(cond, _mm256_setzero_pd()));
               // Or, use _mm256_cmpeq_epi64() to invert the conditions too, then the arguments to blendv can be passed in the same order as for this function's params;
               // that doesn't matter much here but might be more natural for the overload accepting the memory pointer for one of the operands.

               // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPD are that the
               // the component from the second/right operand is taken when the condition is true, rather than
               // the first/left component as with a C-style ternary operator.
               return _mm256_castpd_si256(_mm256_blendv_pd(
                  _mm256_castsi256_pd(y),
                  _mm256_castsi256_pd(x),
                  _mm256_castsi256_pd(cond)
               ));
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed, signed 64-bit integers whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The memory location of a vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type* const x, const reg_type y)
            {
               // There's no _mm256_blendv_epi64() intrinsic; however, since VBLENDVPD just looks at the sign bit
               // of each condition value (component of `cond`) and that's the same semantics we want here,
               // just use it to achieve what we're looking for.
               // NOTE: If this is subject to "data bypass delay" due to mixing integer and floating-point operations,
               // an alternative implementation that only costs 1 additional cycle (but no bypass delay, so it may be faster) is:
               //    return _mm256_blend_epi8(y, x, _mm256_cmpgt_epi64(cond, _mm256_setzero_pd()));

               // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPD are that the
               // the component from the second/right operand is taken when the condition is true, rather than
               // the first/left component as with a C-style ternary operator.
               return _mm256_castpd_si256(_mm256_blendv_pd(
                  _mm256_castsi256_pd(y),
                  _mm256_castsi256_pd(*x),
                  _mm256_castsi256_pd(cond)
               ));
            }

            /**
             * @brief Get a vector of packed 64-bit integers representing the 0-based SIMD lane indices.
             *
             * @param x The input vector.
             * @return reg_type A vector containing packed integers {0, 1, 2, 3}.
             * @remark Similar to the APL and C++ iota() function.
             */
            static RT_FORCEINLINE reg_type iota()
            {
               return _mm256_set_epi64x(
                  3, 2, 1, 0);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_si256(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_si256(reinterpret_cast<const reg_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_si256(src);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y) {
               return _mm256_max_epu64(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type* const y) {
               return _mm256_max_epu64(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return __m256i A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y) {
               return _mm256_min_epu64(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return __m256i A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type* const y) {
               return _mm256_min_epu64(x, *y);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_si256(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(reinterpret_cast<reg_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_si256(dest, value);
            }

            /**
             * @brief Pairwise subtraction of packed 64-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi64(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed 64-bit unsigned integers.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               // Same intrinsic used as for signed integers; the resulting bits are the same but
               // just need to be interpreted as being unsigned integers.
               return _mm256_sub_epi64(x, *y);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return __m256i The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_si256();
            }
         };

         template <>
         struct vec256<float>
         {
            using element_type = float;

            using reg_type = __m256;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type x)
            {
               const reg_type mask = bitmask::abs();
               return _mm256_and_ps(mask, x);
            }

            /**
             * @brief Compute the absolute value of a vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the per-element absolute values of the input vector.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type* const src)
            {
               const reg_type mask = bitmask::abs();
               return _mm256_and_ps(mask, *src);
            }

            /**
             * @brief Pairwise addition of packed single-precision floating-point values.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               return _mm256_add_ps(x, y);
            }

            /**
             * @brief Pairwise addition of packed single-precision floating-point values.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               return _mm256_add_ps(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(element_type value) {
               return _mm256_set1_ps(value);
            }

            /**
             * @brief Vectorized form of @c ceil.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c ceil.
             */
            static RT_FORCEINLINE reg_type ceil(const reg_type x)
            {
               return _mm256_ceil_ps(x);
            }

            /**
             * @brief Vectorized form of @c ceil.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c ceil.
             */
            static RT_FORCEINLINE reg_type ceil(const reg_type* const x)
            {
               return _mm256_ceil_ps(*x);
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed floats whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type x, const reg_type y)
            {
               // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPD are that the
               // the component from the second/right operand is taken when the condition is true, rather than
               // the first/left component as with a C-style ternary operator.
               return _mm256_blendv_ps(y, x, cond);
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed floats whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The memory location of a vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type* const x, const reg_type y)
            {
               // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPD are that the
               // the component from the second/right operand is taken when the condition is true, rather than
               // the first/left component as with a C-style ternary operator.
               return _mm256_blendv_ps(y, *x, cond);
            }

            /**
             * @brief Vectorized form of @c floor.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c floor.
             */
            static RT_FORCEINLINE reg_type floor(const reg_type x)
            {
               return _mm256_floor_ps(x);
            }

            /**
            * @brief Vectorized form of @c floor.
            *
            * @param src A memory location to load an input vector from.
            * @return reg_type A vector containing the elementwise results of @c floor.
            */
            static RT_FORCEINLINE reg_type floor(const reg_type* const x)
            {
               return _mm256_floor_ps(*x);
            }

            /**
             * @brief The C @c fmax function for a vector of packed floats.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type
             * @note In C, this function might be called something like '_mm256_fmax_ps' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmax(const reg_type x, const reg_type y)
            {
               const reg_type max_result = _mm256_max_ps(x, y);
               const reg_type unord_cmp_mask = _mm256_cmp_ps(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_ps(x, y, unord_cmp_mask);
            }

            /**
             * @brief The C @c fmax function for a vector of packed floats.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type
             * @note In C, this function might be called something like '_mm256_fmax_ps' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmax(const reg_type x, const reg_type* const y)
            {
               const reg_type max_result = _mm256_max_ps(x, *y);
               const reg_type unord_cmp_mask = _mm256_cmp_ps(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_ps(x, *y, unord_cmp_mask);
            }

            /**
             * @brief The C @c fmin function for a vector of packed floats.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type
             * @note In C, this function might be called something like '_mm256_fmin_ps' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmin(const reg_type x, const reg_type y)
            {
               const reg_type max_result = _mm256_min_ps(x, y);
               const reg_type unord_cmp_mask = _mm256_cmp_ps(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_ps(x, y, unord_cmp_mask);
            }

            /**
             * @brief The C @c fmin function for a vector of packed floats.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type
             * @note In C, this function might be called something like '_mm256_fmin_ps' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmin(const reg_type x, const reg_type* const y)
            {
               const reg_type max_result = _mm256_min_ps(x, *y);
               const reg_type unord_cmp_mask = _mm256_cmp_ps(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_ps(x, *y, unord_cmp_mask);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_ps(x, *y, _CMP_EQ_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_ps(x, y, _CMP_GT_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_ps(x, *y, _CMP_GT_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreaterequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreaterequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_ps(x, *y, _CMP_GE_OQ);
            }

            /**
             * @brief Vectorized form of @c isless.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
            }

            /**
             * @brief Vectorized form of @c isless.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_ps(x, *y, _CMP_LT_OQ);
            }

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_ps(x, y, _CMP_LE_OQ);
            }

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_ps(x, *y, _CMP_LE_OQ);
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_ps(x, y, _CMP_NEQ_OQ);
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_ps(x, *y, _CMP_NEQ_OQ);
            }

            /**
             * @brief Vectorized form of @c isnan.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c isnan.
             */
            static RT_FORCEINLINE reg_type isnan(const reg_type x)
            {
               return _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
            }

            /**
             * @brief Vectorized form of @c !isnan.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c !isnan.
             */
            static RT_FORCEINLINE reg_type isnotnan(const reg_type x)
            {
               return _mm256_cmp_ps(x, x, _CMP_ORD_Q);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_ps(src);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_ps(reinterpret_cast<const element_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_ps(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_ps(reinterpret_cast<const element_type*>(src));
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors. A NaN is returned (in a vector element)
             *    when either or both of the values in the input pair are NaN.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             * @note This implementation uses the revised semantics for the 'maximum' function in
             *    IEEE754-2019 w.r.t. NaN-propagation. The @c _mm256_max_ps intrinsic follows the semantics
             *    from older revisions of IEEE754 and only propagates NaNs in certain cases, which is
             *    problematic when using it to implement a reduction operation expected to propagate NaNs.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y)
            {
               // Create a mask indicating which components of 'x', if any, are NaNs.
               const reg_type unord_cmp_mask = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

               // Use the blendv operation with our "x is nan" mask to create a new vector
               // containing the components of 'y' where 'x' was _not_ a NaN, and the components
               // of 'x' which were NaNs.
               // TODO: If we invert the comparison above (to use _CMP_ORD_Q), can we swap the order of the operands
               //       here? That'd let us have an overload of this method which accepts 'y' as a memory pointer.
               const reg_type blended_nans = _mm256_blendv_ps(y, x, unord_cmp_mask);

               // Perform the max operation on 'x' and the blended vector.
               // maxpd / vmaxpd will take the component from the 2nd argument whenever one of
               // the components (from the 1st and 2nd arguments) is NaN. Putting the blended
               // vector we created on the r.h.s. means we'll get a NaN component in the output
               // vector for any index (within the vector) where _either_ the left or right side was NaN.
               return _mm256_max_ps(x, blended_nans);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max_raw(const reg_type x, const reg_type y) {
               return _mm256_max_ps(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max_raw(const reg_type x, const reg_type* const y) {
               return _mm256_max_ps(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors. A NaN is returned (in a vector element)
             *    when either or both of the values in the input pair are NaN.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             * @note This implementation uses the revised semantics for the 'minimum' function in
             *    IEEE754-2019 w.r.t. NaN-propagation. The @c _mm256_min_ps intrinsic follows the semantics
             *    from older revisions of IEEE754 and only propagates NaNs in certain cases, which is
             *    problematic when using it to implement a reduction operation expected to propagate NaNs.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y)
            {
               // Create a mask indicating which components of 'x', if any, are NaNs.
               const reg_type unord_cmp_mask = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

               // Use the blendv operation with our "x is nan" mask to create a new vector
               // containing the components of 'y' where 'x' was _not_ a NaN, and the components
               // of 'x' which were NaNs.
               // TODO: If we invert the comparison above (to use _CMP_ORD_Q), can we swap the order of the operands
               //       here? That'd let us have an overload of this method which accepts 'y' as a memory pointer.
               const reg_type blended_nans = _mm256_blendv_ps(y, x, unord_cmp_mask);

               // Perform the min operation on 'x' and the blended vector.
               // minpd / vminpd will take the component from the 2nd argument whenever one of
               // the components (from the 1st and 2nd arguments) is NaN. Putting the blended
               // vector we created on the r.h.s. means we'll get a NaN component in the output
               // vector for any index (within the vector) where _either_ the left or right side was NaN.
               return _mm256_min_ps(x, blended_nans);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min_raw(const reg_type x, const reg_type y) {
               return _mm256_min_ps(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min_raw(const reg_type x, const reg_type* const y) {
               return _mm256_min_ps(x, *y);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type x)
            {
               return _mm256_sub_ps(zeros(), x);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type* const src)
            {
               return _mm256_sub_ps(zeros(), *src);
            }

            /**
             * @brief Vectorized form of @c round.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c round.
             */
            static RT_FORCEINLINE reg_type round(const reg_type x)
            {
               return _mm256_round_ps(x, _MM_FROUND_NINT);
            }

            /**
             * @brief Vectorized form of @c round.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c round.
             */
            static RT_FORCEINLINE reg_type round(const reg_type* const x)
            {
               return _mm256_round_ps(*x, _MM_FROUND_NINT);
            }

            /**
             * @brief Vectorized form of @c sqrt.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c sqrt.
             */
            static RT_FORCEINLINE reg_type sqrt(const reg_type x)
            {
               return _mm256_sqrt_ps(x);
            }

            /**
             * @brief Vectorized form of @c sqrt.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c sqrt.
             */
            static RT_FORCEINLINE reg_type sqrt(const reg_type* const x)
            {
               return _mm256_sqrt_ps(*x);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_ps(dest, value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_ps(reinterpret_cast<element_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_ps(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_ps(reinterpret_cast<element_type*>(dest), value);
            }

            /**
             * @brief Pairwise subtraction of packed single-precision floating-point values.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               return _mm256_sub_ps(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed single-precision floating-point values.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               return _mm256_sub_ps(x, *y);
            }

            /**
             * @brief Vectorized form of @c trunc.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c trunc.
             */
            static RT_FORCEINLINE reg_type trunc(const reg_type x)
            {
               return _mm256_round_ps(x, _MM_FROUND_TRUNC);
            }

            /**
             * @brief Vectorized form of @c trunc.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c trunc.
             */
            static RT_FORCEINLINE reg_type trunc(const reg_type* const x)
            {
               return _mm256_round_ps(*x, _MM_FROUND_TRUNC);
            }

            /**
            * @brief Return a vector with all elements set to zero.
            *
            * @return reg_type The zeroed vector.
            */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_ps();
            }

            /**
             * @brief Functions for getting a vector pre-populated with bitmasks needed for implementing certain functions.
             */
            struct bitmask
            {
               /**
                * @brief Return a vector where each component contains a bitmask which will
                *   remove the sign bit from an IEEE754 32-bit floating point value when ANDed together.
                *
                * @return reg_type The vector containing the constant 0x7fffffff in each component.
                */
               static RT_FORCEINLINE reg_type abs()
               {
                  // In all but a few cases it's faster to synthesize a vector with the constant we need here
                  // rather than loading it from memory (even if it's available in L1 cache), and this is also
                  // slightly faster than using the 'vbroadcastq' instruction (since you'd need to first get the
                  // constant into an __m128 first).
                  // The code below executes in 2-3 cycles.
                  // N.B. In a sense, this is trading space in the L1D cache for space in the L1I cache;
                  // but these instructions will be consecutive (in memory) with the code calling it so
                  // we're less likely to have to pay the cost of any cache misses (compared to loading the constant from memory).
                  const vec256<int32_t>::reg_type zeros = _mm256_setzero_si256();
                  const vec256<int32_t>::reg_type all_bits_set = _mm256_cmpeq_epi32(zeros, zeros);
                  return _mm256_castsi256_ps(_mm256_srli_epi32(all_bits_set, 1));
               }

               /**
                * @brief TODO
                *
                * @return reg_type The vector containing the constant 0x7f800000 in each component.
                */
               static RT_FORCEINLINE reg_type finite_compare()
               {
                  return _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
               }

               /**
                * @brief Return a vector where each component contains a bitmask which will
                *   remove the sign bit from an IEEE754 32-bit floating point value when ANDed together.
                *
                * @return reg_type The vector containing the constant 0x7fffffff in each component.
                */
               static RT_FORCEINLINE reg_type finite_mask()
               {
                  // In all but a few cases it's faster to synthesize a vector with the constant we need here
                  // rather than loading it from memory (even if it's available in L1 cache), and this is also
                  // slightly faster than using the 'vbroadcastq' instruction (since you'd need to first get the
                  // constant into an __m128 first).
                  // The code below executes in 2-3 cycles.
                  // N.B. In a sense, this is trading space in the L1D cache for space in the L1I cache;
                  // but these instructions will be consecutive (in memory) with the code calling it so
                  // we're less likely to have to pay the cost of any cache misses (compared to loading the constant from memory).
                  const vec256<int32_t>::reg_type zeros = _mm256_setzero_si256();
                  const vec256<int32_t>::reg_type all_bits_set = _mm256_cmpeq_epi32(zeros, zeros);
                  return _mm256_castsi256_ps(_mm256_srli_epi32(all_bits_set, 1));
               }

               /**
                * @brief TODO
                *
                * @return reg_type The vector containing the constant 0x007fffff in each component.
                */
               static RT_FORCEINLINE reg_type inf()
               {
                  // all 1 bits in exponent must be 1 (11 bits after sign)
                  // and fraction must not be 0
                  const vec256<int32_t>::reg_type zeros = _mm256_setzero_si256();
                  const vec256<int32_t>::reg_type all_bits_set = _mm256_cmpeq_epi32(zeros, zeros);
                  return _mm256_castsi256_ps(_mm256_srli_epi32(all_bits_set, 9));
               }
            };
         };

         template <>
         struct vec256<double>
         {
            using element_type = double;

            using reg_type = __m256d;

            /**
             * @brief The number of elements in each vector.
             */
            static constexpr size_t element_count = sizeof(reg_type) / sizeof(element_type);

            /**
             * @brief Vectorized form of @c abs.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c abs.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type x)
            {
               const reg_type mask = bitmask::abs();
               return _mm256_and_pd(mask, x);
            }

            /**
             * @brief Vectorized form of @c abs.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c abs.
             */
            static RT_FORCEINLINE reg_type abs(const reg_type* const src)
            {
               const reg_type mask = bitmask::abs();
               return _mm256_and_pd(mask, *src);
            }

            /**
             * @brief Pairwise addition of packed double-precision floating-point values.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type y)
            {
               return _mm256_add_pd(x, y);
            }

            /**
             * @brief Pairwise addition of packed double-precision floating-point values.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise sums of the two input vectors.
             */
            static RT_FORCEINLINE reg_type add(const reg_type x, const reg_type* const y)
            {
               return _mm256_add_pd(x, *y);
            }

            /**
             * @brief Return a vector with all elements set to the specified scalar value.
             *
             * @param value The value to broadcast to all elements of a vector.
             * @return reg_type The broadcasted vector.
             */
            static RT_FORCEINLINE reg_type broadcast(element_type value) {
               return _mm256_set1_pd(value);
            }

            /**
             * @brief Vectorized form of @c ceil.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c ceil.
             */
            static RT_FORCEINLINE reg_type ceil(const reg_type x)
            {
               return _mm256_ceil_pd(x);
            }

            /**
             * @brief Vectorized form of @c ceil.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c ceil.
             */
            static RT_FORCEINLINE reg_type ceil(const reg_type* const src)
            {
               return _mm256_ceil_pd(*src);
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed doubles whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type x, const reg_type y)
            {
                // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPD are that the
                // the component from the second/right operand is taken when the condition is true, rather than
                // the first/left component as with a C-style ternary operator.
               return _mm256_blendv_pd(y, x, cond);
            }

            /**
             * @brief Vectorized ternary operator. For each vector lane/component, implements 'cond ? x : y'.
             *
             * @param cond A vector of packed doubles whose sign bits are interpreted as boolean values for the ternary operation.
             * @param x The memory location of a vector to take elements from where @p cond is true.
             * @param y The vector to take elements from where @p cond is false.
             * @return reg_type A vector containing the results of the ternary operations.
             */
            static RT_FORCEINLINE reg_type condition(const reg_type cond, const reg_type* const x, const reg_type y)
            {
               // Swap the order of y and x when calling the intrinsic. The semantics of VBLENDVPD are that the
               // the component from the second/right operand is taken when the condition is true, rather than
               // the first/left component as with a C-style ternary operator.
               return _mm256_blendv_pd(y, *x, cond);
            }

            /**
             * @brief Vectorized form of @c floor.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c floor.
             */
            static RT_FORCEINLINE reg_type floor(const reg_type x)
            {
               return _mm256_floor_pd(x);
            }

            /**
            * @brief Vectorized form of @c floor.
            *
            * @param src A memory location to load an input vector from.
            * @return reg_type A vector containing the elementwise results of @c floor.
            */
            static RT_FORCEINLINE reg_type floor(const reg_type* const src)
            {
               return _mm256_floor_pd(*src);
            }

            /**
             * @brief The C @c fmax function for a vector of packed doubles.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type 
             * @note In C, this function might be called something like '_mm256_fmax_pd' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmax(const reg_type x, const reg_type y)
            {
               const reg_type max_result = _mm256_max_pd(x, y);
               const reg_type unord_cmp_mask = _mm256_cmp_pd(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_pd(x, y, unord_cmp_mask);
            }

            /**
             * @brief The C @c fmax function for a vector of packed doubles.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type
             * @note In C, this function might be called something like '_mm256_fmax_pd' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmax(const reg_type x, const reg_type* const y)
            {
               const reg_type max_result = _mm256_max_pd(x, *y);
               const reg_type unord_cmp_mask = _mm256_cmp_pd(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_pd(x, *y, unord_cmp_mask);
            }

            /**
             * @brief The C @c fmin function for a vector of packed doubles.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type
             * @note In C, this function might be called something like '_mm256_fmin_pd' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmin(const reg_type x, const reg_type y)
            {
               const reg_type max_result = _mm256_min_pd(x, y);
               const reg_type unord_cmp_mask = _mm256_cmp_pd(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_pd(x, y, unord_cmp_mask);
            }

            /**
             * @brief The C @c fmin function for a vector of packed doubles.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type
             * @note In C, this function might be called something like '_mm256_fmin_pd' to adhere to the
             *    naming conventions used by the other AVX intrinsics.
            */
            static RT_FORCEINLINE reg_type fmin(const reg_type x, const reg_type* const y)
            {
               const reg_type max_result = _mm256_min_pd(x, *y);
               const reg_type unord_cmp_mask = _mm256_cmp_pd(max_result, max_result, _CMP_UNORD_Q);
               return _mm256_blendv_pd(x, *y, unord_cmp_mask);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
            }

            /**
             * @brief Vectorized form of @c isequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_pd(x, *y, _CMP_EQ_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_pd(x, y, _CMP_GT_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreater(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_pd(x, *y, _CMP_GT_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreaterequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_pd(x, y, _CMP_GE_OQ);
            }

            /**
             * @brief Vectorized form of @c isgreaterequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isgreaterequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_pd(x, *y, _CMP_GE_OQ);
            }

            /**
             * @brief Vectorized form of @c isless.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_pd(x, y, _CMP_LT_OQ);
            }

            /**
             * @brief Vectorized form of @c isless.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isless(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_pd(x, *y, _CMP_LT_OQ);
            }

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_pd(x, y, _CMP_LE_OQ);
            }

            /**
             * @brief Vectorized form of @c islessequal.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessequal(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_pd(x, *y, _CMP_LE_OQ);
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_pd(x, y, _CMP_NEQ_OQ);
            }

            /**
             * @brief Vectorized form of @c islessgreater.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type islessgreater(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_pd(x, *y, _CMP_NEQ_OQ);
            }

            /**
             * @brief Vectorized form of @c isnan.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c isnan.
             */
            static RT_FORCEINLINE reg_type isnan(const reg_type x)
            {
               return _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
            }

            /**
             * @brief Vectorized form of @c !isnan.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c !isnan.
             */
            static RT_FORCEINLINE reg_type isnotnan(const reg_type x)
            {
               return _mm256_cmp_pd(x, x, _CMP_ORD_Q);
            }

            /**
             * @brief Vectorized form of @c isunordered.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isunordered(const reg_type x, const reg_type y)
            {
               return _mm256_cmp_pd(x, y, _CMP_UNORD_Q);
            }

            /**
             * @brief Vectorized form of @c isunordered.
             *
             * @param x The first vector.
             * @param y A memory location to load the second vector from.
             * @return reg_type A vector containing the results of the pairwise comparisons.
             */
            static RT_FORCEINLINE reg_type isunordered(const reg_type x, const reg_type* const y)
            {
               return _mm256_cmp_pd(x, *y, _CMP_UNORD_Q);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const element_type* const src)
            {
               return _mm256_load_pd(src);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_aligned(const reg_type* const src)
            {
               return _mm256_load_pd(reinterpret_cast<const element_type*>(src));
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const element_type* const src)
            {
               return _mm256_loadu_pd(src);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param src The source address to load the vector data from.
             * @return reg_type A vector containing the data loaded from memory.
             */
            static RT_FORCEINLINE reg_type load_unaligned(const reg_type* const src)
            {
               return _mm256_loadu_pd(reinterpret_cast<const element_type*>(src));
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors. A NaN is returned (in a vector element)
             *    when either or both of the values in the input pair are NaN.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             * @note This implementation uses the revised semantics for the 'maximum' function in
             *    IEEE754-2019 w.r.t. NaN-propagation. The @c _mm256_max_pd intrinsic follows the semantics
             *    from older revisions of IEEE754 and only propagates NaNs in certain cases, which is
             *    problematic when using it to implement a reduction operation expected to propagate NaNs.
             */
            static RT_FORCEINLINE reg_type max(const reg_type x, const reg_type y)
            {
               // Create a mask indicating which components of 'x', if any, are NaNs.
               const reg_type unord_cmp_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

               // Use the blendv operation with our "x is nan" mask to create a new vector
               // containing the components of 'y' where 'x' was _not_ a NaN, and the components
               // of 'x' which were NaNs.
               // TODO: If we invert the comparison above (to use _CMP_ORD_Q), can we swap the order of the operands
               //       here? That'd let us have an overload of this method which accepts 'y' as a memory pointer.
               const reg_type blended_nans = _mm256_blendv_pd(y, x, unord_cmp_mask);

               // Perform the max operation on 'x' and the blended vector.
               // maxpd / vmaxpd will take the component from the 2nd argument whenever one of
               // the components (from the 1st and 2nd arguments) is NaN. Putting the blended
               // vector we created on the r.h.s. means we'll get a NaN component in the output
               // vector for any index (within the vector) where _either_ the left or right side was NaN.
               return _mm256_max_pd(x, blended_nans);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max_raw(const reg_type x, const reg_type y)
            {
               return _mm256_max_pd(x, y);
            }

            /**
             * @brief Compute the pairwise maximum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise maximum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type max_raw(const reg_type x, const reg_type* const y)
            {
               return _mm256_max_pd(x, *y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors. A NaN is returned (in a vector element)
             *    when either or both of the values in the input pair are NaN.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             * @note This implementation uses the revised semantics for the 'minimum' function in
             *    IEEE754-2019 w.r.t. NaN-propagation. The @c _mm256_min_pd intrinsic follows the semantics
             *    from older revisions of IEEE754 and only propagates NaNs in certain cases, which is
             *    problematic when using it to implement a reduction operation expected to propagate NaNs.
             */
            static RT_FORCEINLINE reg_type min(const reg_type x, const reg_type y)
            {
               // Create a mask indicating which components of 'x', if any, are NaNs.
               const reg_type unord_cmp_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);

               // Use the blendv operation with our "x is nan" mask to create a new vector
               // containing the components of 'y' where 'x' was _not_ a NaN, and the components
               // of 'x' which were NaNs.
               // TODO: If we invert the comparison above (to use _CMP_ORD_Q), can we swap the order of the operands
               //       here? That'd let us have an overload of this method which accepts 'y' as a memory pointer.
               const reg_type blended_nans = _mm256_blendv_pd(y, x, unord_cmp_mask);

               // Perform the min operation on 'x' and the blended vector.
               // minpd / vminpd will take the component from the 2nd argument whenever one of
               // the components (from the 1st and 2nd arguments) is NaN. Putting the blended
               // vector we created on the r.h.s. means we'll get a NaN component in the output
               // vector for any index (within the vector) where _either_ the left or right side was NaN.
               return _mm256_min_pd(x, blended_nans);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min_raw(const reg_type x, const reg_type y)
            {
               return _mm256_min_pd(x, y);
            }

            /**
             * @brief Compute the pairwise minimum values of two vectors.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise minimum values from the two input vectors.
             */
            static RT_FORCEINLINE reg_type min_raw(const reg_type x, const reg_type* const y)
            {
               return _mm256_min_pd(x, *y);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param x The input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type x)
            {
               return _mm256_sub_pd(zeros(), x);
            }

            /**
             * @brief Negate each value in the input vector.
             *
             * @param src A pointer to the input vector.
             * @return reg_type A vector containing the negated values of the input vector.
             */
            static RT_FORCEINLINE reg_type negate(const reg_type* const src)
            {
               return _mm256_sub_pd(zeros(), *src);
            }

            /**
             * @brief Vectorized form of @c round.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c round.
             */
            static RT_FORCEINLINE reg_type round(const reg_type x)
            {
               return _mm256_round_pd(x, _MM_FROUND_NINT);
            }

            /**
             * @brief Vectorized form of @c round.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c round.
             */
            static RT_FORCEINLINE reg_type round(const reg_type* const x)
            {
               return _mm256_round_pd(*x, _MM_FROUND_NINT);
            }

            /**
             * @brief Vectorized form of @c sqrt.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c sqrt.
             */
            static RT_FORCEINLINE reg_type sqrt(const reg_type x)
            {
               return _mm256_sqrt_pd(x);
            }

            /**
             * @brief Vectorized form of @c sqrt.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c sqrt.
             */
            static RT_FORCEINLINE reg_type sqrt(const reg_type* const x)
            {
               return _mm256_sqrt_pd(*x);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(element_type* const dest, const reg_type value)
            {
               _mm256_store_pd(dest, value);
            }

            /**
             * @brief Read a vector from a memory location aligned to the size of the vector.
             * A memory fault is triggered if the source location is unaligned.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_aligned(reg_type* const dest, const reg_type value)
            {
               _mm256_store_pd(reinterpret_cast<element_type*>(dest), value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(element_type* const dest, const reg_type value)
            {
               _mm256_storeu_pd(dest, value);
            }

            /**
             * @brief Read a vector from a (possibly) unaligned memory location.
             *
             * @param dest The destination address where the data will be written.
             * @param value A vector containing data to write to memory.
             */
            static RT_FORCEINLINE void store_unaligned(reg_type* const dest, const reg_type value)
            {
               _mm256_storeu_pd(reinterpret_cast<element_type*>(dest), value);
            }

            /**
             * @brief Pairwise subtraction of packed double-precision floating-point values.
             *
             * @param x The first vector.
             * @param y The second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type y)
            {
               return _mm256_sub_pd(x, y);
            }

            /**
             * @brief Pairwise subtraction of packed double-precision floating-point values.
             *
             * @param x The first vector.
             * @param y A pointer to the second vector.
             * @return reg_type A vector containing the pairwise differences of the two input vectors.
             */
            static RT_FORCEINLINE reg_type sub(const reg_type x, const reg_type* const y)
            {
               return _mm256_sub_pd(x, *y);
            }

            /**
             * @brief Vectorized form of @c trunc.
             *
             * @param x A vector.
             * @return reg_type A vector containing the elementwise results of @c trunc.
             */
            static RT_FORCEINLINE reg_type trunc(const reg_type x)
            {
               return _mm256_round_pd(x, _MM_FROUND_TRUNC);
            }

            /**
             * @brief Vectorized form of @c trunc.
             *
             * @param src A memory location to load an input vector from.
             * @return reg_type A vector containing the elementwise results of @c trunc.
             */
            static RT_FORCEINLINE reg_type trunc(const reg_type* const src)
            {
               return _mm256_round_pd(*src, _MM_FROUND_TRUNC);
            }

            /**
             * @brief Return a vector with all elements set to zero.
             *
             * @return reg_type The zeroed vector.
             */
            static RT_FORCEINLINE reg_type zeros() {
               return _mm256_setzero_pd();
            }

            /**
             * @brief Functions for getting a vector pre-populated with bitmasks needed for implementing certain functions.
             */
            struct bitmask
            {
                /**
                 * @brief Return a vector where each component contains a bitmask which will
                 *   remove the sign bit from an IEEE754 64-bit floating point value when ANDed together.
                 *
                 * @return reg_type The vector containing the constant 0x7fffffffffffffff in each component.
                 */
                static RT_FORCEINLINE reg_type abs()
                {
                    // In all but a few cases it's faster to synthesize a vector with the constant we need here
                    // rather than loading it from memory (even if it's available in L1 cache), and this is also
                    // slightly faster than using the 'vbroadcastq' instruction (since you'd need to first get the
                    // constant into an __m128 first).
                    // The code below executes in 2-3 cycles.
                    // N.B. In a sense, this is trading space in the L1D cache for space in the L1I cache;
                    // but these instructions will be consecutive (in memory) with the code calling it so
                    // we're less likely to have to pay the cost of any cache misses (compared to loading the constant from memory).
                    const vec256<int64_t>::reg_type zeros = _mm256_setzero_si256();
                    const vec256<int64_t>::reg_type all_bits_set = _mm256_cmpeq_epi64(zeros, zeros);
                    return _mm256_castsi256_pd(_mm256_srli_epi64(all_bits_set, 1));
                }

                /**
                 * @brief TODO
                 *
                 * @return reg_type The vector containing the constant 0x7ff0000000000000 in each component.
                 */
                static RT_FORCEINLINE reg_type finite_compare()
                {
                   return _mm256_castsi256_pd(_mm256_set1_epi64x(0x7ff0000000000000));
                }

                /**
                 * @brief Return a vector where each component contains a bitmask which will
                 *   remove the sign bit from an IEEE754 64-bit floating point value when ANDed together.
                 *
                 * @return reg_type The vector containing the constant 0x7fffffffffffffff in each component.
                 */
                static RT_FORCEINLINE reg_type finite_mask()
                {
                   // In all but a few cases it's faster to synthesize a vector with the constant we need here
                   // rather than loading it from memory (even if it's available in L1 cache), and this is also
                   // slightly faster than using the 'vbroadcastq' instruction (since you'd need to first get the
                   // constant into an __m128 first).
                   // The code below executes in 2-3 cycles.
                   // N.B. In a sense, this is trading space in the L1D cache for space in the L1I cache;
                   // but these instructions will be consecutive (in memory) with the code calling it so
                   // we're less likely to have to pay the cost of any cache misses (compared to loading the constant from memory).
                   const vec256<int64_t>::reg_type zeros = _mm256_setzero_si256();
                   const vec256<int64_t>::reg_type all_bits_set = _mm256_cmpeq_epi64(zeros, zeros);
                   return _mm256_castsi256_pd(_mm256_srli_epi64(all_bits_set, 1));
                }

                /**
                 * @brief TODO
                 *
                 * @return reg_type The vector containing the constant 0x000fffffffffffff in each component.
                 */
                static RT_FORCEINLINE reg_type inf()
                {
                   // all 1 bits in exponent must be 1 (11 bits after sign)
                   // and fraction must not be 0
                   const vec256<int64_t>::reg_type zeros = _mm256_setzero_si256();
                   const vec256<int64_t>::reg_type all_bits_set = _mm256_cmpeq_epi64(zeros, zeros);
                   return _mm256_castsi256_pd(_mm256_srli_epi64(all_bits_set, 12));
                }
            };
         };
      }
   }
}
