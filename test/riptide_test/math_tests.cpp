#include "CommonInc.h"
#include "RipTide.h"
#include "rt_math.h"

#include "ut_extensions.h"

using namespace boost::ut;
using riptide_utility::ut::file_suite;

namespace
{
    // Bools have no invalid (though there is inconsistent support for such),
    // All bools are treated as valid, so must special case the tests.
    template <typename T>
    constexpr bool is_bool_v = std::is_same_v<bool, std::remove_cvref_t<T>>;

    using SupportedTypes =
        std::tuple<bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, long double>;

    struct min_with_nan_passthru_tester
    {
        template <typename T>
        void operator()()
        {
            auto const invalid{ riptide::invalid_for_type<T>::value };
            T const valid{ 0 };

            {
                auto const result{ riptide::math::min_with_nan_passthru(invalid, valid) };
                typed_expect<T>(is_bool_v<T> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "invalid,valid";
            }
            {
                auto const result{ riptide::math::min_with_nan_passthru(valid, invalid) };
                typed_expect<T>(is_bool_v<T> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "valid,invalid";
            }
        }
    };

    struct max_with_nan_passthru_tester
    {
        template <typename T>
        void operator()()
        {
            auto const invalid{ riptide::invalid_for_type<T>::value };
            T const valid{ 0 };

            {
                auto const result{ riptide::math::max_with_nan_passthru(invalid, valid) };
                typed_expect<T>(is_bool_v<T> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "invalid,valid";
            }
            {
                auto const result{ riptide::math::max_with_nan_passthru(valid, invalid) };
                typed_expect<T>(is_bool_v<T> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "valid,invalid";
            }
        }
    };

    file_suite reduce_ops = []
    {
        "min_with_nan_passthru"_test = min_with_nan_passthru_tester{} | SupportedTypes{};
        "max_with_nan_passthru"_test = max_with_nan_passthru_tester{} | SupportedTypes{};
    };
}