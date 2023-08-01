#include "CommonInc.h"
#include "RipTide.h"
#include "missing_values.h"

#include "ut_extensions.h"

#include "boost/ut.hpp"

#include <cmath>
#include <type_traits>

using namespace riptide;
using namespace boost::ut;
using boost::ut::suite;

namespace
{
    using SupportedTypes =
        std::tuple<bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, long double>;

    template <typename T>
    std::enable_if_t<std::is_arithmetic_v<T>, bool> compare_invalids(T const & t1, T const & t2)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            // Floating sentinels must be NaNs, which are never comparable,
            // and may not even be bit comparable (happens with GCC and long double!)
            return isnan(t1) && isnan(t2);
        }
        else
        {
            return t1 == t2;
        }
    }

    struct same_getinvalid_invalid_tester
    {
        template <typename T>
        void operator()()
        {
            T const * gotten_invalid{ static_cast<T *>(GetInvalid<T>()) };
            typed_expect<T>((gotten_invalid) >> fatal);
            T const * type_invalid{ &invalid_for_type<T>::value };
            bool const ok{ compare_invalids(*gotten_invalid, *type_invalid) };
            typed_expect<T>(ok);
        }
    };

    struct same_get_invalid_invalid_tester
    {
        template <typename T>
        void operator()()
        {
            T const gotten_invalid{ GET_INVALID(T{}) };
            T const type_invalid{ invalid_for_type<T>::value };
            bool const ok{ compare_invalids(gotten_invalid, type_invalid) };
            typed_expect<T>(ok);
        }
    };

    struct invalid_sentinel_not_valid_tester
    {
        template <typename T>
        void operator()()
        {
            bool const actual{ ! invalid_for_type<T>::is_valid(invalid_for_type<T>::value) };
            // invalid_for_type<bool> doesn't treat any bool as invalid. Probably never will.
            bool const expected{ std::conditional_t<std::is_same_v<T, bool>, std::false_type, std::true_type>::value };
            typed_expect<T>(actual == expected);
        }
    };

    struct valid_is_valid_tester
    {
        template <typename T>
        void operator()()
        {
            T const valid{ 0 }; // zero is valid for all types
            bool const actual{ invalid_for_type<T>::is_valid(valid) };
            typed_expect<T>(actual == true);
        }
    };

    struct getinvalid_sentinel_not_valid_tester
    {
        template <typename T>
        void operator()()
        {
            T const * gotten_invalid{ reinterpret_cast<T *>(GetInvalid<T>()) };
            typed_expect<T>((gotten_invalid) >> fatal);
            bool const actual{ ! invalid_for_type<T>::is_valid(*gotten_invalid) };
            // invalid_for_type<bool> doesn't treat any bool as invalid. Probably never will.
            bool const expected{ std::conditional_t<std::is_same_v<T, bool>, std::false_type, std::true_type>::value };
            typed_expect<T>(actual == expected);
        }
    };

    suite invalids_compatibility = []
    {
        "same_getinvalid_invalid"_test = same_getinvalid_invalid_tester{} | SupportedTypes{};
        "same_get_invalid_invalid"_test = same_get_invalid_invalid_tester{} | SupportedTypes{};
        "invalid_sentinel_not_valid"_test = invalid_sentinel_not_valid_tester{} | SupportedTypes{};
        "getinvalid_sentinel_not_valid"_test = getinvalid_sentinel_not_valid_tester{} | SupportedTypes{};
        "valid_is_valid"_test = valid_is_valid_tester{} | SupportedTypes{};
    };
}
