#include "CommonInc.h"
#include "RipTide.h"
#include "missing_values.h"

#include "add_typed_tests.h"
#include "to_type_str.h"

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
            // Floatint sentinals must be NaNs, which are never comparable,
            // and may not even be bit comparable (happens with GCC and long double!)
            return isnan(t1) && isnan(t2);
        }
        else
        {
            return t1 == t2;
        }
    }

    template <typename T>
    void same_getinvalid_invalid_test()
    {
        T const * gotten_invalid{ static_cast<T *>(GetInvalid<T>()) };
        expect((gotten_invalid) >> fatal) << "Failed null for: " << to_type_str<T>();
        T const * type_invalid{ &invalid_for_type<T>::value };
        bool const ok{ compare_invalids(*gotten_invalid, *type_invalid) };
        expect(ok) << "Failed for: " << to_type_str<T>();
    }
    DECL_ADD_TYPED_TESTS(same_getinvalid_invalid_test)

    template <typename T>
    void same_get_invalid_invalid_test()
    {
        T const gotten_invalid{ GET_INVALID(T{}) };
        T const type_invalid{ invalid_for_type<T>::value };
        bool const ok{ compare_invalids(gotten_invalid, type_invalid) };
        expect(ok) << "Failed for: " << to_type_str<T>();
    }
    DECL_ADD_TYPED_TESTS(same_get_invalid_invalid_test)

    template <typename T>
    void invalid_sentinel_not_valid_test()
    {
        T const * gotten_invalid{ reinterpret_cast<T *>(GetInvalid<T>()) };
        expect((gotten_invalid) >> fatal) << "Failed null for: " << to_type_str<T>();
        bool const actual{ ! invalid_for_type<T>::is_valid(*gotten_invalid) };
        // invalid_for_type<bool> doesn't treat any bool as invalid. Probably never will.
        bool const expected{ std::conditional_t<std::is_same_v<T, bool>, std::false_type, std::true_type>::value };
        expect(actual == expected) << "Failed for: " << to_type_str<T>();
    }
    DECL_ADD_TYPED_TESTS(invalid_sentinel_not_valid_test)

    suite invalids_compatibility = []
    {
        ADD_TYPED_TESTS(same_getinvalid_invalid_test)("same_getinvalid_invalid"_test, SupportedTypes{});
        ADD_TYPED_TESTS(same_get_invalid_invalid_test)("same_get_invalid_invalid"_test, SupportedTypes{});
        ADD_TYPED_TESTS(invalid_sentinel_not_valid_test)("invalid_sentinel_not_valid"_test, SupportedTypes{});
    };
}
