#include "simd.h"
#include "ut_core.h"

#include <random>

using namespace riptide;
using namespace boost::ut;
using riptide_utility::ut::file_suite;

namespace
{
    using supported_types = std::tuple<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double>;

    file_suite simd_tests = []
    {
        "contains_invalid"_test = []<typename T>
        {
            std::vector<T> input(100000, T{});
            // Should return false when there are no invalids
            expect(! riptide::simd::contains_invalid(std::span(input.begin(), input.end())));
            // Shuffle in some invalids
            std::fill(input.begin(), input.begin() + 13, riptide::invalid_for_type<T>::value);
            std::shuffle(input.begin(), input.end(), std::mt19937{ 42 });
            // Should return true when there are invalids
            expect(riptide::simd::contains_invalid(std::span(input.begin(), input.end())));
        } | supported_types{};
    };
}