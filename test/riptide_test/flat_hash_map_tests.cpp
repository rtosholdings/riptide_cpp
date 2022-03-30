#include "flat_hash_map.h"

#define BOOST_UT_DISABLE_MODULE
#include "boost/ut.hpp"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    std::array<int64_t, 1024ULL * 1024ULL> output;
    std::array<int8_t, 1024ULL * 1024ULL> bools;
    std::random_device dev{};
    std::mt19937 engine{ dev() };

    suite flat_hash_map_ops = []
    {
        "is_member_uint64"_test = [&]
        {
            std::vector<uint64_t> haystack(10);
            std::uniform_int_distribution<uint64_t> dist(0, ULLONG_MAX);

            std::generate(std::begin(haystack), std::end(haystack), [&] { return dist(engine); });
            std::vector<uint64_t> needles{ haystack[3], haystack[9], haystack[0], haystack[5] };

            expect(is_member(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                             reinterpret_cast<char const *>(haystack.data()), output.data(), bools.data(), 8) == 0_i);

            expect( output[ 0 ] == 3_i );
            expect( output[ 1 ] == 9_i );
            expect( output[ 2 ] == 0_i );
            expect( output[ 3 ] == 5_i );
        };

        "is_member_many_uniques"_test = [&]
        {
            std::vector<uint64_t> haystack(2 * 128 * 1024ULL * 1024ULL);
            std::vector<uint64_t> needles(1024ULL * 1024ULL);
            std::uniform_int_distribution<uint64_t> dist(3002950000, haystack.size() + 3002950000);

            std::iota(std::begin(haystack), std::end(haystack), 3002954000);
            std::generate(std::begin(needles), std::end(needles), [&] { return dist(engine); });

            expect(is_member(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                             reinterpret_cast<char const *>(haystack.data()), output.data(), bools.data(), 8) == 0_i);
        };
    };
}
