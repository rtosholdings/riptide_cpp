#include "riptide_python_test.h"

#include "HashLinear.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <exception>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    std::array<int64_t, 1024ULL * 1024ULL> output;
    std::array<int8_t, 1024ULL * 1024ULL> bools;
    
    suite hash_linear_ops = []
    {
        "ismember64_uint64_too_many_uniques"_test = [&]
        {
            std::vector<uint64_t> haystack(2 * 128 * 1024ULL * 1024ULL);
            std::vector<uint64_t> needles(1024ULL * 1024ULL);
            std::random_device dev{};
            std::mt19937 engine(dev());
            std::uniform_int_distribution<uint64_t> dist(3002950000, haystack.size() + 3002950000);
            
            std::iota(std::begin(haystack), std::end(haystack), 3002954000);
            std::generate(std::begin(needles), std::end(needles), [&]{return dist(engine);});
    
            expect(throws<std::runtime_error>(
                       [&]
                       {
                           // Note the reversed order of haystack and needles in this call - more unique needles than uniques in haystack
                           IsMemberHash64(haystack.size(), haystack.data(), needles.size(), needles.data(), output.data(), bools.data(), 8, HASH_MODE(1), 0);
                       }
                       ));
        };
    };
}