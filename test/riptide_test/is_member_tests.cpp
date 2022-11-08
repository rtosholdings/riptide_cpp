#include "is_member_tg.h"
#include "HashLinear.h"

#include "boost/ut.hpp"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <tuple>
#include <type_traits>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    std::array<int64_t, 1024ULL * 1024ULL> output;
    std::array<int32_t, 1024ULL * 1024ULL> small_output;
    std::array<int8_t, 1024ULL * 1024ULL> bools;

    suite is_member_ops = []
    {
        "is_member_uint64_tbb"_test = [&]
        {
            std::random_device dev{};
            uint32_t seed{ dev() };
            boost::ut::log << "is_member_uint_tbb random seed is " << seed << "\n";
            std::mt19937 engine{ seed };
            std::vector<uint64_t> haystack(10);
            std::uniform_int_distribution<uint64_t> dist(0, ULLONG_MAX - 2);

            std::generate(std::begin(haystack), std::end(haystack),
                          [&]
                          {
                              return dist(engine);
                          });
            std::vector<uint64_t> needles{ haystack[3], haystack[9], haystack[0], haystack[5], ULLONG_MAX - 1 };

            is_member_tg(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                         reinterpret_cast<char const *>(haystack.data()), 1, output.data(), bools.data(), needles[0]);

            expect(output[0] == 3_i);
            expect(output[1] == 9_i);
            expect(output[2] == 0_i);
            expect(output[3] == 5_i);
            expect(output[4] < -9'223'372'036'854'775'807_ll);

            expect(static_cast<int32_t>(bools[0]) == 1_i);
            expect(static_cast<int32_t>(bools[1]) == 1_i);
            expect(static_cast<int32_t>(bools[2]) == 1_i);
            expect(static_cast<int32_t>(bools[3]) == 1_i);
            expect(static_cast<int32_t>(bools[4]) == 0_i);
        };

        "pytest_lookalike"_test = [&]
        {
            std::vector<int8_t> needles{ 1, 2, 3, 4, 5 };
            std::vector<int8_t> haystack{ 1, 2 };

            std::vector<int8_t> results(16);

            is_member_tg(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                         reinterpret_cast<char const *>(haystack.data()), 1, results.data(), bools.data(), haystack[0]);

            expect(results[0] == 0_i);
            expect(results[1] == 1_i);

            expect(bools[0] == 1_i);
            expect(bools[1] == 1_i);
            expect(bools[2] == 0_i);
            expect(bools[3] == 0_i);
            expect(bools[4] == 0_i);
        };

        "multiple_ismember_calls"_test = [&]
        {
            std::vector<int> needles1{ 10, 8, 5 };

            std::vector<int> haystack1{ 5, 8, 10 };

            std::vector<int8_t> results(8);

            is_member_tg(needles1.size(), reinterpret_cast<char const *>(needles1.data()), haystack1.size(),
                         reinterpret_cast<char const *>(haystack1.data()), 1, results.data(), bools.data(), haystack1[0]);

            expect(bools[0] == 1_i);
            expect(bools[1] == 1_i);
            expect(bools[2] == 1_i);

            expect(results[0] == 2_i);
            expect(results[1] == 1_i);
            expect(results[2] == 0_i);

            std::vector<int> haystack2{ 5 };

            is_member_tg(needles1.size(), reinterpret_cast<char const *>(needles1.data()), haystack2.size(),
                         reinterpret_cast<char const *>(haystack2.data()), 1, results.data(), bools.data(), haystack2[0]);

            expect(bools[0] == 0_i);
            expect(bools[1] == 0_i);
            expect(bools[2] == 1_i);

            expect(results[0] == -128_i);
            expect(results[1] == -128_i);
            expect(results[2] == 0_i);
        };

        "zero_as_a_needle"_test = [&]
        {
            std::vector<uint64_t> haystack{ 42, 19, 0, 512, 65535 };
            std::vector<uint64_t> needles{ 65535, 55, 0, 19 };

            std::vector<int8_t> results(4);

            is_member_tg(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                         reinterpret_cast<char const *>(haystack.data()), 1, results.data(), bools.data(), haystack[0]);

            expect(bools[0] == 1_i);
            expect(bools[1] == 0_i);
            expect(bools[2] == 1_i);
            expect(bools[3] == 1_i);

            expect(results[0] == 4_i);
            expect(results[1] == -128_i);
            expect(results[2] == 2_i);
            expect(results[3] == 1_i);
        };

        "is_member_find_lots_tbb"_test = [&]
        {
            std::random_device dev{};
            uint32_t seed{ dev() };
            boost::ut::log << "is_member_find_lots_tbb random seed is " << seed << "\n";
            std::mt19937 engine{ seed };
            std::vector<uint64_t> haystack(2ULL * 1024ULL * 1024ULL);
            std::vector<uint64_t> needles(32ULL * 1024ULL);
            std::uniform_int_distribution<uint64_t> dist(0, haystack.size() - 1);

            std::iota(std::begin(haystack), std::end(haystack), 0);
            std::generate(std::begin(needles), std::end(needles),
                          [&]
                          {
                              return haystack[dist(engine)];
                          });

            is_member_tg(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                         reinterpret_cast<char const *>(haystack.data()), 1, output.data(), bools.data(), needles[0]);

            for (size_t i{ 0 }; i != needles.size(); ++i)
            {
                expect(bools[i] == 1_l) << "expected result at index " << i << ", loooking for " << needles[i] << "\n";
            }
        };

        "is_member_call_many_times_tbb"_test = [&]
        {
            std::vector<int32_t> needles_sizes{ 127, 129, 254, 256 };
            std::vector<int32_t> needles{};
            for (int32_t needle_size : needles_sizes)
            {
                needles.resize(needle_size);
                std::iota(std::begin(needles), std::end(needles), 1);
                std::vector<int32_t> haystack{};
                for (int32_t haystack_size{ 1 }; haystack_size != 130; ++haystack_size)
                {
                    haystack.resize(haystack_size);
                    std::iota(std::begin(haystack), std::end(haystack), 1);
                    expect(nothrow(
                        [&]()
                        {
                            is_member_tg(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                                         reinterpret_cast<char const *>(haystack.data()), 1, output.data(), bools.data(),
                                         needles[0]);
                        }));

                    for (int32_t i{ 0 }; i != needle_size; ++i)
                    {
                        expect(output[i] == (static_cast<size_t>(i) < haystack.size() ? i : std::numeric_limits<int64_t>::min()))
                            << "saw " << output[i] << " but wanted " << i << "\n";
                    }
                }
            }
        };

        skip / "is_member_many_uniques_tbb"_test = [&]
        {
            std::random_device dev{};
            uint32_t seed{ dev() };
            boost::ut::log << "is_member_many_uniques_tbb random seed is " << seed << "\n";
            std::mt19937 engine{ seed };
            std::vector<uint64_t> haystack(2 * 1024ULL * 1024ULL);
            std::vector<uint64_t> needles(1024ULL * 1024ULL);
            std::uniform_int_distribution<uint64_t> dist(3002950000, haystack.size() + 3002950000);

            std::iota(std::begin(haystack), std::end(haystack), 3002954000);
            std::generate(std::begin(needles), std::end(needles),
                          [&]
                          {
                              return dist(engine);
                          });

            expect(nothrow(
                [&]()
                {
                    is_member_tg(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                                 reinterpret_cast<char const *>(haystack.data()), 1, output.data(), bools.data(), needles[0]);
                }));
        };

        skip / "ismember64_uint64_too_many_uniques"_test = [&]
        {
            std::random_device dev{};
            uint32_t seed{ dev() };
            boost::ut::log << "ismember64_uint64_too_many_uniques random seed is " << seed << "\n";
            std::mt19937 engine{ seed };
            std::vector<uint64_t> haystack(128ULL * 1024ULL * 1024ULL);
            std::vector<uint64_t> needles(1024ULL * 1024ULL);
            std::uniform_int_distribution<uint64_t> dist(3002950000, haystack.size() + 3002950000);

            std::iota(std::begin(haystack), std::end(haystack), 3002954000);
            std::generate(std::begin(needles), std::end(needles),
                          [&]
                          {
                              return dist(engine);
                          });

            expect(throws<std::runtime_error>(
                [&]
                {
                    // Note the reversed order of haystack and needles in this call - more unique needles than uniques in haystack
                    IsMemberHash32(haystack.size(), haystack.data(), needles.size(), needles.data(), small_output.data(),
                                   bools.data(), 8, HASH_MODE(1), 0);
                }));
        };

        skip / "ismember64_type_coercion_effects_rip-254"_test = [&]
        {
            std::vector<uint64_t> haystack{ 3ULL, 0x7F'FF'FF'FF'FF'FF'FF'FEULL };
            std::vector<int64_t> needles{ 0x7F'FF'FF'FF'FF'FF'FF'FEULL };

            expect(not throws<std::runtime_error>(
                [&]
                {
                    IsMemberHash32(needles.size(), needles.data(), haystack.size(), haystack.data(), small_output.data(),
                                   bools.data(), 8, HASH_MODE(0), 0);
                }));

            expect(not throws<std::runtime_error>(
                [&]
                {
                    IsMemberHash32(needles.size(), needles.data(), haystack.size(), haystack.data(), small_output.data(),
                                   bools.data(), 8, HASH_MODE(1), 0);
                }));
        };
    };
}
