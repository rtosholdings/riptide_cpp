#include "flat_hash_map.h"

#define BOOST_UT_DISABLE_MODULE
#include "boost/ut.hpp"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <tuple>

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
        "make_hash"_test = [&]<class DataT>(DataT arg)
        {
            boost::ut::log << "Data Type: " << reflection::type_name<DataT>() << " ";
            std::array<DataT, 18> data{};
            if constexpr (std::is_floating_point_v<DataT>)
            {
                std::uniform_real_distribution<DataT> dist(0, std::numeric_limits<DataT>::max());
                std::generate(std::begin(data), std::end(data), [&] { return dist(engine); });
            }
            else
            {
                std::uniform_int_distribution<DataT> dist(0, std::numeric_limits<DataT>::max());
                std::generate(std::begin(data), std::end(data), [&] { return dist(engine); });
            }

            fhm_hasher<DataT, int64_t> hash{};

            hash.make_hash(data.size(), reinterpret_cast<char const *>(data.data()), 0);

            if constexpr (not std::is_same_v<decltype(hash.hasher), oneapi::tbb::concurrent_hash_map<DataT, int64_t>>)
            {
                expect(hash.hasher.find(data[0])->second == 0u);
                expect(hash.hasher.find(data[1])->second == 1u);
                expect(hash.hasher.find(data[2])->second == 2u);
                expect(hash.hasher.find(data[3])->second == 3u);
                expect(hash.hasher.find(data[4])->second == 4u);
                expect(hash.hasher.find(data[5])->second == 5u);
                expect(hash.hasher.find(data[6])->second == 6u);
                expect(hash.hasher.find(data[7])->second == 7u);
                expect(hash.hasher.find(data[8])->second == 8u);
                expect(hash.hasher.find(data[9])->second == 9u);
                expect(hash.hasher.find(data[10])->second == 10u);
                expect(hash.hasher.find(data[11])->second == 11u);
                expect(hash.hasher.find(data[12])->second == 12u);
                expect(hash.hasher.find(data[13])->second == 13u);
                expect(hash.hasher.find(data[14])->second == 14u);
                expect(hash.hasher.find(data[15])->second == 15u);
                expect(hash.hasher.find(data[16])->second == 16u);
                expect(hash.hasher.find(data[17])->second == 17u);
            }
            else
            {
                using const_acc = typename oneapi::tbb::concurrent_hash_map<DataT, int64_t>::const_accessor;
                const oneapi::tbb::concurrent_hash_map<DataT, int64_t> & const_map = hash.hasher;
                const_acc ca{};
                expect(hash.hasher.find(ca, data[0]));
                expect(ca->second == 0_i);
                expect(hash.hasher.find(ca, data[1]));
                expect(ca->second == 1_i);
                expect(hash.hasher.find(ca, data[2]));
                expect(ca->second == 2_i);
                expect(hash.hasher.find(ca, data[3]));
                expect(ca->second == 3_i);
                expect(hash.hasher.find(ca, data[4]));
                expect(ca->second == 4_i);
                expect(hash.hasher.find(ca, data[5]));
                expect(ca->second == 5_i);
                expect(hash.hasher.find(ca, data[6]));
                expect(ca->second == 6_i);
                expect(hash.hasher.find(ca, data[7]));
                expect(ca->second == 7_i);
                expect(hash.hasher.find(ca, data[8]));
                expect(ca->second == 8_i);
                expect(hash.hasher.find(ca, data[9]));
                expect(ca->second == 9_i);
                expect(hash.hasher.find(ca, data[10]));
                expect(ca->second == 10_i);
                expect(hash.hasher.find(ca, data[11]));
                expect(ca->second == 11_i);
                expect(hash.hasher.find(ca, data[12]));
                expect(ca->second == 12_i);
                expect(hash.hasher.find(ca, data[13]));
                expect(ca->second == 13_i);
                expect(hash.hasher.find(ca, data[14]));
                expect(ca->second == 14_i);
                expect(hash.hasher.find(ca, data[15]));
                expect(ca->second == 15_i);
                expect(hash.hasher.find(ca, data[16]));
                expect(ca->second == 16_i);
                expect(hash.hasher.find(ca, data[17]));
                expect(ca->second == 17_i);
            }
        } | std::tuple<int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>{};

        "is_member_uint64"_test = [&]
        {
            std::vector<uint64_t> haystack(10);
            std::uniform_int_distribution<uint64_t> dist(0, ULLONG_MAX - 2);

            std::generate(std::begin(haystack), std::end(haystack), [&] { return dist(engine); });
            std::vector<uint64_t> needles{ haystack[3], haystack[9], haystack[0], haystack[5], ULLONG_MAX - 1 };

            expect(is_member(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                             reinterpret_cast<char const *>(haystack.data()), output.data(), bools.data(), needles[0]) == 0_i);

            expect( output[ 0 ] == 3_i );
            expect( output[ 1 ] == 9_i );
            expect( output[ 2 ] == 0_i );
            expect( output[ 3 ] == 5_i );
            expect( output[ 4 ] == -1_i );

            expect( bools[ 0 ] == 1_i );
            expect( bools[ 1 ] == 1_i );
            expect( bools[ 2 ] == 1_i );
            expect( bools[ 3 ] == 1_i );
            expect( bools[ 4 ] == 0_i );
        };

        "is_member_many_uniques"_test = [&]
        {
            std::vector<uint64_t> haystack(128 * 1024ULL * 1024ULL);
            std::vector<uint64_t> needles(1024ULL * 1024ULL);
            std::uniform_int_distribution<uint64_t> dist(3002950000, haystack.size() + 3002950000);

            std::iota(std::begin(haystack), std::end(haystack), 3002954000);
            std::generate(std::begin(needles), std::end(needles), [&] { return dist(engine); });

            expect(is_member(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                             reinterpret_cast<char const *>(haystack.data()), output.data(), bools.data(), needles[0]) == 0_i);
        };
    };
}
