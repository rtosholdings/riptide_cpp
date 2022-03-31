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
            if constexpr( std::is_floating_point_v<DataT> )
            {
                std::uniform_real_distribution<DataT> dist(0, std::numeric_limits<DataT>::max());
                std::generate(std::begin(data), std::end(data), [&] { return dist(engine); });
            }
            else
            {
                std::uniform_int_distribution<DataT> dist(0, std::numeric_limits<DataT>::max());
                std::generate(std::begin(data), std::end(data), [&] { return dist(engine); });
            }

            fhm_hasher<DataT> hash{};

            hash.make_hash(data.size(), reinterpret_cast< char const * >(data.data()), 0);

            expect( hash.hasher.find(data[0])->second == 0u);
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
        } | std::tuple<int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>{};
        
        "is_member_uint64"_test = [&]
        {
            std::vector<uint64_t> haystack(10);
            std::uniform_int_distribution<uint64_t> dist(0, ULLONG_MAX);

            std::generate(std::begin(haystack), std::end(haystack), [&] { return dist(engine); });
            std::vector<uint64_t> needles{ haystack[3], haystack[9], haystack[0], haystack[5] };

            expect(is_member(needles.size(), reinterpret_cast<char const *>(needles.data()), haystack.size(),
                             reinterpret_cast<char const *>(haystack.data()), output.data(), bools.data(), needles[0]) == 0_i);

            expect( output[ 0 ] == 3_i );
            expect( output[ 1 ] == 9_i );
            expect( output[ 2 ] == -1_i );
            expect( output[ 3 ] == 5_i );

            expect( bools[ 0 ] == 1_i );
            expect( bools[ 1 ] == 1_i );
            expect( bools[ 2 ] == 0_i );
            expect( bools[ 3 ] == 1_i );
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
