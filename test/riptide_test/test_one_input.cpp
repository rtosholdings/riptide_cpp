#include "../../src/one_input_impl.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    using namespace riptable_cpp;
    using namespace riptable_cpp::implementation;
    using namespace riptide::simd::avx2;

    static constexpr std::array< float const, 33 > input_data_simple = {-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2, 2.5, 3,
        3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11};

    suite one_input = []
    {
        size_t len{ sizeof(input_data_simple) };
        "expected_array_len"_test = [len]
        {
            expect(33_i == len / sizeof( float ) );
        };

        "calculate_abs"_test = [&]
        {
            abs_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(input_data_simple.data()+5), &op, &data_type, vec256<float>{});
            expect(x.m256_f32[0] == 1.5_f);
            expect(x.m256_f32[1] == 1.0_f);
            expect(x.m256_f32[2] == 0.5_f);
            expect(x.m256_f32[3] == 0.0_f);
            expect(x.m256_f32[4] == 0.5_f);
            expect(x.m256_f32[5] == 1.0_f);
            expect(x.m256_f32[6] == 1.5_f);
            expect(x.m256_f32[7] == 2.0_f);
        };
    };
}