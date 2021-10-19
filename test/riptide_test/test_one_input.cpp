#include "../../src/one_input_impl.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <type_traits>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    using namespace riptable_cpp;
    using namespace riptable_cpp::implementation;
    using namespace riptide::simd::avx2;

    std::array< float const, 33 > const input_data_simple_f = {-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2, 2.5, 3,
        3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11};

    std::array< int32_t const, 33 > const input_data_simple_i = { -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

    float const * p_float = input_data_simple_f.data();
    int32_t const* p_int32 = input_data_simple_i.data();

    suite one_input = []
    {
        size_t len{ sizeof(input_data_simple_f) };
        "expected_array_len_f"_test = [len]
        {
            expect(33_i == len / sizeof(float));
        };

        "calculate_abs_int"_test = [&]
        {
            abs_op op{};
            int32_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(p_int32 + 5), &op, &data_type, vec256<int32_t>{});
            expect(x.m256i_i32[0] == 3_i);
            expect(x.m256i_i32[1] == 2_i);
            expect(x.m256i_i32[2] == 1_i);
            expect(x.m256i_i32[3] == 0_i);
            expect(x.m256i_i32[4] == 1_i);
            expect(x.m256i_i32[5] == 2_i);
            expect(x.m256i_i32[6] == 3_i);
            expect(x.m256i_i32[7] == 4_i);
            expect(sizeof(decltype(x)) == 8 * sizeof(int32_t));
        };

        "calculate_abs_float"_test = [&]
        {
            abs_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(p_float + 5), &op, &data_type, vec256<float>{});
            expect(x.m256_f32[0] == 1.5_f);
            expect(x.m256_f32[1] == 1.0_f);
            expect(x.m256_f32[2] == 0.5_f);
            expect(x.m256_f32[3] == 0.0_f);
            expect(x.m256_f32[4] == 0.5_f);
            expect(x.m256_f32[5] == 1.0_f);
            expect(x.m256_f32[6] == 1.5_f);
            expect(x.m256_f32[7] == 2.0_f);
            expect(sizeof(decltype(x)) == 8 * sizeof(float));
        };

        "walk_fabs_float"_test = [&]
        {
            operation_t op{ fabs_op{} };
            data_type_t data_type{ float_traits{} };
            std::array< float, 28 > x{};
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const*>(p_float + 5), reinterpret_cast<char*>(x.data()), op, data_type);
            expect(x[0] == 1.5_f);
            expect(x[1] == 1.0_f);
            expect(x[2] == 0.5_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.5_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 1.5_f);
            expect(x[7] == 2.0_f);
        };

        "walk_abs_float"_test = [&]
        {
            operation_t op{ abs_op{} };
            data_type_t data_type{ float_traits{} };
            std::array< float, 28 > x{};
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const*>(p_float + 5), reinterpret_cast<char*>(x.data()), op, data_type);
            expect(x[0] == 1.5_f);
            expect(x[1] == 1.0_f);
            expect(x[2] == 0.5_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.5_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 1.5_f);
            expect(x[7] == 2.0_f);
        };

        "calculate_abs_void"_test = []
        {
            abs_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const*>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 3.141592);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_fabs_not_fp"_test = [&]
        {
            fabs_op op{};
            int32_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(p_int32 + 5), &op, &data_type, vec256<int32_t>{});
            expect(x == -3_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_fabs_fp"_test = [&]
        {
            fabs_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(p_float + 5), &op, &data_type, vec256<float>{});
            expect(x == 1.5_f);
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
        };

        "calculate_fabs_fp_positive"_test = [&]
        {
            fabs_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(p_float + 9), &op, &data_type, vec256<float>{});
            expect(x == 0.5_f);
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
        };

        "calculate_sign_unsigned"_test = []
        {
            sign_op op{};
            uint32_traits data_type{};
            uint32_t data{};
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return uint32_t";
            data = 1;
            x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 1);
        };

        "calculate_sign_signed_float"_test = []
        {
            sign_op op{};
            float_traits data_type{};
            float data{};
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<float>{});
            expect(x == 0.0_f);
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
            data = 1.3f;
            x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<float>{});
            expect(x == 1.0_f);
            data = -4.25f;
            x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<float>{});
            expect(x == -1.0_f);
        };

        "calculate_floatsign_int"_test = []
        {
            floatsign_op op{};
            int32_traits data_type{};
            int32_t data{ 1 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<void>{});
            expect(x == 0_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_floatsign_float"_test = []
        {
            sign_op op{};
            float_traits data_type{};
            float data{};
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<float>{});
            expect(x == 0.0_f);
            expect(std::is_same_v < decltype(x), float>) << "Should return a float";
            data = 1.3f;
            x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<float>{});
            expect(x == 1.0_f);
            data = -4.25f;
            x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<float>{});
            expect(x == -1.0_f);
        };

        "calculate_negate_int"_test = []
        {
            neg_op op{};
            int32_traits data_type{};
            int32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -42_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_negate_uint"_test = []
        {
            neg_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_negate_double"_test = []
        {
            neg_op op{};
            double_traits data_type{};
            double data{ 42.0 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<double>{});
            expect(x == -42.0_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_bitwise_neg_float"_test = []
        {
            bitwise_not_op op{};
            float_traits data_type{};
            float data{ -4.2f };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<float>{});
            expect(std::isnan(x)) << "Return value should be a NaN";
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
        };

        "calculate_bitwise_neg_unsigned"_test = []
        {
            bitwise_not_op op{};
            uint32_traits data_type{};
            uint32_t data{ 0x5A5A5A5A };
            constexpr uint32_t expected{ 0xA5A5A5A5 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == expected) << "Return value should be 0xa5a5a5a5, but it was" << std::hex << x;
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_bitwise_neg_signed"_test = []
        {
            bitwise_not_op op{};
            int32_traits data_type{};
            int32_t data{ 0x5A5A5A5A };
            int32_t expected{ std::make_signed_t<uint32_t>(0xA5A5A5A5) };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == expected) << "Return value should be 0xa5a5a5a5, but it was" << std::hex << x;
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_round_signed"_test = []
        {
            round_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -13_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_round_unsigned"_test = []
        {
            round_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_i);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return an uint32_t";
        };

        // Illustrates the difference between std::round (always round away from zero) and _mm256_round_ps (round to nearest even)
        "calculate_round_float_simd"_test = []
        {
            round_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(p_float + 5), &op, &data_type, vec256<float>{});
            expect(x.m256_f32[0] == -2.0_f);
            expect(x.m256_f32[1] == -1.0_f);
            expect(x.m256_f32[2] == 0.0_f);
            expect(std::round(-0.5) == -1.0_f);
            expect(x.m256_f32[3] == 0.0_f);
            expect(std::round(0.5) == 1.0_f);
            expect(x.m256_f32[4] == 0.0_f);
            expect(x.m256_f32[5] == 1.0_f);
            expect(x.m256_f32[6] == 2.0_f);
            expect(x.m256_f32[7] == 2.0_f);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_round_float"_test = [&]
        {
            operation_t op{ round_op{} };
            data_type_t data_type{ float_traits{} };
            std::array< float, 28 > x{};
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const*>(p_float + 5), reinterpret_cast<char*>(x.data()), op, data_type);
            expect(x[0] == -2.0_f);
            expect(x[1] == -1.0_f);
            expect(x[2] == 0.0_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.0_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 2.0_f);
            expect(x[7] == 2.0_f);
            expect(x[8] == 2.0_f);
            expect(x[27] == 11.0_f); // Need to look into why we're not walking to the end
        };

        "calculate_floor_signed"_test = []
        {
            floor_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -13_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_floor_unsigned"_test = []
        {
            floor_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const*>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_i);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return an uint32_t";
        };

        "calculate_floor_float_simd"_test = []
        {
            floor_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const*>(p_float + 5), &op, &data_type, vec256<float>{});
            expect(x.m256_f32[0] == -2.0_f);
            expect(x.m256_f32[1] == -1.0_f);
            expect(x.m256_f32[2] == -1.0_f);
            expect(x.m256_f32[3] == 0.0_f);
            expect(x.m256_f32[4] == 0.0_f);
            expect(x.m256_f32[5] == 1.0_f);
            expect(x.m256_f32[6] == 1.0_f);
            expect(x.m256_f32[7] == 2.0_f);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_floor_float"_test = [&]
        {
            operation_t op{ floor_op{} };
            data_type_t data_type{ float_traits{} };
            std::array< float, 28 > x{};
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const*>(p_float + 5), reinterpret_cast<char*>(x.data()), op, data_type);
            expect(x[0] == -2.0_f);
            expect(x[1] == -1.0_f);
            expect(x[2] == -1.0_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.0_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 1.0_f);
            expect(x[7] == 2.0_f);
            expect(x[8] == 2.0_f);
            expect(x[27] == 11.0_f); // Need to look into why we're not walking to the end
        };
    };
}