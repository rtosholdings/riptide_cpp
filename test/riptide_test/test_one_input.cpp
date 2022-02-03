#include "../../src/one_input_impl.h"
#include "../../src/platform_detect.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <type_traits>
#include <cfloat>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    using namespace riptable_cpp;
    using namespace riptable_cpp::implementation;
    using namespace riptide::simd::avx2;

    std::array<float const, 31> const input_data_simple_f = { -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,   -0.5, 0,   0.5, 1.0,
                                                              1.5, 2,    2.5, 3,    3.5, 4,    4.5,  5,    5.5, 6,   6.5,
                                                              7,   7.5,  8,   8.5,  9,   9.5,  10.5, 10.5, 11 };

    std::array<float const, 31> const input_data_nan_f = { -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,   -0.5, 0,   NAN, 1.0,
                                                           1.5, 2,    2.5, 3,    3.5, 4,    4.5,  5,    5.5, 6,   6.5,
                                                           7,   7.5,  8,   8.5,  9,   9.5,  10.5, 10.5, 11 };

    std::array<float const, 31> const input_data_inf_f = { -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,   -0.5, 0,   INFINITY, 1.0,
                                                           1.5, 2,    2.5, 3,    3.5, 4,    4.5,  5,    5.5, 6,        6.5,
                                                           7,   7.5,  8,   8.5,  9,   9.5,  10.5, 10.5, 11 };

    std::array<float const, 31> const input_data_normal_f = { -4,       -3.5,          -3,   -2.5, -2,  -1.5, -1,  -0.5, 0,
                                                              INFINITY, FLT_MIN / 2.0, NAN,  2,    2.5, 3,    3.5, 4,    4.5,
                                                              5,        5.5,           6,    6.5,  7,   7.5,  8,   8.5,  9,
                                                              9.5,      10.5,          10.5, 11 };

    std::array<int32_t const, 31> const input_data_simple_i = { -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  6, 7,
                                                                8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

    float const * p_float = input_data_simple_f.data();
    float const * p_nans = input_data_nan_f.data();
    float const * p_inf = input_data_inf_f.data();
    float const * p_norm = input_data_normal_f.data();
    int32_t const * p_int32 = input_data_simple_i.data();

    suite one_input = []
    {
        size_t len{ sizeof(input_data_simple_f) };
        "expected_array_len_f"_test = [len] { expect(31_u == len / sizeof(float)); };

        "calculate_abs_int"_test = [&]
        {
            abs_op op{};
            int32_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_int32 + 5), &op, &data_type, vec256<int32_t>{});
            int32_t const * res_ptr{ reinterpret_cast<int32_t const *>(&x) };
            expect(res_ptr[0] == 3_i);
            expect(res_ptr[1] == 2_i);
            expect(res_ptr[2] == 1_i);
            expect(res_ptr[3] == 0_i);
            expect(res_ptr[4] == 1_i);
            expect(res_ptr[5] == 2_i);
            expect(res_ptr[6] == 3_i);
            expect(res_ptr[7] == 4_i);
            expect(sizeof(decltype(x)) == 8 * sizeof(int32_t));
        };

        "calculate_abs_float"_test = [&]
        {
            abs_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 5), &op, &data_type, vec256<float>{});
            float const * res_ptr{ reinterpret_cast<float const *>(&x) };
            expect(res_ptr[0] == 1.5_f);
            expect(res_ptr[1] == 1.0_f);
            expect(res_ptr[2] == 0.5_f);
            expect(res_ptr[3] == 0.0_f);
            expect(res_ptr[4] == 0.5_f);
            expect(res_ptr[5] == 1.0_f);
            expect(res_ptr[6] == 1.5_f);
            expect(res_ptr[7] == 2.0_f);
            expect(sizeof(decltype(x)) == 8 * sizeof(float));
        };

        "walk_fabs_float"_test = [&]
        {
            operation_t op{ fabs_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<float, 28> x{};
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
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
            std::array<float, 28> x{};
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
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
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 3.141592_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_fabs_not_fp"_test = [&]
        {
            fabs_op op{};
            int32_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_int32 + 5), &op, &data_type, vec256<int32_t>{});
            expect(x == -3_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_fabs_fp"_test = [&]
        {
            fabs_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 5), &op, &data_type, vec256<float>{});
            expect(x == 1.5_f);
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
        };

        "calculate_fabs_fp_positive"_test = [&]
        {
            fabs_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 9), &op, &data_type, vec256<float>{});
            expect(x == 0.5_f);
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
        };

        "calculate_sign_unsigned"_test = []
        {
            sign_op op{};
            uint32_traits data_type{};
            uint32_t data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return uint32_t";
            data = 1;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 1_u);
        };

        "calculate_sign_signed_float"_test = []
        {
            sign_op op{};
            float_traits data_type{};
            float data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == 0.0_f);
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
            data = 1.3f;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == 1.0_f);
            data = -4.25f;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == -1.0_f);
        };

        "calculate_floatsign_int"_test = []
        {
            floatsign_op op{};
            int32_traits data_type{};
            int32_t data{ 1 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<void>{});
            expect(x == 0_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_floatsign_float"_test = []
        {
            sign_op op{};
            float_traits data_type{};
            float data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == 0.0_f);
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
            data = 1.3f;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == 1.0_f);
            data = -4.25f;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == -1.0_f);
        };

        "calculate_negate_int"_test = []
        {
            neg_op op{};
            int32_traits data_type{};
            int32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -42_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_negate_uint"_test = []
        {
            neg_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_negate_double"_test = []
        {
            neg_op op{};
            double_traits data_type{};
            double data{ 42.0 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<double>{});
            expect(x == -42.0_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_bitwise_neg_float"_test = []
        {
            bitwise_not_op op{};
            float_traits data_type{};
            float data{ -4.2f };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(std::isnan(x)) << "Return value should be a NaN";
            expect(std::is_same_v<decltype(x), float>) << "Should return a float";
        };

        "calculate_bitwise_neg_unsigned"_test = []
        {
            bitwise_not_op op{};
            uint32_traits data_type{};
            uint32_t data{ 0x5A5A5A5A };
            constexpr uint32_t expected{ 0xA5A5A5A5 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == expected) << "Return value should be 0xa5a5a5a5, but it was" << std::hex << x;
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_bitwise_neg_signed"_test = []
        {
            bitwise_not_op op{};
            int32_traits data_type{};
            int32_t data{ 0x5A5A5A5A };
            int32_t expected{ std::make_signed_t<uint32_t>(0xA5A5A5A5) };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == expected) << "Return value should be 0xa5a5a5a5, but it was" << std::hex << x;
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_round_signed"_test = []
        {
            round_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -13_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_round_unsigned"_test = []
        {
            round_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return an uint32_t";
        };

        // Illustrates the difference between std::round (always round away from zero) and _mm256_round_ps (round to nearest even)
        "calculate_round_float_simd"_test = []
        {
            round_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 5), &op, &data_type, vec256<float>{});
            float const * res_ptr{ reinterpret_cast<float const *>(&x) };
            expect(res_ptr[0] == -2.0_f);
            expect(res_ptr[1] == -1.0_f);
            expect(res_ptr[2] == 0.0_f);
            expect(std::round(-0.5) == -1.0_f);
            expect(res_ptr[3] == 0.0_f);
            expect(std::round(0.5) == 1.0_f);
            expect(res_ptr[4] == 0.0_f);
            expect(res_ptr[5] == 1.0_f);
            expect(res_ptr[6] == 2.0_f);
            expect(res_ptr[7] == 2.0_f);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_round_float"_test = [&]
        {
            operation_t op{ round_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<float, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == -2.0_f);
            expect(x[1] == -1.0_f);
            expect(x[2] == 0.0_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.0_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 2.0_f);
            expect(x[7] == 2.0_f);
            expect(x[8] == 2.0_f);
            expect(x[24] == 11.0_f);
            expect(x[25] == 11.0_f);
        };

        "calculate_floor_signed"_test = []
        {
            floor_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -13_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_floor_unsigned"_test = []
        {
            floor_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return an uint32_t";
        };

        "calculate_floor_float_simd"_test = []
        {
            floor_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 5), &op, &data_type, vec256<float>{});
            float const * res_ptr{ reinterpret_cast<float const *>(&x) };
            expect(res_ptr[0] == -2.0_f);
            expect(res_ptr[1] == -1.0_f);
            expect(res_ptr[2] == -1.0_f);
            expect(res_ptr[3] == 0.0_f);
            expect(res_ptr[4] == 0.0_f);
            expect(res_ptr[5] == 1.0_f);
            expect(res_ptr[6] == 1.0_f);
            expect(res_ptr[7] == 2.0_f);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_floor_float"_test = [&]
        {
            operation_t op{ floor_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<float, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == -2.0_f);
            expect(x[1] == -1.0_f);
            expect(x[2] == -1.0_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.0_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 1.0_f);
            expect(x[7] == 2.0_f);
            expect(x[8] == 2.0_f);
            expect(x[24] == 10.0_f);
            expect(x[25] == 11.0_f);
        };

        "calculate_trunc_signed"_test = [&]
        {
            trunc_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -13_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_trunc_unsigned"_test = [&]
        {
            trunc_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42u };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return an uint32_t";
        };

        "calculate_trunc_float_simd"_test = [&]
        {
            trunc_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 5), &op, &data_type, vec256<float>{});
            float const * res_ptr{ reinterpret_cast<float const *>(&x) };
            expect(res_ptr[0] == -1.0_f);
            expect(res_ptr[1] == -1.0_f);
            expect(res_ptr[2] == 0.0_f);
            expect(res_ptr[3] == 0.0_f);
            expect(res_ptr[4] == 0.0_f);
            expect(res_ptr[5] == 1.0_f);
            expect(res_ptr[6] == 1.0_f);
            expect(res_ptr[7] == 2.0_f);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_trunc_float"_test = [&]
        {
            operation_t op{ trunc_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<float, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == -1.0_f);
            expect(x[1] == -1.0_f);
            expect(x[2] == 0.0_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.0_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 1.0_f);
            expect(x[7] == 2.0_f);
            expect(x[8] == 2.0_f);
            expect(x[24] == 10.0_f);
            expect(x[25] == 11.0_f);
        };

        "calculate_ceil_signed"_test = [&]
        {
            ceil_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == -13_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_ceil_unsigned"_test = [&]
        {
            ceil_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42u };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 42_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return an uint32_t";
        };

        "calculate_ceil_float_simd"_test = [&]
        {
            ceil_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 5), &op, &data_type, vec256<float>{});
            float const * res_ptr{ reinterpret_cast<float const *>(&x) };
            expect(res_ptr[0] == -1.0_f);
            expect(res_ptr[1] == -1.0_f);
            expect(res_ptr[2] == 0.0_f);
            expect(res_ptr[3] == 0.0_f);
            expect(res_ptr[4] == 1.0_f);
            expect(res_ptr[5] == 1.0_f);
            expect(res_ptr[6] == 2.0_f);
            expect(res_ptr[7] == 2.0_f);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_ceil_float"_test = [&]
        {
            operation_t op{ ceil_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<float, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == -1.0_f);
            expect(x[1] == -1.0_f);
            expect(x[2] == 0.0_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 1.0_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 2.0_f);
            expect(x[7] == 2.0_f);
            expect(x[8] == 3.0_f);
            expect(x[24] == 11.0_f);
            expect(x[25] == 11.0_f);
        };

        "calculate_sqrt_signed"_test = [&]
        {
            sqrt_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            errno = 0;
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
#ifdef _DEBUG
            // The release build sees through the entire call stack, optimizes it out, and since errno is a side-effect
            expect(errno == EDOM) << "Expect Domain Error for sqrt (negative), errno is currently " << errno;
#endif
            data = 48;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 6_i);
            expect(std::is_same_v<decltype(x), int32_t>) << "Should return an int32_t";
        };

        "calculate_sqrt_unsigned"_test = [&]
        {
            sqrt_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42u };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 6_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return an uint32_t";
        };

        "calculate_sqrt_float_simd"_test = [&]
        {
            sqrt_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_float + 5), &op, &data_type, vec256<float>{});
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_sqrt_float"_test = [&]
        {
            operation_t op{ sqrt_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<float, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(std::isnan(x[0]));
            expect(std::isnan(x[1]));
            expect(std::isnan(x[2]));
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.707107_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 1.224745_f);
            expect(x[7] == 1.41421_f);
            expect(x[8] == 1.58114_f);
            expect(x[24] == 3.24037_f);
            expect(x[25] == 3.31662_f);
        };

        "calculate_log_void"_test = []
        {
            log_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(std::isnan(x)) << "Log of a negative number returns a NaN";
            expect(errno == EDOM) << "Log of a negative number sets errno to Domain Error";
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 1.14473_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_log2_void"_test = []
        {
            log2_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(std::isnan(x)) << "Log2 of a negative number returns a NaN";
            expect(errno == EDOM) << "Log2 of a negative number sets errno to Domain Error";
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 1.6515_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_log10_void"_test = []
        {
            log10_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(std::isnan(x)) << "Log2 of a negative number returns a NaN";
            expect(errno == EDOM) << "Log2 of a negative number sets errno to Domain Error";
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 0.49715_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_exp_void"_test = []
        {
            exp_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 0.04321395_d);
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 23.1407_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_exp2_void"_test = []
        {
            exp2_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 0.1133148_d);
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 8.82497_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_cbrt_void"_test = []
        {
            cbrt_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == -1.4646_d);
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 1.4646_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_tan_void"_test = []
        {
            tan_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 0.00000065_d);
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == -0.00000065_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_cos_void"_test = []
        {
            cos_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == -1.0_d);
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == -1.0_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_sin_void"_test = []
        {
            sin_op op{};
            double_traits data_type{};
            double in_data{ -3.141592 };
            auto x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == -0.00000065359_d);
            in_data = 3.141592;
            x = calculate(reinterpret_cast<char const *>(&in_data), &op, &data_type, vec256<void>{});
            expect(x == 0.00000065359_d);
            expect(std::is_same_v<decltype(x), double>) << "Should return a double";
        };

        "calculate_signbit_unsigned"_test = []
        {
            signbit_op op{};
            uint32_traits data_type{};
            uint32_t data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == false);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return uint32_t";
            data = 1;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == false);
        };

        "calculate_signbit_signed"_test = []
        {
            signbit_op op{};
            int32_traits data_type{};
            int32_t data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return uint32_t";
            data = 1;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            data = -1;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == std::numeric_limits<uint32_t>::max());
        };

        "calculate_sign_signed_float"_test = []
        {
            signbit_op op{};
            float_traits data_type{};
            float data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
            data = 1.3f;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            data = -4.25f;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == std::numeric_limits<uint32_t>::max());
        };

        "calculate_not_integral"_test = []
        {
            not_op op{};
            int32_traits data_type{};
            int32_t data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == std::numeric_limits<uint32_t>::max());
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return uint32_t";
            data = 1;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
        };

        "calculate_not_float"_test = []
        {
            not_op op{};
            float_traits data_type{};
            float data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == std::numeric_limits<uint32_t>::max());
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
            data = 0.0000001f;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
        };

        "calculate_isnotnan_signed"_test = []
        {
            isnotnan_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotnan_unsigned"_test = []
        {
            isnotnan_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotnan_float_simd"_test = []
        {
            isnotnan_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_nans + 5), &op, &data_type, vec256<float>{});
            int32_t const * res_ptr{ reinterpret_cast<int32_t const *>(&x) };
            expect(res_ptr[0] == -1_i);
            expect(res_ptr[1] == -1_i);
            expect(res_ptr[2] == -1_i);
            expect(res_ptr[3] == -1_i);
            expect(res_ptr[4] == 0_i);
            expect(res_ptr[5] == -1_i);
            expect(res_ptr[6] == -1_i);
            expect(res_ptr[7] == -1_i);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_isnotnan_float"_test = [&]
        {
            operation_t op{ isnotnan_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<int, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_nans + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == -1_i);
            expect(x[1] == -1_i);
            expect(x[2] == -1_i);
            expect(x[3] == -1_i);
            expect(x[4] == 0_i);
            expect(x[5] == -1_i);
            expect(x[6] == -1_i);
            expect(x[7] == -1_i);
            expect(x[8] == -1_i);
            expect(x[24] == -1_i);
            expect(x[25] == -1_i);
        };

        "calculate_isnan_signed"_test = []
        {
            isnan_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnan_unsigned"_test = []
        {
            isnan_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnan_float_simd"_test = []
        {
            isnan_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_nans + 5), &op, &data_type, vec256<float>{});
            int32_t const * res_ptr{ reinterpret_cast<int32_t const *>(&x) };
            expect(res_ptr[0] == 0_i);
            expect(res_ptr[1] == 0_i);
            expect(res_ptr[2] == 0_i);
            expect(res_ptr[3] == 0_i);
            expect(res_ptr[4] == -1_i);
            expect(res_ptr[5] == 0_i);
            expect(res_ptr[6] == 0_i);
            expect(res_ptr[7] == 0_i);
            expect(std::is_same_v<__m256, decltype(x)>) << "Should return a __m256";
        };

        "walk_isnan_float"_test = [&]
        {
            operation_t op{ isnan_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<int, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_nans + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 0_i);
            expect(x[1] == 0_i);
            expect(x[2] == 0_i);
            expect(x[3] == 0_i);
            expect(x[4] == -1_i);
            expect(x[5] == 0_i);
            expect(x[6] == 0_i);
            expect(x[7] == 0_i);
            expect(x[8] == 0_i);
            expect(x[24] == 0_i);
            expect(x[25] == 0_i);
        };

        "calculate_isfinite_signed"_test = []
        {
            isfinite_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isfinite_unsigned"_test = []
        {
            isfinite_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isfinite_float"_test = []
        {
            isfinite_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_inf + 5), &op, &data_type, vec256<float>{});
            expect(x == std::numeric_limits<uint32_t>::max()) << "Should be true / uint32_t::max, was " << x;
            x = calculate(reinterpret_cast<char const *>(p_inf + 9), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            expect(std::is_same_v<uint32_t, decltype(x)>) << "Should return a uint32_t";
        };

        "walk_isfinite_float"_test = [&]
        {
            operation_t op{ isfinite_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<uint32_t, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_inf + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 4294967295_u);
            expect(x[1] == 4294967295_u);
            expect(x[2] == 4294967295_u);
            expect(x[3] == 4294967295_u);
            expect(x[4] == 0_u);
            expect(x[5] == 4294967295_u);
            expect(x[6] == 4294967295_u);
            expect(x[7] == 4294967295_u);
            expect(x[8] == 4294967295_u);
            expect(x[24] == 4294967295_u);
            expect(x[25] == 4294967295_u);
        };

        "calculate_isnotfinite_signed"_test = []
        {
            isnotfinite_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotfinite_unsigned"_test = []
        {
            isnotfinite_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotfinite_float"_test = []
        {
            isnotfinite_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_inf + 5), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_inf + 9), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            expect(std::is_same_v<uint32_t, decltype(x)>) << "Should return a uint32_t";
        };

        "walk_isnotfinite_float"_test = [&]
        {
            operation_t op{ isnotfinite_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<uint32_t, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_inf + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 0_u);
            expect(x[1] == 0_u);
            expect(x[2] == 0_u);
            expect(x[3] == 0_u);
            expect(x[4] == 4294967295_u);
            expect(x[5] == 0_u);
            expect(x[6] == 0_u);
            expect(x[7] == 0_u);
            expect(x[8] == 0_u);
            expect(x[24] == 0_u);
            expect(x[25] == 0_u);
        };

        "calculate_isinf_signed"_test = []
        {
            isinf_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isinf_unsigned"_test = []
        {
            isinf_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isinf_float"_test = []
        {
            isinf_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_inf + 5), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_inf + 9), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            expect(std::is_same_v<uint32_t, decltype(x)>) << "Should return a uint32_t";
        };

        "walk_isinf_float"_test = [&]
        {
            operation_t op{ isinf_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<uint32_t, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_inf + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 0_u);
            expect(x[1] == 0_u);
            expect(x[2] == 0_u);
            expect(x[3] == 0_u);
            expect(x[4] == 4294967295_u);
            expect(x[5] == 0_u);
            expect(x[6] == 0_u);
            expect(x[7] == 0_u);
            expect(x[8] == 0_u);
            expect(x[24] == 0_u);
            expect(x[25] == 0_u);
        };

        "calculate_isnotinf_signed"_test = []
        {
            isnotinf_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotinf_unsigned"_test = []
        {
            isnotinf_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotinf_float"_test = []
        {
            isnotinf_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_inf + 5), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            x = calculate(reinterpret_cast<char const *>(p_inf + 9), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            expect(std::is_same_v<uint32_t, decltype(x)>) << "Should return a uint32_t";
        };

        "walk_isnotinf_float"_test = [&]
        {
            operation_t op{ isnotinf_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<uint32_t, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_inf + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 4294967295_u);
            expect(x[1] == 4294967295_u);
            expect(x[2] == 4294967295_u);
            expect(x[3] == 4294967295_u);
            expect(x[4] == 0_u);
            expect(x[5] == 4294967295_u);
            expect(x[6] == 4294967295_u);
            expect(x[7] == 4294967295_u);
            expect(x[8] == 4294967295_u);
            expect(x[24] == 4294967295_u);
            expect(x[25] == 4294967295_u);
        };

        "calculate_isnormal_signed"_test = []
        {
            isnormal_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnormal_unsigned"_test = []
        {
            isnormal_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnormal_float"_test = []
        {
            isnormal_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_norm + 5), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 8), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 9), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 10), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 11), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            expect(std::is_same_v<uint32_t, decltype(x)>) << "Should return a uint32_t";
        };

        "walk_isnormal_float"_test = [&]
        {
            operation_t op{ isnormal_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<uint32_t, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_norm + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 4294967295_u);
            expect(x[1] == 4294967295_u);
            expect(x[2] == 4294967295_u);
            expect(x[3] == 0_u);
            expect(x[4] == 0_u);
            expect(x[5] == 0_u);
            expect(x[6] == 0_u);
            expect(x[7] == 4294967295_u);
            expect(x[8] == 4294967295_u);
            expect(x[24] == 4294967295_u);
            expect(x[25] == 4294967295_u);
        };

        "calculate_isnotnormal_signed"_test = []
        {
            isnotnormal_op op{};
            int32_traits data_type{};
            int32_t data{ -13 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotnormal_unsigned"_test = []
        {
            isnotnormal_op op{};
            uint32_traits data_type{};
            uint32_t data{ 42 };
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnotnormal_float"_test = []
        {
            isnotnormal_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_norm + 5), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 8), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 9), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 10), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 11), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            expect(std::is_same_v<uint32_t, decltype(x)>) << "Should return a uint32_t";
        };

        "walk_isnotnormal_float"_test = [&]
        {
            operation_t op{ isnotnormal_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<uint32_t, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_norm + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 0_u);
            expect(x[1] == 0_u);
            expect(x[2] == 0_u);
            expect(x[3] == 4294967295_u);
            expect(x[4] == 4294967295_u);
            expect(x[5] == 4294967295_u);
            expect(x[6] == 4294967295_u);
            expect(x[7] == 0_u);
            expect(x[8] == 0_u);
            expect(x[24] == 0_u);
            expect(x[25] == 0_u);
        };

        "calculate_isnanorzero_signed"_test = []
        {
            isnanorzero_op op{};
            int32_traits data_type{};
            int32_t data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 4294967295_u);
            data = -13;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<int32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnanorzero_unsigned"_test = []
        {
            isnanorzero_op op{};
            uint32_traits data_type{};
            uint32_t data{};
            auto x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 4294967295_u);
            data = 42;
            x = calculate(reinterpret_cast<char const *>(&data), &op, &data_type, vec256<uint32_t>{});
            expect(x == 0_u);
            expect(std::is_same_v<decltype(x), uint32_t>) << "Should return a uint32_t";
        };

        "calculate_isnanorzero_float"_test = []
        {
            isnanorzero_op op{};
            float_traits data_type{};
            auto x = calculate(reinterpret_cast<char const *>(p_norm + 5), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 8), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 9), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 10), &op, &data_type, vec256<float>{});
            expect(x == 0_u);
            x = calculate(reinterpret_cast<char const *>(p_norm + 11), &op, &data_type, vec256<float>{});
            expect(x == 4294967295_u);
            expect(std::is_same_v<uint32_t, decltype(x)>) << "Should return a uint32_t";
        };

        "walk_isnanorzero_float"_test = [&]
        {
            operation_t op{ isnanorzero_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<uint32_t, 26> x{};
            walk_data_array(1, 26, 4, 4, reinterpret_cast<char const *>(p_norm + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 0_u);
            expect(x[1] == 0_u);
            expect(x[2] == 0_u);
            expect(x[3] == 4294967295_u);
            expect(x[4] == 0_u);
            expect(x[5] == 0_u);
            expect(x[6] == 4294967295_u);
            expect(x[7] == 0_u);
            expect(x[8] == 0_u);
            expect(x[24] == 0_u);
            expect(x[25] == 0_u);
        };
    };
}
