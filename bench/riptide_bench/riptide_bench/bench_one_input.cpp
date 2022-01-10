#include "../../../src/one_input_impl.h"

#include "../../benchmark/include/benchmark/benchmark.h"

#include <array>
#include <cfloat>

namespace
{
    using namespace riptable_cpp;
    using namespace riptable_cpp::implementation;
    using namespace riptide::simd::avx2;

    alignas(256) std::array<float const, 31> const input_data_simple_f = { -4, -3.5, -3,  -2.5, -2,   -1.5, -1, -0.5,
                                                                           0,  0.5,  1.0, 1.5,  2,    2.5,  3,  3.5,
                                                                           4,  4.5,  5,   5.5,  6,    6.5,  7,  7.5,
                                                                           8,  8.5,  9,   9.5,  10.5, 10.5, 11 };

    alignas(256) std::array<float const, 31> const input_data_nan_f = { -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,   -0.5, 0,   NAN, 1.0,
                                                                        1.5, 2,    2.5, 3,    3.5, 4,    4.5,  5,    5.5, 6,   6.5,
                                                                        7,   7.5,  8,   8.5,  9,   9.5,  10.5, 10.5, 11 };

    alignas(256) std::array<float const, 31> const input_data_inf_f = { -4, -3.5,     -3,  -2.5, -2,   -1.5, -1, -0.5,
                                                                        0,  INFINITY, 1.0, 1.5,  2,    2.5,  3,  3.5,
                                                                        4,  4.5,      5,   5.5,  6,    6.5,  7,  7.5,
                                                                        8,  8.5,      9,   9.5,  10.5, 10.5, 11 };

    alignas(256) std::array<float const, 31> const input_data_normal_f = {
        -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,  -0.5, 0,   INFINITY, FLT_MIN / 2.0, NAN,  2,    2.5, 3, 3.5, 4,
        4.5, 5,    5.5, 6,    6.5, 7,    7.5, 8,    8.5, 9,        9.5,           10.5, 10.5, 11
    };

    alignas(256) std::array<int32_t const, 31> const input_data_simple_i = { -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,
                                                                             3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                                                             14, 15, 16, 17, 18, 19, 20, 21, 22 };

    float const * p_float = input_data_simple_f.data();
    float const * p_nans = input_data_nan_f.data();
    float const * p_inf = input_data_inf_f.data();
    float const * p_norm = input_data_normal_f.data();
    int32_t const * p_int32 = input_data_simple_i.data();

    static constexpr size_t len{ sizeof(input_data_simple_f) };

    void bench_calculate_abs_int(benchmark::State & state)
    {
        abs_op op{};
        int32_traits data_type{};
        for (auto _ : state)
        {
            auto x = calculate(reinterpret_cast<char const *>(p_int32), &op, &data_type, vec256<int32_t>{});
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_calculate_abs_int);

    void bench_calculate_abs_float(benchmark::State & state)
    {
        abs_op op{};
        float_traits data_type{};
        for (auto _ : state)
        {
            auto x = calculate(reinterpret_cast<char const *>(p_float), &op, &data_type, vec256<float>{});
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_calculate_abs_float);

    void bench_walk_fabs_float(benchmark::State & state)
    {
        operation_t op{ fabs_op{} };
        data_type_t data_type{ float_traits{} };
        std::array<float, 28> x{};
        for (auto _ : state)
        {
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const *>(p_float), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    };

    BENCHMARK(bench_walk_fabs_float);

    void bench_walk_abs_float(benchmark::State & state)
    {
        operation_t op{ abs_op{} };
        data_type_t data_type{ float_traits{} };
        std::array<float, 28> x{};
        for (auto _ : state)
        {
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const *>(p_float), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    };
    BENCHMARK(bench_walk_abs_float);
}
