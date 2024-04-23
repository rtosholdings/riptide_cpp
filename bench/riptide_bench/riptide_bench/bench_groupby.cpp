
#include <benchmark/benchmark.h>
#include "RipTide.h" // Required for MultiKey.h included from GroupBy.h
#include "GroupBy.h"
#include "numpy_traits.h"
#include <random>

using namespace riptide::benchmark;

template <typename T>
std::vector<T> uniform_random_vector(size_t length, T min, T max)
{
    std::default_random_engine engine;
    std::uniform_int_distribution<T> distribution(min, max);

    auto random = [&]
    {
        return distribution(engine);
    };

    std::vector<T> result(length);
    std::generate(result.begin(), result.end(), random);
    return result;
}

template <GB_FUNCTIONS function, NPY_TYPES TypeCode>
static void BM_GroupByTwo(benchmark::State & state)
{
    int64_t length = state.range(0);
    int64_t bins = state.range(1);

    auto groupby = get_groupby_two_function(function, TypeCode);

    // Allocate input/output/etc buffers
    std::vector<typename riptide::numpy_cpp_type<TypeCode>::type> input(length);
    std::vector<uint8_t> output(bins * groupby.output_type_size);
    std::vector<uint8_t> temp(bins * groupby.temp_type_size);
    std::vector<CountType> count(bins);

    // Populate index vector to simulate real memory access patterns
    auto index = uniform_random_vector<IndexType>(length, 0, bins - 1);

    for (auto _ : state)
    {
        groupby.function(input.data(), index.data(), count.data(), output.data(), length, 0, bins, -1, temp.data());
    }
}

template <typename T>
struct first_and_count
{
    std::vector<T> first;
    std::vector<T> count;
};

template <typename T>
first_and_count<T> generate_first_and_count(int64_t length, int64_t bins)
{
    // Groups are packed such that all elements in the same group are adjacent
    // Each entry is the index to the first element of that group
    auto first = uniform_random_vector<T>(bins, 0, length - 1);

    // Sort it so that generating count is easier
    std::sort(first.begin(), first.end());
    // Set first entry to 0 so that we aren't missing items
    first[0] = 0;

    // Each entry is the number of items in the group
    // We can just compute this by finding the difference between entries in first
    std::vector<T> count(bins);
    for (size_t i = 0; i < count.size() - 1; i++)
        count[i] = first[i + 1] - first[i];
    count.back() = length - first.back();

    return { first, count };
}

template <GB_FUNCTIONS function, NPY_TYPES TypeCode, int64_t funcParam = 0>
static void BM_GroupByX(benchmark::State & state)
{
    int64_t length = state.range(0);
    int64_t bins = state.range(1);

    auto groupby = get_groupby_x_function(function, TypeCode);

    // The values here don't matter
    std::vector<typename riptide::numpy_cpp_type<TypeCode>::type> input(length);
    std::vector<uint8_t> output;

    if constexpr (function >= GB_ROLLING_SUM)
        // Rolling functions require that the output the same size as the input
        output = std::vector<uint8_t>(length * groupby.output_type_size);
    else
        // Other functions require the number of bins
        output = std::vector<uint8_t>(bins * groupby.output_type_size);

    // Populate group, first and count to simulate real memory access patterns
    auto group = uniform_random_vector<IndexType>(length, 0, length - 1);
    auto [first, count] = generate_first_and_count<IndexType>(length, bins);

    for (auto _ : state)
    {
        groupby.function(input.data(), group.data(), first.data(), count.data(), output.data(), 0, bins, length,
                         groupby.output_type_size, funcParam);
    }
}

static void GroupByArguments(benchmark::internal::Benchmark * b)
{
    // Benchmark with 1 million values and 1000 groups
    b->Args({ 1000000, 1000 });
}

#define BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_FUNCTION) \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_INT8>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_INT16>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_INT32>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_INT64>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_UINT8>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_UINT16>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_UINT32>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_UINT64>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_FLOAT>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_DOUBLE>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByTwo<GB_FUNCTION, NPY_LONGDOUBLE>)->Apply(GroupByArguments);

BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_SUM);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_MEAN);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_MIN);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_MAX);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_VAR);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_STD);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_NANSUM);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_NANMEAN);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_NANMIN);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_NANMAX);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_NANVAR);
BENCHMARK_GROUPBY_TWO_ALL_TYPES(GB_NANSTD);

// Parameters for GroupByX functions
namespace
{
    constexpr int64_t unused = 0;
    constexpr int64_t nth = 5;
    constexpr int64_t window = 10;
    constexpr int64_t multiplier = 1e9;
    constexpr int64_t quantile = 0.2 * multiplier;
    constexpr int64_t quantile_and_window = quantile + window * (multiplier + 1);
}

#define BENCHMARK_GROUPBY_X_ALL_TYPES(GB_FUNCTION, funcParam) \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_INT8, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_INT16, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_INT32, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_INT64, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_UINT8, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_UINT16, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_UINT32, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_UINT64, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_FLOAT, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_DOUBLE, funcParam>)->Apply(GroupByArguments); \
    BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_LONGDOUBLE, funcParam>)->Apply(GroupByArguments);

BENCHMARK_GROUPBY_X_ALL_TYPES(GB_FIRST, unused);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_NTH, nth);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_LAST, unused);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_MEDIAN, unused);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_MODE, unused);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_TRIMBR, unused);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_QUANTILE_MULT, quantile);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_ROLLING_SUM, window);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_ROLLING_NANSUM, window);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_ROLLING_DIFF, window);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_ROLLING_SHIFT, window);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_ROLLING_MEAN, window);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_ROLLING_NANMEAN, window);
BENCHMARK_GROUPBY_X_ALL_TYPES(GB_ROLLING_QUANTILE, quantile_and_window);

BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_INT8, 0>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_INT16, 0>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_INT32, 0>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_INT64, 0>)->Apply(GroupByArguments);
// BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_UINT8, 0>)->Apply(GroupByArguments);
// BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_UINT16, 0>)->Apply(GroupByArguments);
// BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_UINT32, 0>)->Apply(GroupByArguments);
// BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_UINT64, 0>)->Apply(GroupByArguments);
// BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_FLOAT, 0>)->Apply(GroupByArguments);
// BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, NPY_DOUBLE, 0>)->Apply(GroupByArguments);
// BENCHMARK(BM_GroupByX<GB_FUNCTION, NPY_LONGDOUBLE, 0>)->Apply(GroupByArguments);