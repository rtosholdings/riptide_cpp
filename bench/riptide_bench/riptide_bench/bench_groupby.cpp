
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

template <GB_FUNCTIONS function, typename InputType>
static void BM_GroupByTwo(benchmark::State & state)
{
    int64_t length = state.range(0);
    int64_t bins = state.range(1);

    auto groupby = get_groupby_two_function(function, riptide::numpy_type_code<InputType>::value);

    // Allocate input/output/etc buffers
    std::vector<InputType> input(length);
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

template <GB_FUNCTIONS function, typename InputType, int64_t funcParam = 0>
static void BM_GroupByX(benchmark::State & state)
{
    int64_t length = state.range(0);
    int64_t bins = state.range(1);

    auto groupby = get_groupby_x_function(function, riptide::numpy_type_code<InputType>::value);

    // The values here don't matter
    std::vector<InputType> input(length);
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
    b->Args({ 16384, 16384 });
}

BENCHMARK(BM_GroupByTwo<GB_SUM, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_MEAN, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_MIN, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_MAX, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_VAR, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_STD, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_NANSUM, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_NANMEAN, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_NANMIN, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_NANMAX, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_NANVAR, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByTwo<GB_NANSTD, double>)->Apply(GroupByArguments);

// Parameters for GroupByX functions
namespace
{
    constexpr int64_t nth = 5;
    constexpr int64_t window = 10;
    constexpr int64_t multiplier = 1e9;
    constexpr int64_t quantile = 0.2 * multiplier;
    constexpr int64_t quantile_and_window = quantile + window * (multiplier + 1);
}

BENCHMARK(BM_GroupByX<GB_FIRST, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_NTH, double, nth>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_LAST, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_MEDIAN, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_MODE, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_TRIMBR, double>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_QUANTILE_MULT, double, quantile>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_SUM, double, window>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_NANSUM, double, window>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_DIFF, double, window>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_SHIFT, double, window>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_COUNT, int64_t, window>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_NANMEAN, double, window>)->Apply(GroupByArguments);
BENCHMARK(BM_GroupByX<GB_ROLLING_QUANTILE, double, quantile_and_window>)->Apply(GroupByArguments);