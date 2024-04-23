#include <benchmark/benchmark.h>
#include "RipTide.h"
#include "numpy_traits.h"
#include "Reduce.h"

using namespace riptide::benchmark;

template <REDUCE_FUNCTIONS function, NPY_TYPES TypeCode>
static void BM_Reduce(benchmark::State & state)
{
    int64_t length = state.range(0);
    std::vector<typename riptide::numpy_cpp_type<TypeCode>::type> input(length);

    for (auto _ : state)
    {
        call_reduce_function(function, TypeCode, input.data(), length);
    }
}

static void BM_ReduceArguments(benchmark::internal::Benchmark * b)
{
    // Benchmark with input length of 1m values
    b->Args({ 1000000 });
}

#define BENCHMARK_REDUCE_ALL_TYPES(REDUCE_FUNCTION) \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_INT8>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_INT16>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_INT32>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_INT64>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_UINT8>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_UINT16>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_UINT32>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_UINT64>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_FLOAT>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_DOUBLE>)->Apply(BM_ReduceArguments); \
    BENCHMARK(BM_Reduce<REDUCE_FUNCTION, NPY_LONGDOUBLE>)->Apply(BM_ReduceArguments);

BENCHMARK_REDUCE_ALL_TYPES(REDUCE_SUM);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANSUM);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_MEAN);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANMEAN);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_VAR);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANVAR);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_STD);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANSTD);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_MIN);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANMIN);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_MAX);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANMAX);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_ARGMIN);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANARGMIN);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_ARGMAX);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_NANARGMAX);
BENCHMARK_REDUCE_ALL_TYPES(REDUCE_MIN_NANAWARE);