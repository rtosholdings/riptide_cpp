#include "RipTide.h"
#include "HashLinear.h"

#include "benchmark/benchmark.h"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

namespace
{
    std::vector<uint64_t> test_data(1024ULL * 1024ULL * 1024ULL);
    std::random_device dev{};
    CHashLinear<uint64_t, int64_t> hasher{};

    void bench_MakeHashLocation(benchmark::State & state)
    {
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
        for (auto _ : state)
        {
            hasher.MakeHashLocation(test_data.size(), test_data.data(), 0);
            benchmark::DoNotOptimize(test_data.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_MakeHashLocation);
}
