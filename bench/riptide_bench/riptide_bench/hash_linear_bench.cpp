#include "RipTide.h"
#include "HashLinear.h"
#include "flat_hash_map.h"

#include "benchmark/benchmark.h"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

namespace
{
    std::vector<uint64_t> test_data(128ULL * 1024ULL * 1024ULL);
    std::random_device dev{};
    CHashLinear<uint64_t, int64_t> hasher{};
    fhm_hasher<uint64_t> new_hasher{};
    std::vector<uint64_t> needles(1024ULL * 1024ULL);
    std::array<int64_t, 1024ULL * 1024ULL> output;
    std::array<int8_t, 1024ULL * 1024ULL> bools;

    void bench_IsMemberHash64(benchmark::State & state)
    {
        std::mt19937 engine(dev());
        std::uniform_int_distribution<uint64_t> dist(3002950000, test_data.size() + 3002950000);
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::generate(std::begin(needles), std::end(needles), [&] { return dist(engine); });

        for (auto _ : state)
        {
            IsMemberHash64(needles.size(), needles.data(), test_data.size(), test_data.data(), output.data(), bools.data(), 8, HASH_MODE(1), 0);
            benchmark::DoNotOptimize(test_data.data());
            benchmark::DoNotOptimize(output.data());
            benchmark::DoNotOptimize(bools.data());
            benchmark::DoNotOptimize(needles.data());
            benchmark::ClobberMemory();
        }

    }

    BENCHMARK(bench_IsMemberHash64)->Unit(benchmark::kMillisecond);

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

    BENCHMARK(bench_MakeHashLocation)->Unit(benchmark::kMillisecond);

    void bench_is_member(benchmark::State & state)
    {
        std::mt19937 engine(dev());
        std::uniform_int_distribution<uint64_t> dist(3002950000, test_data.size() + 3002950000);
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::generate(std::begin(needles), std::end(needles), [&] { return dist(engine); });

        for (auto _ : state)
        {
            is_member(needles.size(), reinterpret_cast<char const *>(needles.data()), test_data.size(), reinterpret_cast<char const *>(test_data.data()), output.data(), bools.data(), uint64_t{});
            benchmark::DoNotOptimize(test_data.data());
            benchmark::DoNotOptimize(output.data());
            benchmark::DoNotOptimize(bools.data());
            benchmark::DoNotOptimize(needles.data());
            benchmark::ClobberMemory();
        }
    }
    
    BENCHMARK(bench_is_member)->Unit(benchmark::kMillisecond);

    void bench_make_hash(benchmark::State & state)
    {
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
        for (auto _ : state)
        {
            new_hasher.make_hash(test_data.size(), reinterpret_cast<char const*>(test_data.data()), 0);
            benchmark::DoNotOptimize(test_data.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_make_hash)->Unit(benchmark::kMillisecond);
}
