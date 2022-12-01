#include "RipTide.h"
#include "HashLinear.h"
#include "is_member_tg.h"
#include "operation_traits.h"

#include "benchmark/benchmark.h"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

namespace
{
    std::vector<uint64_t> test_data{};
    std::random_device dev{};
    CHashLinear<uint64_t, int64_t> hasher{};
    std::vector<uint64_t> needles(1024ULL * 1024ULL);
    std::array<int32_t, 1024ULL * 1024ULL> output{};
    std::array<int8_t, 1024ULL * 1024ULL> bools{};
    volatile int32_t size_type{ 8 };

    void bench_h_unsorted_MakeHashLocation(benchmark::State & state)
    {
        test_data.resize(state.range(0));
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
        for (auto _ : state)
        {
            hasher.MakeHashLocation(test_data.size(), test_data.data(), 0);
            benchmark::DoNotOptimize(test_data.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_h_unsorted_MakeHashLocation)
        ->Unit(benchmark::kMillisecond)
        ->Arg(2ULL * 1024 * 1024)
        ->Arg(2ULL * 1024 * 1024)
        ->UseRealTime()
        ->MeasureProcessCPUTime();

    void bench_h_sorted_MakeHashLocation(benchmark::State & state)
    {
        test_data.resize(state.range(0));
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        for (auto _ : state)
        {
            hasher.MakeHashLocation(test_data.size(), test_data.data(), 0);
            benchmark::DoNotOptimize(test_data.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_h_sorted_MakeHashLocation)
        ->Unit(benchmark::kMillisecond)
        ->Arg(2ULL * 1024 * 1024)
        ->UseRealTime()
        ->MeasureProcessCPUTime();

    void bench_h_unsorted_IsMemberHash32(benchmark::State & state)
    {
        test_data.resize(state.range(0));
        std::mt19937 engine(dev());
        std::uniform_int_distribution<uint64_t> dist(3002950000, test_data.size() + 3002950000);
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
        std::generate(std::begin(needles), std::end(needles),
                      [&]
                      {
                          return dist(engine);
                      });

        for (auto _ : state)
        {
            IsMemberHash32(needles.size(), needles.data(), test_data.size(), test_data.data(), output.data(), bools.data(),
                           size_type, HASH_MODE(1), 0);
            benchmark::DoNotOptimize(test_data.data());
            benchmark::DoNotOptimize(output.data());
            benchmark::DoNotOptimize(bools.data());
            benchmark::DoNotOptimize(needles.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_h_unsorted_IsMemberHash32)
        ->Unit(benchmark::kMillisecond)
        ->Arg(2ULL * 1024 * 1024)
        ->UseRealTime()
        ->MeasureProcessCPUTime();

    void bench_h_sorted_IsMemberHash32(benchmark::State & state)
    {
        test_data.resize(state.range(0));
        std::mt19937 engine(dev());
        std::uniform_int_distribution<uint64_t> dist(3002950000, test_data.size() + 3002950000);
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::generate(std::begin(needles), std::end(needles),
                      [&]
                      {
                          return dist(engine);
                      });

        for (auto _ : state)
        {
            IsMemberHash32(needles.size(), needles.data(), test_data.size(), test_data.data(), output.data(), bools.data(),
                           size_type, HASH_MODE(1), 0);
            benchmark::DoNotOptimize(test_data.data());
            benchmark::DoNotOptimize(output.data());
            benchmark::DoNotOptimize(bools.data());
            benchmark::DoNotOptimize(needles.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_h_sorted_IsMemberHash32)
        ->Unit(benchmark::kMillisecond)
        ->Arg(2ULL * 1024 * 1024)
        ->UseRealTime()
        ->MeasureProcessCPUTime();

    void bench_h_2unsorted_make_hash_tbb(benchmark::State & state)
    {
        test_data.resize(state.range(0));

        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });

        oneapi::tbb::task_arena::constraints const local_arena_setters{ oneapi::tbb::numa_node_id{ 0 },
                                                                        static_cast<int32_t>(state.range(1)) };
        oneapi::tbb::task_arena local_arena{ local_arena_setters };

        local_arena.execute(
            [&]()
            {
                hashing_graph<uint64_t, int32_t> hasher{ test_data.data(), static_cast<size_t>(state.range(0)), 1 };

                for (auto _ : state)
                {
                    hasher();
                    benchmark::DoNotOptimize(hasher.grouped_hashes.begin());
                    benchmark::ClobberMemory();
                }
            });
    }

    BENCHMARK(bench_h_2unsorted_make_hash_tbb)
        ->Unit(benchmark::kMillisecond)
        ->ArgsProduct({ { 2ULL * 1024 * 1024 }, { 1, 2, 4, 8, 12, 16, 20, 24, 28 } })
        ->UseRealTime()
        ->MeasureProcessCPUTime();

    void bench_h_2unsorted_is_member_tbb(benchmark::State & state)
    {
        test_data.resize(state.range(0));
        std::mt19937 engine(dev());
        std::uniform_int_distribution<uint64_t> dist(3002950000, test_data.size() + 3002950000);
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);
        std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
        std::generate(std::begin(needles), std::end(needles),
                      [&]
                      {
                          return dist(engine);
                      });

        riptable_cpp::data_type_t variant = riptable_cpp::uint64_traits{};

        for (auto _ : state)
        {
            is_member_for_type(needles.size(), reinterpret_cast<char const *>(needles.data()), 1, test_data.size(),
                               reinterpret_cast<char const *>(test_data.data()), 1, output.data(), bools.data(), variant,
                               state.range(1), std::make_index_sequence<std::variant_size_v<riptable_cpp::data_type_t>>{});
            benchmark::DoNotOptimize(output.data());
            benchmark::DoNotOptimize(bools.data());
            benchmark::DoNotOptimize(needles.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_h_2unsorted_is_member_tbb)
        ->Unit(benchmark::kMillisecond)
        ->ArgsProduct({ { 2ULL * 1024 * 1024 }, { 1, 2, 4, 8, 12, 16, 20, 24, 28 } })
        ->UseRealTime()
        ->MeasureProcessCPUTime();

    void bench_h_2sorted_make_hash_tbb(benchmark::State & state)
    {
        test_data.resize(state.range(0));
        std::iota(std::begin(test_data), std::end(test_data), 3002954500);

        oneapi::tbb::task_arena::constraints const local_arena_setters{ oneapi::tbb::numa_node_id{ 0 },
                                                                        static_cast<int32_t>(state.range(1)) };
        oneapi::tbb::task_arena local_arena{ local_arena_setters };

        local_arena.execute(
            [&]()
            {
                hashing_graph<uint64_t, int32_t> hasher{ test_data.data(), static_cast<size_t>(state.range(0)), 1 };

                for (auto _ : state)
                {
                    hasher();
                    benchmark::DoNotOptimize(hasher.grouped_hashes.begin());
                    benchmark::ClobberMemory();
                }
            });
    }

    BENCHMARK(bench_h_2sorted_make_hash_tbb)
        ->Unit(benchmark::kMillisecond)
        ->ArgsProduct({ { 2ULL * 1024 * 1024 }, { 1, 2, 4, 8, 12, 16, 20, 24, 28 } })
        ->UseRealTime()
        ->MeasureProcessCPUTime();

    void bench_h_2sorted_is_member_tbb(benchmark::State & state)
    {
        test_data.resize(state.range(0));
        std::mt19937 engine(dev());
        std::uniform_int_distribution<uint64_t> dist(3002950000, test_data.size() + 3002950000);
        std::iota(std::begin(test_data), std::end(test_data), 1);
        std::generate(std::begin(needles), std::end(needles),
                      [&]
                      {
                          return dist(engine);
                      });

        for (auto _ : state)
        {
            riptable_cpp::data_type_t variant = riptable_cpp::uint64_traits{};
            is_member_for_type(needles.size(), reinterpret_cast<char const *>(needles.data()), 1, test_data.size(),
                               reinterpret_cast<char const *>(test_data.data()), 1, output.data(), bools.data(), variant,
                               state.range(1), std::make_index_sequence<std::variant_size_v<riptable_cpp::data_type_t>>{});
            benchmark::DoNotOptimize(test_data.data());
            benchmark::DoNotOptimize(output.data());
            benchmark::DoNotOptimize(bools.data());
            benchmark::DoNotOptimize(needles.data());
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_h_2sorted_is_member_tbb)
        ->Unit(benchmark::kMillisecond)
        ->ArgsProduct({ { 2ULL * 1024 * 1024 }, { 2, 4, 8, 12, 16, 20, 24, 28 } })
        ->UseRealTime()
        ->MeasureProcessCPUTime();
}
