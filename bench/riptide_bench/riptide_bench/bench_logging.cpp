#include "logging/logger.h"

#include <benchmark/benchmark.h>

namespace
{
    using namespace riptide::logging;
    auto & logg = logger::get();

    static void bench_logging(benchmark::State & state)
    {
        for (auto _ : state)
        {
            auto num_threads = state.range(0);
            auto num_logs = state.range(1);
            auto produce = [=](int id)
            {
                for (auto i = 0; i < num_logs; i++)
                {
                    logg.log(loglevel::debug, "Thread: {0} log number: {1}", id, i);
                }
            };

            auto consume = [=]()
            {
                while (logg.active())
                {
                    auto curr{ logg.receive() };
                    if (! curr)
                        break;

                    benchmark::DoNotOptimize(curr.value().level);
                    benchmark::DoNotOptimize(curr.value().message);
                }
            };

            logg.enable({ .max_size = 1'000'000'000, .level = loglevel::debug });

            std::thread consumer{ consume };
            std::vector<std::thread> threads;

            for (auto i = 0; i < num_threads; i++)
            {
                threads.emplace_back(produce, i);
            }

            for (auto & t : threads)
            {
                t.join();
            }

            logg.disable();
            consumer.join();
        }
    }

    BENCHMARK(bench_logging)
        ->Unit(benchmark::TimeUnit::kMillisecond)
        ->ArgsProduct({ { 2, 4, 8, 16, 32, 64 }, { 1000, 10'000, 100'000 } })
        ->UseRealTime();
}