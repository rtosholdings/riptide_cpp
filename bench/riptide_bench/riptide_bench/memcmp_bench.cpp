#include <benchmark/benchmark.h>

#include <array>
#include <cfloat>

namespace
{
#if 0
    char *arena1;
    char *arena2;

    void init(benchmark::State const &)
    {
        arena1 = new char[4096ull*1024*1024];
        arena2 = new char[4096ull*1024*1024];
    }

    void teardown(benchmark::State const &)
    {
        delete arena1;
        delete arena2;
    }

    void bench_memcmp(benchmark::State & state)
    {
        for (auto _ : state)
        {
            auto x = memcmp(arena1, arena2, state.range(0));
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_memcmp)->RangeMultiplier(64)->Range(1,4096ull*1024*1024)->Setup(init)->Teardown(teardown)->Repetitions(30)->ReportAggregatesOnly(true);

    #define MEMCMP_NEW(ARG1, ARG2, ARG3, ARG4) \
        { \
            ARG1 = 0; \
            const char * pSrc1 = ARG2; \
            const char * pSrc2 = ARG3; \
            int64_t length = ARG4; \
            while (length >= 8) \
            { \
                uint64_t * p1 = (uint64_t *)pSrc1; \
                uint64_t * p2 = (uint64_t *)pSrc2; \
                if (*p1 != *p2) \
                { \
                    ARG1 = 1; \
                    length = 0; \
                    break; \
                } \
                length -= 8; \
                pSrc1 += 8; \
                pSrc2 += 8; \
            } \
            if (length >= 4) \
            { \
                uint32_t * p1 = (uint32_t *)pSrc1; \
                uint32_t * p2 = (uint32_t *)pSrc2; \
                if (*p1 != *p2) \
                { \
                    ARG1 = 1; \
                    length = 0; \
                } \
                else \
                { \
                    length -= 4; \
                    pSrc1 += 4; \
                    pSrc2 += 4; \
                } \
            } \
            while (length > 0) \
            { \
                if (*pSrc1 != *pSrc2) \
                { \
                    ARG1 = 1; \
                    break; \
                } \
                length -= 1; \
                pSrc1 += 1; \
                pSrc2 += 1; \
            } \
        }

    void bench_memcmp_new(benchmark::State & state)
    {
        for (auto _ : state)
        {
            int x{};
            MEMCMP_NEW(x, arena1, arena2, state.range(0));
            benchmark::DoNotOptimize(x);
            benchmark::ClobberMemory();
        }
    }

    BENCHMARK(bench_memcmp_new)->RangeMultiplier(64)->Range(1,4096ull*1024*1024)->Setup(init)->Teardown(teardown)->Repetitions(30)->ReportAggregatesOnly(true);
#endif
}
