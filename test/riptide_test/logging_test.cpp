#include "ut_core.h"
#include "tuple_util.h"
#include "logging/logging.h"

#include <tuple>
#include <array>
#include <thread>
#include <vector>
#include <type_traits>
#include <unordered_map>

using namespace boost::ut;
using riptide_utility::ut::file_suite;
using namespace riptide::logging;
using namespace riptide_utility::internal;

namespace
{
    auto service = get_service();
    auto logger = get_logger();

    void produce(size_t log_count, int id)
    {
        for (size_t i = 0; i < log_count; i++)
        {
            logger->log(loglevel::debug, "{0} {1}", id, i);
        }
    }

    auto producer(size_t thread_count, size_t log_count)
    {
        std::vector<std::thread> threads;

        for (size_t i = 0; i < thread_count; i++)
            threads.emplace_back(produce, log_count, i);

        return threads;
    }

    void consume(std::vector<log_record> & result)
    {
        while (service->active())
        {
            auto curr{ service->receive() };

            if (! curr)
                continue;

            result.push_back(std::move(curr.value()));
        }
    }

    template <size_t T>
    using int_type = std::integral_constant<size_t, T>;

    using SupportedThreadSize = std::tuple<int_type<1>, int_type<2>, int_type<16>>;
    using SupportedLogSize = std::tuple<int_type<1>, int_type<100>, int_type<1000>>;

    using SupportedArgs = decltype(tuple_prod(SupportedThreadSize{}, SupportedLogSize{}));

    struct logging_tester
    {
        template <typename T>
        void operator()()
        {
            using ThreadSizeType = std::tuple_element_t<0, T>;
            using LogSizeType = std::tuple_element_t<1, T>;

            constexpr auto Threads = ThreadSizeType::value;
            constexpr auto Logs = LogSizeType::value;

            std::vector<log_record> result;

            logger->set_level(loglevel::debug);
            service->enable({ .max_size = 1'000'000'000 });

            auto prods{ producer(Threads, Logs) };
            std::thread cons{ consume, std::ref(result) };

            for (auto & t : prods)
                t.join();

            service->shutdown();

            cons.join();
            service->disable();

            std::unordered_map<int, int> log_count;
            while (! result.empty())
            {
                auto log{ std::move(result.back()) };
                result.pop_back();

                std::istringstream iss(log.message);

                expect(! iss.fail());

                int info[2];
                for (auto & i : info)
                {
                    iss >> i;
                }

                log_count[info[0]]++;
            }

            expect(log_count.size() == Threads);

            for (auto & kv : log_count)
            {
                expect(kv.second == Logs);
            }
        }
    };

    struct nullptr_format_validation_tester
    {
        template <typename T>
        void operator()()
        {
            char const * expected{ "(null)" };
            auto const actual{ details::validate_arg(static_cast<T const *>(nullptr)) };
            expect(actual != nullptr) << fatal;
            for (int i{ 0 }; i < 5; ++i)
            {
                expect(actual[i] != 0) << fatal;
                expect(actual[i] == static_cast<T>(expected[i]));
            }
            expect(! actual[6]);
        }
    };

    file_suite riptide_ops = []
    {
        // TODO: move this to riptide_python_test
        "test_logging_normal"_test = logging_tester{} | SupportedArgs{};

        "test_format_validation"_test =
            nullptr_format_validation_tester{} | std::tuple<char, unsigned char, wchar_t, char16_t, char32_t>{};
    };
}
