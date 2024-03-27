
#include "riptide_python_test.h"
#include "logging/logging.h"
#include "tuple_util.h"
#include "Logger.h"

#include "ut_core.h"

#include <array>
#include <ranges>
#include <vector>
#include <thread>

using namespace boost::ut;
using riptide_utility::ut::file_suite;
using namespace riptide_utility::internal;
using namespace riptide_python_test::internal;
using riptide_python_test::internal::get_named_function;

using loglevel = riptide::logging::loglevel;
using logger = riptide::logging::logger;

namespace
{
    auto generate_log(const std::vector<std::shared_ptr<logger>> & loggers, size_t nthreads, size_t nlogs)
    {
        std::vector<std::thread> threads;

        auto gen = [&](const std::vector<std::shared_ptr<logger>> & loggers, size_t thread_id, size_t nlogs)
        {
            auto log_count{ nlogs / loggers.size() };
            for (auto & logg : loggers)
            {
                for (size_t i = 0; i < log_count; i++)
                    logg->debug("[{}] {}", thread_id, i);
            }
        };

        for (size_t th = 0; th < nthreads; th++)
            threads.emplace_back(gen, loggers, th, nlogs);

        for (auto & th : threads)
            th.join();
    }

    template <size_t T>
    using int_type = std::integral_constant<size_t, T>;

    using SupportedThreadSize = std::tuple<int_type<1>, int_type<16>>;
    using SupportedLogSize = std::tuple<int_type<500>>;
    using SupportedBatchSize = std::tuple<int_type<1>, int_type<50>, int_type<500>>;

    using SupportedArgs = decltype(tuple_prod(tuple_prod(SupportedThreadSize{}, SupportedLogSize{}), SupportedBatchSize{}));

    struct logging_handler_tester
    {
        static void exec(size_t nloggers, size_t nthreads, size_t nlogs, size_t batch_size)
        {
            auto mlogging{ PyImport_ImportModule("logging") };
            auto mhandlers{ get_named_function(mlogging, "handlers") };
            auto mqueue{ PyImport_ImportModule("queue") };
            auto enable_logging{ get_named_function(riptide_module_p, "EnableLogging") };
            auto disable_logging{ get_named_function(riptide_module_p, "DisableLogging") };

            pyobject_ptr queue{ PyObject_CallFunction(get_named_function(mqueue, "Queue"), nullptr) };
            expect(queue != nullptr);

            pyobject_ptr queue_handler{ PyObject_CallFunction(get_named_function(mhandlers, "QueueHandler"), "O", queue.get()) };
            expect(queue_handler != nullptr);

            std::vector<PyObject *> py_loggers;
            std::vector<std::shared_ptr<logger>> loggers;

            for (size_t i = 0; i < nloggers; i++)
            {
                auto name{ std::to_string(i) };
                loggers.push_back(riptide::logging::get_logger(name));

                auto py_logger{ PyObject_CallMethod(mlogging, "getLogger", "s",
                                                    std::format("riptable.riptide_cpp.{}", name).c_str()) };
                expect(py_logger != nullptr);

                py_loggers.push_back(py_logger);
                PyObject_CallMethod(py_logger, "setLevel", "i", static_cast<int>(loglevel::debug));

                auto add_result{ PyObject_CallMethod(py_logger, "addHandler", "O", queue_handler.get()) };
                expect(add_result != nullptr);
            }

            SetupLogging();
            // EnableLogging(interval, batch_size, max_size, call_back)
            auto enable_res{ PyObject_CallFunction(enable_logging, "iiiO", 1000, (int32_t)batch_size, 100'000, Py_None) };
            expect(enable_res != nullptr);

            generate_log(loggers, nthreads, nlogs);

            // DisableLogging(timeout), -1 = no timeout
            auto disable_res{ PyObject_CallFunction(disable_logging, "i", -1) };
            expect(disable_res != nullptr);

            auto received{ PyLong_AsLong(PyRun_String("len(handler.queue.queue)", Py_eval_input,
                                                      Py_BuildValue("{sO}", "handler", queue_handler.get()),
                                                      Py_BuildValue("{}"))) };

            expect(received == static_cast<long>(nthreads * nlogs))
                << "received=" << received << ", expected=" << (nthreads * nlogs);

            for (auto & py_logger : py_loggers)
            {
                auto remove_res{ PyObject_CallMethod(py_logger, "removeHandler", "O", queue_handler.get()) };
                expect(remove_res != nullptr);
            }
        }
    };

    template <size_t NumLoggers>
    struct logging_handler_test
    {
        template <typename T>
        void operator()()
        {
            using ThreadLogPair = std::tuple_element_t<0, T>;
            using ThreadSizeType = std::tuple_element_t<0, ThreadLogPair>;
            using LogSizeType = std::tuple_element_t<1, ThreadLogPair>;

            using BatchSizeType = std::tuple_element_t<1, T>;

            constexpr auto Threads = ThreadSizeType::value;
            constexpr auto Logs = LogSizeType::value;
            constexpr auto BatchSize = BatchSizeType::value;

            static_assert(Logs % NumLoggers == 0, "The number of logs must be divisible by number of subloggers for testing.");
            logging_handler_tester::exec(NumLoggers, Threads, Logs, BatchSize);
        }
    };

    file_suite riptide_ops = []
    {
        "logging_handler_test_single"_test = logging_handler_test<1>{} | SupportedArgs{};
        "logging_handler_test_multiple"_test = logging_handler_test<5>{} | SupportedArgs{};
    };
}