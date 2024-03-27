#pragma once

#include <chrono>
#include <source_location>
#include <string>

namespace riptide::logging
{
    using namespace std::literals;

    enum class loglevel : int32_t
    {
        notset = 0,
        debug = 10,
        info = 20,
        warn = 30,
        error = 40,
        critical = 50,
        none = 60
    };

    struct log_config
    {
        uint32_t batch_size = 50;
        std::chrono::milliseconds flush_interval = 1000ms;
        uint32_t max_size = 1'000'000;
    };

    struct log_record
    {
        std::string name;
        std::string message;
        loglevel level;
        std::source_location loc;
        std::chrono::time_point<std::chrono::system_clock> time;
    };

    struct log_format
    {
        std::string_view format;
        std::source_location loc;

        template <class FormatString>
        log_format(FormatString format, const std::source_location & loc = std::source_location::current())
            : format{ format }
            , loc{ loc }
        {
        }
    };
}