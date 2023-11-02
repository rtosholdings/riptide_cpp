#include "timer.h"
#include "waiting_deque.h"

#include <string>
#include <iostream>
#include <memory>
#include <fstream>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string_view>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include <format>

using namespace std::literals;

namespace riptide::logging
{
    enum class loglevel : int32_t
    {
        notset = 0,
        debug = 10,
        info = 20,
        warn = 30,
        error = 40,
        crtical = 50,
        none = 60
    };

    struct log_config
    {
        uint32_t batch_size = 50;
        std::chrono::milliseconds flush_interval = 1000ms;
        uint32_t max_size = 1'000'000;
        loglevel level = loglevel::warn;
    };

    struct log_record
    {
        std::string message;
        loglevel level;
    };

    class logger
    {
    public:
        ~logger()
        {
            disable();
        }

        template <typename... Args>
        void log(loglevel level, const char * format, const Args &... args)
        {
            if (level < logLevel_)
                return;

            log_record record{ .message = std::vformat(format, std::make_format_args(args...)), .level = level };
            logs_.push_back(std::move(record));
        }

        std::optional<log_record> receive()
        {
            if (! active())
                return std::nullopt;

            auto log = logs_.pop_front();
            if (! log)
                return std::nullopt;

            return log;
        }

        bool active() noexcept
        {
            return ! logs_.empty() or logLevel_ != loglevel::none;
        }

        bool should_log(loglevel level) noexcept
        {
            return level >= logLevel_;
        }

        void set_level(loglevel level) noexcept
        {
            if (level == loglevel::notset)
                level = loglevel::none;
            logLevel_ = level;
        }

        void wakeup()
        {
            logs_.cancel_wait();
        }

        void enable(const log_config config = log_config())
        {
            if (logLevel_ != loglevel::none)
                return;

            set_level(config.level);
            interval_ = config.flush_interval;
            batch_size_ = config.batch_size;
            max_size_ = config.max_size;

            logs_.set_max_size(max_size_);

            flusher_.set_interval(
                [this]
                {
                    wakeup();
                },
                interval_);
        }

        void disable(const std::optional<std::chrono::milliseconds> timeout = std::nullopt)
        {
            logLevel_ = loglevel::none;
            wakeup();

            if (handler_.joinable())
            {
                if (! timeout)
                {
                    handler_.join();
                }
                else
                {
                    auto future = std::async(std::launch::async, &std::thread::join, &handler_);
                    if (future.wait_for(timeout.value()) == std::future_status::timeout)
                    {
                        // if couldn't join within timeout, force thread to drop all current logs and force join
                        logs_.clear();
                        wakeup();
                    }
                }
            }

            flusher_.cancel();
        }

        uint32_t batch_size() noexcept
        {
            return batch_size_;
        }

        std::chrono::milliseconds interval() noexcept
        {
            return interval_;
        }

        uint32_t max_size() noexcept
        {
            return max_size_;
        }

        template <class Handler>
        void set_handler(Handler handler)
        {
            handler_ = std::thread{ handler };
        }

        static inline logger & get() noexcept
        {
            static logger instance;
            return instance;
        }

    private:
        details::waiting_deque<log_record> logs_;
        details::timer flusher_;

        std::thread handler_;

        loglevel logLevel_ = loglevel::none;
        uint32_t batch_size_;
        std::chrono::milliseconds interval_;
        uint32_t max_size_;
    };
}
