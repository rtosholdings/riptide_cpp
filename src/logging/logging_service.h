
#pragma once

#include "timer.h"
#include "waiting_deque.h"
#include "log_common.h"

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
    namespace details
    {
        template <class T>
        T && validate_arg(T && t)
        {
            return std::forward<T>(t);
        }

        template <typename T, typename U = std::remove_cvref_t<T>>
        constexpr bool is_char_type_v = std::is_same_v<U, char> || std::is_same_v<U, unsigned char> ||
                                        std::is_same_v<U, wchar_t> || std::is_same_v<U, char16_t> || std::is_same_v<U, char32_t>;

        template <class T>
            requires is_char_type_v<T>
        T * validate_arg(T * const t)
        {
            if (not t)
            {
                static T result[]{ '(', 'n', 'u', 'l', 'l', ')', '\0' };
                return result;
            }
            return t;
        }
    }

    class logging_service
    {
    public:
        ~logging_service()
        {
            disable();
        }

        template <typename... Args>
        void log(const std::string & name, loglevel level, const log_format & format, Args &&... args)
        {
            try
            {
                log_record record{ .name = name,
                                   .message = std::vformat(format.format, std::make_format_args(details::validate_arg(args)...)),
                                   .level = level,
                                   .loc = format.loc,
                                   .time = std::chrono::system_clock::now() };

                logs_.push_back(std::move(record));
            }
            catch (std::exception const & e)
            {
                log(name, loglevel::error, "Caught exception: {}\nwhen formatting: {}", e.what(), format.format);
            }
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
            return ! logs_.empty() or ! shutdown_;
        }

        bool enabled() noexcept
        {
            return not shutdown_;
        }

        void wakeup()
        {
            logs_.cancel_wait();
        }

        void shutdown()
        {
            shutdown_ = true;
        }

        void enable(const log_config config = log_config())
        {
            if (not shutdown_)
                return;

            shutdown_ = false;
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
            if (shutdown_)
                return;

            shutdown();
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

        static logging_service & get() noexcept
        {
            static logging_service instance;
            return instance;
        }

    private:
        details::waiting_deque<log_record> logs_;
        details::timer flusher_;

        std::thread handler_;

        std::atomic<bool> shutdown_ = true;

        uint32_t batch_size_;
        std::chrono::milliseconds interval_;
        uint32_t max_size_;
    };
}