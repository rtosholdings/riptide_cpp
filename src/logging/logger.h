#pragma once

#include "logging_service.h"

#include <memory>

namespace riptide::logging
{
    class logger
    {
    public:
        logger(const std::string_view name, const std::shared_ptr<logging_service> logger)
            : name_{ name }
            , logger_{ logger }
        {
        }

        template <typename... Args>
        void log(const loglevel level, const log_format & format, Args &&... args)
        {
            if (not should_log(level))
                return;

            logger_->log(name_, level, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void debug(const log_format & format, Args &&... args)
        {
            log(loglevel::debug, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void warn(const log_format & format, Args &&... args)
        {
            log(loglevel::warn, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void info(const log_format & format, Args &&... args)
        {
            log(loglevel::info, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void error(const log_format & format, Args &&... args)
        {
            log(loglevel::error, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void critical(const log_format & format, Args &&... args)
        {
            log(loglevel::critical, format, std::forward<Args>(args)...);
        }

        [[nodiscard]] bool should_log(const loglevel level) noexcept
        {
            return logger_->enabled() and level >= logLevel_;
        }

        void set_level(const loglevel level) noexcept
        {
            logLevel_ = level;
        }

    private:
        std::string name_;
        std::shared_ptr<logging_service> logger_;
        loglevel logLevel_ = loglevel::notset;
    };
}
