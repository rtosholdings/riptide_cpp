#pragma once

#include "logger.h"
#include "../Defs.h"

#include <unordered_map>

namespace riptide::logging
{
    class registry
    {
    public:
        auto get_logger(const std::string & name)
        {
            if (not loggers_.contains(name))
                loggers_[name] = std::make_shared<logger>(name, service_);
            return loggers_[name];
        }

        auto get_logger_names()
        {
            std::vector<std::string> res(loggers_.size());
            std::transform(loggers_.begin(), loggers_.end(), res.begin(),
                           [](auto p)
                           {
                               return p.first;
                           });
            return res;
        }

        auto service()
        {
            return service_;
        }

        static registry & get()
        {
            static registry instance;
            return instance;
        }

    private:
        registry()
            : service_{ std::make_shared<logging_service>() }
        {
        }

        registry(const registry &) = delete;

        std::shared_ptr<logging_service> service_;
        std::unordered_map<std::string, std::shared_ptr<logger>> loggers_;
    };

    RT_DLLEXPORT std::shared_ptr<logger> get_logger(const std::string & name = {});
    RT_DLLEXPORT std::shared_ptr<logging_service> get_service();
    RT_DLLEXPORT std::vector<std::string> get_logger_names();
}