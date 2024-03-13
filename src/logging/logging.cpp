#include "logging.h"

#include <algorithm>

namespace riptide::logging
{
    std::shared_ptr<logger> get_logger(const std::string & name)
    {
        return registry::get().get_logger(name);
    }

    std::shared_ptr<logging_service> get_service()
    {
        return registry::get().service();
    }

    std::vector<std::string> get_logger_names()
    {
        return registry::get().get_logger_names();
    }
}
