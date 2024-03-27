#include "logging/logging.h"

#include "Logger.h"

#include <thread>
#include <vector>
#include <chrono>
#include <queue>

namespace
{
    PyObject * logging_lib = NULL;

    auto service = riptide::logging::get_service();
    using log_record = riptide::logging::log_record;
    using loglevel = riptide::logging::loglevel;

    std::unordered_map<std::string, PyObject *> sinks;
    std::optional<PyObject *> exception_callback;
    std::optional<std::string> exception_message;

    void UpdateLogLevel()
    {
        for (const auto & [name, sink] : sinks)
        {
            auto level_attr = PyObject_GetAttrString(sink, "level");
            if (! PyLong_Check(level_attr))
            {
                Py_XDECREF(level_attr);
                continue;
                ;
            }

            int32_t level = PyLong_AsLong(level_attr);
            Py_XDECREF(level_attr);

            auto logger{ riptide::logging::get_logger(name) };
            if (level >= static_cast<int>(loglevel::notset) and level <= static_cast<int>(loglevel::critical))
                logger->set_level(static_cast<loglevel>(level));
        }
    }

    void LoggingHandler()
    {
        PyGILState_STATE state;
        auto last_sent{ std::chrono::steady_clock::now() };

        auto handle_exception_message = [&](const char * msg)
        {
            if (_Py_IsFinalizing())
                return;

            if (! PyGILState_Check())
                state = PyGILState_Ensure();

            exception_message = msg;
            if (exception_callback)
            {
                auto res{ PyObject_CallFunction(exception_callback.value(), "s", msg) };
                Py_XDECREF(res);
            }
            PyGILState_Release(state);
        };

        try
        {
            std::vector<log_record> batch;
            // make sure we have the GIL initially.
            while (service->active())
            {
                batch.clear();

                auto diff = [](auto last_sent)
                {
                    auto delta{ std::chrono::steady_clock::now() - last_sent };
                    return std::chrono::duration_cast<std::chrono::milliseconds>(delta);
                };

                auto batch_size{ service->batch_size() };
                while (batch.size() < batch_size and diff(last_sent) < service->interval())
                {
                    // if there is no log, then this thread will sleep here.
                    auto curr_log{ service->receive() };

                    if (! curr_log)
                        break;

                    batch.push_back(std::move(curr_log.value()));
                }
                last_sent = std::chrono::steady_clock::now();

                // if py is finalizing, we can't call the log function.
                if (_Py_IsFinalizing())
                    return;

                // grab gil
                state = PyGILState_Ensure();

                UpdateLogLevel();

                for (auto & msg : batch)
                {
                    auto sink{ sinks[msg.name] };

                    auto log_method{ PyObject_GetAttrString(sink, "log") };
                    if (not log_method)
                        continue;

                    auto source_loc{ msg.loc };

                    auto source_info{ Py_BuildValue("{s:s, s:i, s:i, s:s}", "filename", source_loc.file_name(), "line",
                                                    source_loc.line(), "column", source_loc.column(), "function",
                                                    source_loc.function_name()) };

                    auto timestamp_ns{ msg.time.time_since_epoch().count() };

                    auto extra{ Py_BuildValue("{s:O, s:I}", "source_info", source_info, "timestamp", timestamp_ns) };

                    auto args{ Py_BuildValue("(iN)", static_cast<int>(msg.level),
                                             PyUnicode_Decode(msg.message.c_str(), msg.message.size(), NULL, "replace")) };
                    auto kwargs{ Py_BuildValue("{s:O}", "extra", extra) };

                    if (args and kwargs)
                    {
                        auto res{ PyObject_Call(log_method, args, kwargs) };
                        Py_XDECREF(res);
                    }
                    Py_DECREF(log_method);
                }

                // release
                PyGILState_Release(state);
            }
        }
        catch (const std::exception & ex)
        {
            handle_exception_message(ex.what());
        }
        catch (...)
        {
            handle_exception_message("Unknown error caught.");
        }
    }
}

void SetupLogging()
{
    // these ref should stay alive during the entire lifetime of the module.
    logging_lib = PyImport_ImportModule("logging");

    for (auto & name : riptide::logging::get_logger_names())
    {
        auto logger_name{ "riptable.riptide_cpp" + (name.empty() ? "" : "." + name) };
        auto sink{ PyObject_CallMethod(logging_lib, "getLogger", "s", logger_name.c_str()) };
        if (sink == nullptr)
            continue;
        sinks[name] = sink;
    }
}

void CleanupLogging()
{
    service->disable();
}

PyObject * EnableLogging(PyObject * self, PyObject * args)
{
    if (service->active())
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    int64_t interval = 0;
    uint32_t bsize = 0;
    uint32_t msize = 0;
    PyObject * callback;

    if (! PyArg_ParseTuple(args, "lIIO", &interval, &bsize, &msize, &callback))
    {
        PyErr_Format(PyExc_ValueError, "Bad argments passed");
        return NULL;
    }

    // reset previous exception or callbacks
    exception_message = std::nullopt;
    exception_callback = std::nullopt;

    if (PyFunction_Check(callback))
        exception_callback = callback;

    service->enable({ .batch_size = bsize, .flush_interval = std::chrono::milliseconds(interval), .max_size = msize });
    UpdateLogLevel();
    service->set_handler(LoggingHandler);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * GetRiptideLoggers(PyObject * self, PyObject * args)
{
    auto names{ riptide::logging::get_logger_names() };
    auto list{ PyList_New(names.size()) };
    for (size_t i = 0; i < names.size(); i++)
    {
        auto logger_name{ "riptable.riptide_cpp" + (names[i].empty() ? "" : "." + names[i]) };
        PyList_SetItem(list, i, PyUnicode_FromString(logger_name.c_str()));
    }
    return list;
}

PyObject * DisableLogging(PyObject * self, PyObject * args)
{
    if (! service->active())
    {
        Py_INCREF(Py_None);
        return Py_None;
    }

    int64_t timeout = 0;
    if (! PyArg_ParseTuple(args, "l", &timeout))
    {
        PyErr_Format(PyExc_ValueError, "Invalid arguments passed");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    if (timeout != -1)
        service->disable(std::chrono::milliseconds(timeout));
    else
        service->disable();

    Py_END_ALLOW_THREADS;
    Py_INCREF(Py_None);
    return Py_None;
}