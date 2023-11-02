#include "logging/logger.h"
#include "Logger.h"

#include <thread>
#include <vector>
#include <chrono>
#include <queue>

namespace
{
    PyObject * logging_lib = NULL;
    PyObject * log_sink = NULL;

    riptide::logging::logger & logger = riptide::logging::logger::get();
    using log_record = riptide::logging::log_record;
    using loglevel = riptide::logging::loglevel;

    std::optional<PyObject *> exception_callback;
    std::optional<std::string> exception_message;

    void UpdateLogLevel()
    {
        auto level_attr = PyObject_GetAttrString(log_sink, "level");
        if (! PyLong_Check(level_attr))
        {
            Py_XDECREF(level_attr);
            return;
        }
        int32_t level = PyLong_AsLong(level_attr);
        Py_XDECREF(level_attr);

        if (level >= static_cast<int>(loglevel::notset) and level <= static_cast<int>(loglevel::crtical))
            logger.set_level(static_cast<loglevel>(level));
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
            while (logger.active())
            {
                batch.clear();

                auto diff = [](auto last_sent)
                {
                    auto delta{ std::chrono::steady_clock::now() - last_sent };
                    return std::chrono::duration_cast<std::chrono::milliseconds>(delta);
                };

                auto batch_size{ logger.batch_size() };
                while (batch.size() < batch_size and diff(last_sent) < logger.interval())
                {
                    // if there is no log, then this thread will sleep here.
                    auto curr_log{ logger.receive() };

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

                for (size_t index = 0; index < batch.size(); index++)
                {
                    auto res = PyObject_CallMethod(log_sink, "log", "is", static_cast<int>(batch[index].level),
                                                   batch[index].message.c_str());
                    Py_XDECREF(res);
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
    log_sink = PyObject_CallMethod(logging_lib, "getLogger", "s", "riptable.riptide_cpp");
}

void CleanupLogging()
{
    logger.disable();
}

PyObject * EnableLogging(PyObject * self, PyObject * args)
{
    if (logger.active())
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

    logger.enable({ .batch_size = bsize, .flush_interval = std::chrono::milliseconds(interval), .max_size = msize });
    UpdateLogLevel();
    logger.set_handler(LoggingHandler);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject * DisableLogging(PyObject * self, PyObject * args)
{
    if (! logger.active())
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
        logger.disable(std::chrono::milliseconds(timeout));
    else
        logger.disable();

    Py_END_ALLOW_THREADS;
    Py_INCREF(Py_None);
    return Py_None;
}