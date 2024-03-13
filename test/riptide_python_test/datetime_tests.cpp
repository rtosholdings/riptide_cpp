#include "CommonInc.h"
#include "riptide_python_test.h"
#include "ut_core.h"

#include <format>

using namespace boost::ut;
using namespace riptide_python_test::internal;

using riptide_utility::ut::file_suite;

namespace
{
    PyObject * call_datetime_function(std::string function_name, std::string argument)
    {
        PyObject * const function = get_named_function(riptide_module_p, function_name.c_str());
        PyObject * const list = Py_BuildValue("[s]", argument.c_str());
        PyObject * const array = PyArray_FromAny(list, nullptr, 0, 0, 0, nullptr);
        PyObject * const result = PyObject_CallFunction(function, "O", array);
        Py_XDECREF(list);
        Py_XDECREF(array);
        return result;
    }

    class capture_warnings
    {
        PyObject * mlogging;
        pyobject_ptr handler;
        pyobject_ptr logger;
        pyobject_ptr queue;

    public:
        capture_warnings()
        {
            mlogging = PyImport_ImportModule("logging");
            PyObject_CallMethod(mlogging, "captureWarnings", "O", Py_True);

            auto mhandlers{ get_named_function(mlogging, "handlers") };
            auto mqueue{ PyImport_ImportModule("queue") };

            logger = pyobject_ptr{ PyObject_CallMethod(mlogging, "getLogger", "s", "py.warnings") };
            queue = pyobject_ptr{ PyObject_CallFunction(get_named_function(mqueue, "SimpleQueue"), nullptr) };
            handler = pyobject_ptr{ PyObject_CallFunction(get_named_function(mhandlers, "QueueHandler"), "O", queue.get()) };

            PyObject_CallMethod(logger.get(), "addHandler", "O", handler.get());
        }

        ~capture_warnings()
        {
            PyObject_CallMethod(mlogging, "captureWarnings", "O", Py_False);
            PyObject_CallMethod(logger.get(), "removeHandler", "O", handler.get());
        }

        std::vector<std::string> get_warnings()
        {
            std::vector<std::string> result;

            while (PyObject_CallMethod(queue.get(), "empty", nullptr) == Py_False)
            {
                pyobject_ptr log_record{ PyObject_CallMethod(queue.get(), "get", nullptr) };
                pyobject_ptr message{ PyObject_CallMethod(log_record.get(), "getMessage", nullptr) };
                result.emplace_back(PyUnicode_AsUTF8(message.get()));
            }

            return result;
        }
    };

    struct test_case
    {
        std::string argument;
        int64_t expected;
        std::string warning;

        test_case(std::string argument_, int64_t expected_, std::string warning_ = "")
            : argument(argument_)
            , expected(expected_)
            , warning(warning_)
        {
        }
    };

    void check_datetime_result(pyobject_ptr && result, const test_case & tc)
    {
        expect(no_pyerr());
        expect(fatal(result != nullptr));
        expect(fatal(PyArray_Check(result.get()))); // TODO check result

        PyArrayObject * array = reinterpret_cast<PyArrayObject *>(result.get());
        expect(fatal(PyArray_TYPE(array) == NPY_INT64));
        expect(fatal(PyArray_SIZE(array) == 1));

        int64_t nanoseconds = *reinterpret_cast<int64_t *>(PyArray_DATA(array));
        expect(nanoseconds == tc.expected) << std::format("Expected result of \"{}\" to be {}, got {}", tc.argument, tc.expected,
                                                          nanoseconds);
    }

    auto expect_function_call_ok(std::string function_name)
    {
        return [=](const test_case & tc)
        {
            capture_warnings cw;

            pyobject_ptr result{ call_datetime_function(function_name, tc.argument) };
            check_datetime_result(std::move(result), tc);

            auto warnings = cw.get_warnings();
            expect(warnings.empty()) << std::format("Expected 0 warnings, got {}", warnings.size());
        };
    }

    auto expect_function_call_warn(std::string function_name)
    {
        return [=](const test_case & tc)
        {
            capture_warnings cw;

            pyobject_ptr result{ call_datetime_function(function_name, tc.argument) };
            check_datetime_result(std::move(result), tc);

            auto warnings = cw.get_warnings();
            expect(fatal(warnings.size() == 1)) << std::format("Expected 1 warning, got {}", warnings.size());
            expect(warnings[0].find(tc.warning) != std::string::npos)
                << std::format("Expected warning to contain \"{}\", warning was \"{}\"", tc.warning, warnings[0]);
        };
    }

    file_suite datetime_functions = []
    {
        "date_string_parsing_ok"_test = expect_function_call_ok("DateStringToNanos") | std::vector<test_case>{
            test_case("", 0),
            test_case(" ", 0),
            test_case("    ", 0),
            test_case("20240506", 1714953600000000000),
            test_case("2024-05-06", 1714953600000000000),
            test_case(" 20240506", 1714953600000000000),
            test_case("20240506 ", 1714953600000000000),
            test_case(" 20240506 ", 1714953600000000000),
            test_case("1969-01-01", 0),
            test_case("2134-01-01", 0),
            test_case("1979-13-01", 0),
            test_case("1969-01-42", 0),
        };

        "date_string_parsing_warn"_test = expect_function_call_warn("DateStringToNanos") | std::vector<test_case>{
            test_case("2024-0506", 1714953600000000000, "Expected delimiter"),
            test_case("202405-06", 1714953600000000000, "Unexpected delimiter"),
            test_case("2024:05:06", 1714953600000000000, "Unexpected delimiter"),
            test_case("huh 20240506", 1714953600000000000, "Unexpected characters"),
            test_case("2024huh05hmm06", 1714953600000000000, "Unexpected delimiter"),
            test_case("20240506 junk", 1714953600000000000, "Unexpected characters"),
        };

        "time_string_parsing_ok"_test = expect_function_call_ok("TimeStringToNanos") | std::vector<test_case>{
            test_case("", 0),
            test_case(" ", 0),
            test_case("    ", 0),
            test_case("12", 0),
            test_case(" 12:08", 43680000000000),
            test_case("12:08:34 ", 43714000000000),
            test_case(" 12:08:34.123 ", 43714123000000),
            test_case("12:08:34.123456789", 43714123456789),
            test_case("56:12:12.123", 0),
            test_case("12:93:12.123", 0),
            test_case("12:12:93.123", 0),
        };

        "time_string_parsing_warn"_test = expect_function_call_warn("TimeStringToNanos") | std::vector<test_case>{
            test_case("12-08", 43680000000000, "Unexpected delimiter"),
            test_case("12h08m", 43680000000000, "Unexpected delimiter"),
            test_case("1s", 3600000000000, "Unexpected delimiter"),
            test_case("2m", 7200000000000, "Unexpected delimiter"),
            test_case("2H", 7200000000000, "Unexpected delimiter"),
            test_case("12.08*34.123", 43714123000000, "Unexpected delimiter"),
            test_case("2 any other string!", 7200000000000, "Unexpected delimiter"),
            test_case("56:12:12.123 random junk", 0, "Unexpected characters"),
            test_case("12:93:12.123 random junk", 0, "Unexpected characters"),
            test_case("12:12:93.123 random junk", 0, "Unexpected characters"),
        };

        "datetime_string_parsing_ok"_test = expect_function_call_ok("DateTimeStringToNanos") | std::vector<test_case>{
            test_case("", 0),
            test_case(" ", 0),
            test_case("    ", 0),
            test_case("20240506 12:08", 1714997280000000000),
            test_case("20240506 12:08:34 ", 1714997314000000000),
            test_case(" 20240506T12:08:34.123", 1714997314123000000),
            test_case(" 20240506 12:08:34.123456789 ", 1714997314123456789),
            test_case("1969-01-01T12:12:12.123", 0),
            test_case("1979-13-01T12:12:12.123", 0),
            test_case("1979-01-42T12:12:12.123", 0),
            test_case("1979-01-01T56:12:12.123", 283996800000000000),
            test_case("1979-01-01T12:93:12.123", 283996800000000000),
            test_case("1979-01-01T12:12:93.123", 283996800000000000),
        };

        "datetime_string_parsing_warn"_test = expect_function_call_warn("DateTimeStringToNanos") | std::vector<test_case>{
            test_case("2024y05m06d12h08m34s123ms", 1714997314000000000, "Unexpected delimiter"),
            test_case("2024y05m06d12h08m34.123ms", 1714997314123000000, "Unexpected delimiter"),
            test_case("the date is 2024-05-06", 1714953600000000000, "Unexpected characters"),
            test_case("2024-05-06 random junk", 1714953600000000000, "Unexpected characters"),
            test_case("1979-13-01T12:12:12.123 random junk", 0, "Unexpected characters"),
            test_case("1979-01-01T12:93:12.123 random junk", 283996800000000000, "Unexpected characters"),
        };
    };
}