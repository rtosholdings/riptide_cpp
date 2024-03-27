#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "DateTime.h"

#include <format>
#include <optional>
#include <string>
#include <string_view>

const char * rt_strptime(const char * buf, const char * fmt, struct tm * tm);

#define LOGGING(...)

const char COLON = ':';
const char PERIOD = '.';

template <typename T>
struct parsing_error
{
    const T * start;
    const T * end;
    const char * message;
    int64_t position;

    parsing_error(const T * start, const T * end, const char * message, int64_t position)
        : start(start)
        , end(end)
        , message(message)
        , position(position)
    {
    }
};

template <typename T>
class DateTimeParser
{
    const T * pStartCopy = nullptr; // Start of the string
    const T * pStart = nullptr;     // Current parser position
    const T * pEnd = nullptr;       // End of the string

    std::optional<parsing_error<T>> error; // First parsing error encountered

public:
    void ParseDateTimeString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize);
    void ParseDateString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize);
    void ParseTimeString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize);

    RT_FORCEINLINE int64_t ParseNumber(int64_t maxplaces);
    bool ParseSingleNonNumber(T expected);

    void consume_leading_whitespace()
    {
        while (! at_end() && std::isspace(*pStart))
            pStart++;
    }

    void consume_trailing_characters()
    {
        while (! at_end())
        {
            if (! std::isspace(*pStart))
                set_error("Unexpected characters");

            pStart++;
        }
    }

    bool at_end() const
    {
        return pStart >= pEnd || *pStart == '\0';
    }

    void set_error(const char * message)
    {
        if (! error.has_value())
            error = parsing_error<T>(pStartCopy, pEnd, message, pStart - pStartCopy);
    }

    const std::optional<parsing_error<T>> & get_error() const
    {
        return error;
    }
};

//------------------------------------------------------------------
// T is either a char or UCS4 (byte string or unicode string)
// pStart updated on return
//
template <typename T>
RT_FORCEINLINE int64_t ParseDecimal(const T ** ppStart, const T * pEnd)
{
    const T * pStart = *ppStart;

    // parse a number
    int64_t num = 0;
    int64_t places = 0;
    while (pStart < pEnd)
    {
        if (*pStart >= '0' && *pStart <= '9')
        {
            num = num * 10;
            num += (*pStart - '0');
            pStart++;
            places++;

            if (places == 9)
                break;
            continue;
        }
        break;
    }
    // NOTE: could be faster
    while (places <= 6)
    {
        num = num * 1000;
        places += 3;
    }
    while (places < 9)
    {
        num = num * 10;
        places++;
    }

    *ppStart = pStart;
    return num;
}

// Return non-zero if success and increments pStart
template <typename T>
bool DateTimeParser<T>::ParseSingleNonNumber(T expected)
{
    if (pStart < pEnd && (*pStart < '0' || *pStart > '9'))
    {
        if (*pStart != expected)
            set_error("Unexpected delimiter");

        pStart++;
        return true;
    }

    return false;
}

//------------------------------------------------------------------
// T is either a char or UCS4 (byte string or unicode string)
// will skip whitespace upfront
// will stop at end, stop at nonnumber, or when places reached
// pStart updated on return
template <typename T>
RT_FORCEINLINE int64_t DateTimeParser<T>::ParseNumber(int64_t maxplaces)
{
    // skip non numbers in front
    while (pStart < pEnd)
    {
        if (*pStart < '0' || *pStart > '9')
        {
            set_error("Unexpected characters");
            pStart++;
            continue;
        }
        break;
    }

    // parse a number
    int64_t num = 0;
    int64_t places = 0;
    while (pStart < pEnd)
    {
        if (*pStart >= '0' && *pStart <= '9')
        {
            num = num * 10;
            num += (*pStart - '0');
            pStart++;
            places++;
            if (places == maxplaces)
                break;
            continue;
        }
        break;
    }

    return num;
}

//---------------------------------------
// Look for HH:MM:SS
// T is either a char or UCS4 (byte string or unicode string)
//
template <typename T>
void DateTimeParser<T>::ParseTimeString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize)
{
    for (int64_t i = 0; i < arrayLength; i++)
    {
        pStartCopy = pStart = &pString[i * itemSize];
        pEnd = pStart + itemSize;

        consume_leading_whitespace();

        if (at_end())
        {
            pOutNanoTime[i] = 0;
            continue;
        }

        int64_t hour = 0;
        int64_t minute = 0;
        int64_t seconds = 0;

        bool bSuccess = 1;
        pOutNanoTime[i] = 0;

        hour = ParseNumber(2);
        if (hour >= 24)
            bSuccess = 0;

        // could check for colon here...
        if (ParseSingleNonNumber(':'))
        {
            // could check for colon here...
            minute = ParseNumber(2);
            if (minute >= 60)
                bSuccess = 0;

            // seconds are not required
            if (ParseSingleNonNumber(':'))
            {
                seconds = ParseNumber(2);
                if (seconds >= 60)
                    bSuccess = 0;
            }

            if (bSuccess)
            {
                LOGGING("time: %lld:%lld:%lld\n", hour, minute, seconds);
                pOutNanoTime[i] = 1000000000LL * ((hour * 3600) + (minute * 60) + seconds);
            }

            // check for milli/micro/etc seconds
            if (pStart < pEnd && *pStart == PERIOD)
            {
                // skip decimal
                pStart++;
                int64_t decimal = ParseDecimal<T>(&pStart, pEnd);
                if (bSuccess)
                    pOutNanoTime[i] += decimal;
            }
        }

        consume_trailing_characters();
    }
}

int64_t MONTH_SPLITS[12] = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };

int64_t MONTH_SPLITS_LEAP[12] = { 0,   // Jan
                                  31,  // Feb
                                  60,  // Mar
                                  91,  // Apr
                                  121, // May
                                  152, // June
                                  182, // July
                                  213, // Aug
                                  244, // Sep
                                  274, // Oct
                                  305, // Nov
                                  335 };

int64_t NANOS_PER_SECOND = 1000000000LL;
int64_t NANOS_PER_MINUTE = NANOS_PER_SECOND * 60;
int64_t NANOS_PER_HOUR = NANOS_PER_MINUTE * 60;
int64_t NANOS_PER_DAY = NANOS_PER_HOUR * 24;

//==========================================================
// return -1 for error
int64_t YearToEpochNano(int64_t year, int64_t month, int64_t day)
{
    // NOTE: This is using simplified logic. Per Wikipedia the rules are:
    //   "Every year that is exactly divisible by four is a leap year,
    //    except for years that are exactly divisible by 100,
    //    but these centurial years are leap years if they are exactly divisible by 400.
    //   For example, the years 1700, 1800, and 1900 are not leap years, but the years 1600 and 2000 are."
    // We limit to 2099 since 2100 is not a leap year and detecting that requires implementing the full algorithm.
    // The corresponding lookup tables in rt_datetime are likewise limited to 2099.
    if (year >= 1970 && year <= 2099 && month >= 1 && month <= 12 && day >= 1 && day <= 31)
    {
        // check for leap.
        if ((year % 4) == 0)
        {
            day = MONTH_SPLITS_LEAP[month - 1] + day - 1;
        }
        else
        {
            day = MONTH_SPLITS[month - 1] + day - 1;
        }
        // calculate how many leap years we skipped over
        // 1973 has 1 leap year (3)
        // 1977 has 2 leap years (7)
        // 1972 has 0 leap years (2)
        year = year - 1970;
        int64_t extradays = (year + 1) / 4;
        int64_t daysSinceEpoch = (year * 365) + extradays + day;

        return (daysSinceEpoch * NANOS_PER_DAY);
    }

    // return invalid year
    return -1;
}

//---------------------------------------
// Look for YYYYMMDD  YYYYY-MM-DD
// T is either a char or UCS4 (byte string or unicode string)
//
template <typename T>
void DateTimeParser<T>::ParseDateString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize)
{
    for (int64_t i = 0; i < arrayLength; i++)
    {
        pStartCopy = pStart = &pString[i * itemSize];
        pEnd = pStart + itemSize;

        consume_leading_whitespace();

        if (at_end())
        {
            pOutNanoTime[i] = 0;
            continue;
        }

        int64_t year = 0;
        int64_t month = 0;
        int64_t day = 0;

        year = ParseNumber(4);

        bool expect_delimiter = ParseSingleNonNumber('-');

        month = ParseNumber(2);

        if (expect_delimiter != ParseSingleNonNumber('-'))
            set_error(expect_delimiter ? "Expected delimiter" : "Unexpected delimiter");

        day = ParseNumber(2);

        LOGGING("date: %lld:%lld:%lld\n", year, month, day);
        int64_t result = YearToEpochNano(year, month, day);
        if (result < 0)
            result = 0;

        consume_trailing_characters();

        pOutNanoTime[i] = result;
    }
}

//---------------------------------------
// Look for YYYYMMDD  YYYYY-MM-DD  then HH:MM:SS.mmmuuunnn
// T is either a char or UCS4 (byte string or unicode string)
// Anything invalid is set to 0
template <typename T>
void DateTimeParser<T>::ParseDateTimeString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize)
{
    for (int64_t i = 0; i < arrayLength; i++)
    {
        pStartCopy = pStart = &pString[i * itemSize];
        pEnd = pStart + itemSize;

        consume_leading_whitespace();

        if (at_end())
        {
            pOutNanoTime[i] = 0;
            continue;
        }

        int64_t year = 0;
        int64_t month = 0;
        int64_t day = 0;
        bool bSuccess = 1;

        year = ParseNumber(4);

        bool expect_delimiter = ParseSingleNonNumber('-');

        month = ParseNumber(2);

        if (expect_delimiter != ParseSingleNonNumber('-'))
            set_error(expect_delimiter ? "Expected delimiter" : "Unexpected delimiter");

        day = ParseNumber(2);

        LOGGING("date: %lld:%lld:%lld\n", year, month, day);
        int64_t yearresult = YearToEpochNano(year, month, day);

        if (yearresult < 0)
        {
            bSuccess = 0;
            yearresult = 0;
        }

        int64_t hour = 0;
        int64_t minute = 0;
        int64_t seconds = 0;

        if (pStart < pEnd && (*pStart < '0' || *pStart > '9'))
        {
            if (*pStart != 'T' && *pStart != ' ')
                set_error("Unexpected delimiter");

            pStart++;
            hour = ParseNumber(2);
            if (hour >= 24)
                bSuccess = 0;

            // could check for colon here...
            if (ParseSingleNonNumber(':'))
            {
                minute = ParseNumber(2);
                if (minute >= 60)
                    bSuccess = 0;

                // seconds do not have to exist
                if (ParseSingleNonNumber(':'))
                {
                    seconds = ParseNumber(2);
                    if (seconds >= 60)
                        bSuccess = 0;
                }
            }
        }

        if (bSuccess)
        {
            LOGGING("time: %lld:%lld:%lld\n", hour, minute, seconds);
            yearresult += (1000000000LL * ((hour * 3600) + (minute * 60) + seconds));
        }

        // check for milli/micro/etc seconds
        if (pStart < pEnd && *pStart == PERIOD)
        {
            // skip decimal
            pStart++;
            int64_t decimal = ParseDecimal<T>(&pStart, pEnd);

            if (bSuccess)
                yearresult += decimal;
        }

        consume_trailing_characters();

        pOutNanoTime[i] = yearresult;
    }
}

template <typename T>
void parsing_error_to_py_warning(const parsing_error<T> & error);

template <>
void parsing_error_to_py_warning<char>(const parsing_error<char> & error)
{
    std::string datetime(error.start, error.end);
    PyErr_WarnFormat(PyExc_RuntimeWarning, 1, "%s in \"%s\" at position %lld", error.message, datetime.c_str(), error.position);
}

template <>
void parsing_error_to_py_warning<uint32_t>(const parsing_error<uint32_t> & error)
{
    PyObject * datetime = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, error.start, error.end - error.start);
    PyErr_WarnFormat(PyExc_RuntimeWarning, 1, "%s in \"%U\" at position %lld", error.message, datetime, error.position);
    Py_XDECREF(datetime);
}

//--------------------------------------------------------------
// Arg1: input numpy array time == assumes int64_t for now
// HH:MM:SS format
//
// Output: int64_t numpy array with nanos
//
PyObject * TimeStringToNanos(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr))
    {
        return NULL;
    }

    int32_t dType = PyArray_TYPE(inArr);

    PyArrayObject * outArray = NULL;
    int64_t arrayLength = ArrayLength(inArr);

    // TODO: Check to make sure inArr and timeArr sizes are the same
    if (dType != NPY_STRING && dType != NPY_UNICODE)
    {
        PyErr_Format(PyExc_ValueError,
                     "TimeStringToNanos first argument must be a "
                     "bytes or unicode string array");
        return NULL;
    }

    // Dont bother allocating if we cannot call the function
    outArray = AllocateNumpyArray(1, (npy_intp *)&arrayLength, NPY_INT64);

    if (outArray)
    {
        int64_t itemSize = PyArray_ITEMSIZE(inArr);
        int64_t * pNanoTime = (int64_t *)PyArray_BYTES(outArray);
        char * pString = (char *)PyArray_BYTES(inArr);

        if (dType == NPY_STRING)
        {
            DateTimeParser<char> parser;
            parser.ParseTimeString(pNanoTime, arrayLength, pString, itemSize);

            const auto & maybe_error = parser.get_error();
            if (maybe_error)
                parsing_error_to_py_warning(*maybe_error);
        }
        else
        {
            DateTimeParser<uint32_t> parser;
            parser.ParseTimeString(pNanoTime, arrayLength, (uint32_t *)pString, itemSize / 4);

            const auto & maybe_error = parser.get_error();
            if (maybe_error)
                parsing_error_to_py_warning(*maybe_error);
        }
        return (PyObject *)outArray;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

//--------------------------------------------------------------
// Arg1: input numpy array time == assumes int64_t for now
// YYYYMMDD or YYYY-MM-DD
//
// Output: int64_t numpy array with nanos
//
PyObject * DateStringToNanos(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr))
    {
        return NULL;
    }

    int32_t dType = PyArray_TYPE(inArr);

    PyArrayObject * outArray = NULL;
    int64_t arrayLength = ArrayLength(inArr);

    // TODO: Check to make sure inArr and timeArr sizes are the same
    if (dType != NPY_STRING && dType != NPY_UNICODE)
    {
        PyErr_Format(PyExc_ValueError,
                     "DateStringToNanos first argument must be a "
                     "bytes or unicode string array");
        return NULL;
    }

    // Dont bother allocating if we cannot call the function
    outArray = AllocateNumpyArray(1, (npy_intp *)&arrayLength, NPY_INT64);

    if (outArray)
    {
        int64_t itemSize = PyArray_ITEMSIZE(inArr);
        int64_t * pNanoTime = (int64_t *)PyArray_BYTES(outArray);
        char * pString = (char *)PyArray_BYTES(inArr);

        if (dType == NPY_STRING)
        {
            DateTimeParser<char> parser;
            parser.ParseDateString(pNanoTime, arrayLength, pString, itemSize);

            const auto & maybe_error = parser.get_error();
            if (maybe_error)
                parsing_error_to_py_warning(*maybe_error);
        }
        else
        {
            DateTimeParser<uint32_t> parser;
            parser.ParseDateString(pNanoTime, arrayLength, (uint32_t *)pString, itemSize / 4);

            const auto & maybe_error = parser.get_error();
            if (maybe_error)
                parsing_error_to_py_warning(*maybe_error);
        }
        return (PyObject *)outArray;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

//--------------------------------------------------------------
// Arg1: input numpy array time == assumes int64_t for now
// YYYYMMDD or YYYY-MM-DD
//
// Output: int64_t numpy array with nanos
//
PyObject * DateTimeStringToNanos(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr = NULL;

    if (! PyArg_ParseTuple(args, "O!", &PyArray_Type, &inArr))
    {
        return NULL;
    }

    int32_t dType = PyArray_TYPE(inArr);

    PyArrayObject * outArray = NULL;
    int64_t arrayLength = ArrayLength(inArr);

    // TODO: Check to make sure inArr and timeArr sizes are the same
    if (dType != NPY_STRING && dType != NPY_UNICODE)
    {
        PyErr_Format(PyExc_ValueError,
                     "DateTimeStringToNanos first argument must "
                     "be a bytes or unicode string array");
        return NULL;
    }

    // Dont bother allocating if we cannot call the function
    outArray = AllocateNumpyArray(1, (npy_intp *)&arrayLength, NPY_INT64);

    if (outArray)
    {
        int64_t itemSize = PyArray_ITEMSIZE(inArr);
        int64_t * pNanoTime = (int64_t *)PyArray_BYTES(outArray);
        char * pString = (char *)PyArray_BYTES(inArr);

        if (dType == NPY_STRING)
        {
            DateTimeParser<char> parser;
            parser.ParseDateTimeString(pNanoTime, arrayLength, pString, itemSize);

            const auto & maybe_error = parser.get_error();
            if (maybe_error)
                parsing_error_to_py_warning(*maybe_error);
        }
        else
        {
            DateTimeParser<uint32_t> parser;
            parser.ParseDateTimeString(pNanoTime, arrayLength, (uint32_t *)pString, itemSize / 4);

            const auto & maybe_error = parser.get_error();
            if (maybe_error)
                parsing_error_to_py_warning(*maybe_error);
        }
        return (PyObject *)outArray;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

//---------------------------------------
// Look for YYYYMMDD  YYYYY-MM-DD  then HH:MM:SS.mmmuuunnn
// T is either a char or UCS4 (byte string or unicode string)
// Anything invalid is set to 0
template <typename T>
void ParseStrptime(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize, const char * pFmt)
{
    tm timeBack;

    char * pTempString = (char *)WORKSPACE_ALLOC(itemSize + 8);
    const char * pEnd = pTempString + itemSize;

    for (int64_t i = 0; i < arrayLength; i++)
    {
        const T * pStart = &pString[i * itemSize];

        // copy string over (or convert unicode)
        // so we can add a terminating zero
        for (int64_t j = 0; j < itemSize; j++)
        {
            pTempString[j] = (char)pStart[j];
        }
        pTempString[itemSize] = 0;

        // for every run, reset the tm struct
        memset(&timeBack, 0, sizeof(tm));

        const char * pStop;
        pStop = rt_strptime(pTempString, pFmt, &timeBack);

        int64_t yearresult = 0;

        // Check if we parsed it correctly
        if (pStop && timeBack.tm_year != 0)
        {
            // printf("strtime1: [%lld] %d %d %d %d\n", i, timeBack.tm_year + 1900,
            // timeBack.tm_mon, timeBack.tm_mday, timeBack.tm_yday); printf("strtime2:
            // [%lld] %d %d %d %d\n", i, timeBack.tm_hour, timeBack.tm_min,
            // timeBack.tm_sec, timeBack.tm_wday);

            // 1900 is the strptime base year
            yearresult = YearToEpochNano(timeBack.tm_year + 1900, timeBack.tm_mon + 1, timeBack.tm_mday);

            // addin hours min secs
            yearresult += (1000000000LL * ((timeBack.tm_hour * 3600LL) + (timeBack.tm_min * 60LL) + timeBack.tm_sec));

            // check for milli/micro/etc seconds
            if (*pStop == PERIOD)
            {
                // skip decimal
                pStop++;

                yearresult += ParseDecimal<char>(&pStop, pEnd);
            }
        }
        else
        {
            // printf("!!FAIL strtime1: [%lld] %s %s\n", i, pTempString, pFmt);
        }

        if (yearresult >= 0)
        {
            pOutNanoTime[i] = yearresult;
        }
        else
        {
            pOutNanoTime[i] = 0;
        }
    }

    WORKSPACE_FREE(pTempString);
}

//--------------------------------------------------------------
// Arg1: input numpy string array time
// Arg2: strptime format string (MUSTBE BE BYESTRING)
//
// Output: int64_t numpy array with nanos
//
PyObject * StrptimeToNanos(PyObject * self, PyObject * args)
{
    PyArrayObject * inArr = NULL;
    const char * strTimeFormat;
    Py_ssize_t strTimeFormatSize;

    if (! PyArg_ParseTuple(args, "O!y#", &PyArray_Type, &inArr, &strTimeFormat, &strTimeFormatSize))
    {
        return NULL;
    }

    int32_t dType = PyArray_TYPE(inArr);

    PyArrayObject * outArray = NULL;
    int64_t arrayLength = ArrayLength(inArr);

    // TODO: Check to make sure inArr and timeArr sizes are the same
    if (dType != NPY_STRING && dType != NPY_UNICODE)
    {
        PyErr_Format(PyExc_ValueError,
                     "StrptimeToNanos first argument must be a "
                     "unicode string or bytes string array");
        return NULL;
    }

    // Dont bother allocating if we cannot call the function
    outArray = AllocateNumpyArray(1, (npy_intp *)&arrayLength, NPY_INT64);

    if (outArray)
    {
        int64_t itemSize = PyArray_ITEMSIZE(inArr);
        int64_t * pNanoTime = (int64_t *)PyArray_BYTES(outArray);
        char * pString = (char *)PyArray_BYTES(inArr);

        if (dType == NPY_STRING)
        {
            ParseStrptime<char>(pNanoTime, arrayLength, pString, itemSize, strTimeFormat);
        }
        else
        {
            ParseStrptime<uint32_t>(pNanoTime, arrayLength, (uint32_t *)pString, itemSize / 4, strTimeFormat);
        }

        return (PyObject *)outArray;
    }

    Py_INCREF(Py_None);
    return Py_None;
};
