#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "DateTime.h"

const char * rt_strptime(const char * buf, const char * fmt, struct tm * tm);

#define LOGGING(...)

const char COLON = ':';
const char PERIOD = '.';

//------------------------------------------------------------------
// T is either a char or UCS4 (byte string or unicode string)
// pStart updated on return
//
template <typename T>
FORCE_INLINE int64_t ParseDecimal(const T ** ppStart, const T * pEnd)
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
    return num;
}

// Return non-zero if success and increments pStart
#define ParseSingleNonNumber(pStartX, pEndX) ((pStartX < pEndX && (*pStartX < '0' || *pStartX > '9')) ? ++pStartX : 0)

//------------------------------------------------------------------
// T is either a char or UCS4 (byte string or unicode string)
// will skip whitespace upfront
// will stop at end, stop at nonnumber, or when places reached
// pStart updated on return
template <typename T>
FORCE_INLINE int64_t ParseNumber(const T ** ppStart, const T * pEnd, int64_t maxplaces)
{
    const T * pStart = *ppStart;

    // skip non numbers in front
    while (pStart < pEnd)
    {
        if (*pStart < '0' || *pStart > '9')
        {
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

    // update pStart
    *ppStart = pStart;
    return num;
}

//---------------------------------------
// Look for HH:MM:SS
// T is either a char or UCS4 (byte string or unicode string)
//
template <typename T>
void ParseTimeString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize)
{
    for (int64_t i = 0; i < arrayLength; i++)
    {
        const T * pStart = &pString[i * itemSize];
        const T * pEnd = pStart + itemSize;

        int64_t hour = 0;
        int64_t minute = 0;
        int64_t seconds = 0;

        bool bSuccess = 0;

        hour = ParseNumber<T>(&pStart, pEnd, 2);

        // could check for colon here...
        if (hour < 24 && ParseSingleNonNumber(pStart, pEnd))
        {
            // could check for colon here...
            minute = ParseNumber<T>(&pStart, pEnd, 2);

            if (minute < 60)
            {
                // seconds are not required
                if (ParseSingleNonNumber(pStart, pEnd))
                {
                    seconds = ParseNumber<T>(&pStart, pEnd, 2);
                }

                bSuccess = 1;

                LOGGING("time: %lld:%lld:%lld\n", hour, minute, seconds);
                pOutNanoTime[i] = 1000000000LL * ((hour * 3600) + (minute * 60) + seconds);

                // check for milli/micro/etc seconds
                if (pStart < pEnd && *pStart == PERIOD)
                {
                    // skip decimal
                    pStart++;
                    pOutNanoTime[i] += ParseDecimal<T>(&pStart, pEnd);
                }
            }
        }

        if (! bSuccess)
        {
            pOutNanoTime[i] = 0;
        }
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
void ParseDateString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize)
{
    for (int64_t i = 0; i < arrayLength; i++)
    {
        const T * pStart = &pString[i * itemSize];
        const T * pEnd = pStart + itemSize;

        int64_t year = 0;
        int64_t month = 0;
        int64_t day = 0;

        year = ParseNumber<T>(&pStart, pEnd, 4);
        // could check for dash here...
        month = ParseNumber<T>(&pStart, pEnd, 2);
        day = ParseNumber<T>(&pStart, pEnd, 2);
        LOGGING("date: %lld:%lld:%lld\n", year, month, day);
        int64_t result = YearToEpochNano(year, month, day);
        if (result < 0)
            result = 0;
        pOutNanoTime[i] = result;
    }
}

//---------------------------------------
// Look for YYYYMMDD  YYYYY-MM-DD  then HH:MM:SS.mmmuuunnn
// T is either a char or UCS4 (byte string or unicode string)
// Anything invalid is set to 0
template <typename T>
void ParseDateTimeString(int64_t * pOutNanoTime, int64_t arrayLength, const T * pString, int64_t itemSize)
{
    for (int64_t i = 0; i < arrayLength; i++)
    {
        const T * pStart = &pString[i * itemSize];
        const T * pEnd = pStart + itemSize;

        int64_t year = 0;
        int64_t month = 0;
        int64_t day = 0;

        year = ParseNumber<T>(&pStart, pEnd, 4);
        // could check for dash here...
        month = ParseNumber<T>(&pStart, pEnd, 2);
        day = ParseNumber<T>(&pStart, pEnd, 2);
        LOGGING("date: %lld:%lld:%lld\n", year, month, day);
        int64_t yearresult = YearToEpochNano(year, month, day);

        // What if year is negative?
        if (yearresult >= 0)
        {
            int64_t hour = 0;
            int64_t minute = 0;
            int64_t seconds = 0;
            bool bSuccess = 0;

            if (ParseSingleNonNumber(pStart, pEnd))
            {
                hour = ParseNumber<T>(&pStart, pEnd, 2);

                // could check for colon here...
                if (hour < 24 && ParseSingleNonNumber(pStart, pEnd))
                {
                    minute = ParseNumber<T>(&pStart, pEnd, 2);

                    if (minute < 60)
                    {
                        // seconds do not have to exist
                        if (ParseSingleNonNumber(pStart, pEnd))
                        {
                            seconds = ParseNumber<T>(&pStart, pEnd, 2);
                        }

                        bSuccess = 1;
                    }
                }
            }

            if (bSuccess)
            {
                LOGGING("time: %lld:%lld:%lld\n", hour, minute, seconds);
                yearresult += (1000000000LL * ((hour * 3600) + (minute * 60) + seconds));

                // check for milli/micro/etc seconds
                if (pStart < pEnd && *pStart == PERIOD)
                {
                    // skip decimal
                    pStart++;
                    yearresult += ParseDecimal<T>(&pStart, pEnd);
                }
            }
        }
        else
        {
            yearresult = 0;
        }

        pOutNanoTime[i] = yearresult;
    }
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
            ParseTimeString<char>(pNanoTime, arrayLength, pString, itemSize);
        }
        else
        {
            ParseTimeString<uint32_t>(pNanoTime, arrayLength, (uint32_t *)pString, itemSize / 4);
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
            ParseDateString<char>(pNanoTime, arrayLength, pString, itemSize);
        }
        else
        {
            ParseDateString<uint32_t>(pNanoTime, arrayLength, (uint32_t *)pString, itemSize / 4);
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
            ParseDateTimeString<char>(pNanoTime, arrayLength, pString, itemSize);
        }
        else
        {
            ParseDateTimeString<uint32_t>(pNanoTime, arrayLength, (uint32_t *)pString, itemSize / 4);
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
    uint32_t strTimeFormatSize;

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
