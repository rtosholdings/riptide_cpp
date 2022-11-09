#include "riptide_python_test.h"
#include "platform_detect.h"

#include "boost/ut.hpp"

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <exception>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    std::array<int32_t, 1024ULL * 1024ULL> output;
    std::array<int8_t, 1024ULL * 1024ULL> bools;

    suite hash_linear_ops = []
    {
#ifndef RT_OS_WINDOWS
        skip /
#endif
            "ismember32_int32"_test = [&]
        {
            std::vector<int32_t> haystack(128ULL * 1024ULL * 1024ULL);
            std::vector<int32_t> needles(1024ULL * 1024ULL);
            std::random_device dev{};
            std::mt19937 engine(dev());
            std::uniform_int_distribution<uint64_t> dist(1, haystack.size());

            std::iota(std::begin(haystack), std::end(haystack), 1);
            std::generate(std::begin(needles), std::end(needles),
                          [&]
                          {
                              return dist(engine);
                          });

            npy_intp dim_len{ static_cast<npy_intp>(haystack.size()) };
            PyObject * Py_haystack{ PyArray_SimpleNewFromData(1, &dim_len, NPY_INT32, haystack.data()) };
            Py_INCREF(Py_haystack);

            dim_len = needles.size();
            PyObject * Py_needles{ PyArray_SimpleNewFromData(1, &dim_len, NPY_INT32, needles.data()) };
            Py_INCREF(Py_needles);

            PyObject * function_object = riptide_python_test::internal::get_named_function(riptide_module_p, "IsMember32");
            Py_INCREF(function_object);
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, Py_needles, Py_haystack);

            PyArrayObject * bools{};
            PyArrayObject * indices{};
            expect(PyArg_ParseTuple(retval, "O!O!", &PyArray_Type, &bools, &PyArray_Type, &indices));

            char * bool_bytes{ PyArray_BYTES(bools) };
            char * index_bytes{ PyArray_BYTES(indices) };

            int32_t * index_vals{ reinterpret_cast<int32_t *>(index_bytes) };

            for (int i{ 0 }; i != 200; ++i)
            {
                if (bool_bytes[i])
                {
                    expect(haystack[index_vals[i]] == needles[i]);
                }
            }

            for (size_t i{ needles.size() }; i != needles.size() - 200; --i)
            {
                if (bool_bytes[i])
                {
                    expect(haystack[index_vals[i]] == needles[i]);
                }
            }

            Py_DECREF(Py_haystack);
            Py_DECREF(Py_needles);
            Py_DECREF(function_object);
        };
    };
}
