#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API
#include "Riptide.h"
#include "Convert.h"
#include "platform_detect.h"

#include "numpy/arrayobject.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <type_traits>
#include <cfloat>

using namespace boost::ut;
using boost::ut::suite;

extern "C"
PyObject* CompareNumpyMemAddress(PyObject *self, PyObject *args);

namespace
{
    suite convert = []
    {
        "check_python_type_names"_test = []
        {
            [[maybe_unused]] PyObject * temp{ nullptr };
            [[maybe_unused]] PyArrayObject * temp2{ nullptr };
            std::cout << "In the tests\n";
        };

        "numpy_linkage"_test = []
        {
            int64_t shape{10};
            npy_intp const shape_p = reinterpret_cast< npy_intp >(&shape);
            PyObject * null_array{ PyArray_ZEROS( 1, &shape_p, NPY_FLOAT, 0 ) };
            expect( null_array != nullptr ) << "We should get a PyArray of some kind back, not a nullptr";
        };
    };
}
