#include "../../src/UnaryOps.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/ut.hpp"

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    static constexpr std::array< float, 33 > input_data_simple = {-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2, 2.5, 3,
        3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11};

    suite unary = []
    {
        
        size_t len{ sizeof(input_data_simple) };
        PyArrayObject* p_result{ AllocateNumpyArray(1, (npy_intp*)&len, NPY_FLOAT )};
        memcpy(PyArray_DATA(p_result), input_data_simple.data(), sizeof(input_data_simple));
        "BinaryOpFastStrided"_test = [&]
        {
        };
    };
}