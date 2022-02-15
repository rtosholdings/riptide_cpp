#include "RipTide.h"
#include "UnaryOps.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <type_traits>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    std::array<float const, 31> const input_data_simple_f = { -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,   -0.5, 0,   0.5, 1.0,
                                                              1.5, 2,    2.5, 3,    3.5, 4,    4.5,  5,    5.5, 6,   6.5,
                                                              7,   7.5,  8,   8.5,  9,   9.5,  10.5, 10.5, 11 };

    std::array<float const, 31> const input_data_nan_f = { -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,   -0.5, 0,   NAN, 1.0,
                                                           1.5, 2,    2.5, 3,    3.5, 4,    4.5,  5,    5.5, 6,   6.5,
                                                           7,   7.5,  8,   8.5,  9,   9.5,  10.5, 10.5, 11 };

    std::array<float const, 31> const input_data_inf_f = { -4,  -3.5, -3,  -2.5, -2,  -1.5, -1,   -0.5, 0,   INFINITY, 1.0,
                                                           1.5, 2,    2.5, 3,    3.5, 4,    4.5,  5,    5.5, 6,        6.5,
                                                           7,   7.5,  8,   8.5,  9,   9.5,  10.5, 10.5, 11 };

    std::array<float const, 31> const input_data_normal_f = { -4,       -3.5,          -3,   -2.5, -2,  -1.5, -1,  -0.5, 0,
                                                              INFINITY, FLT_MIN / 2.0, NAN,  2,    2.5, 3,    3.5, 4,    4.5,
                                                              5,        5.5,           6,    6.5,  7,   7.5,  8,   8.5,  9,
                                                              9.5,      10.5,          10.5, 11 };

    std::array<int32_t const, 31> const input_data_simple_i = { -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  6, 7,
                                                                8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };

    float const * p_float = input_data_simple_f.data();
    float const * p_nans = input_data_nan_f.data();
    float const * p_inf = input_data_inf_f.data();
    float const * p_norm = input_data_normal_f.data();
    int32_t const * p_int32 = input_data_simple_i.data();

    suite unary_ops = []
    {
        "walk_abs_float"_test = [&]
        {
            expect(true);

            npy_intp array_len{input_data_simple_f.size()};
            PyObject * input_array{PyArray_SimpleNewFromData( 1, &array_len, NPY_FLOAT, const_cast< float * >(input_data_simple_f.data() ))};
            PyObject * input_arg{ PyTuple_New(1)};
            expect( PyTuple_SetItem( input_arg, 0, input_array ));
//            expect( BasicMathOneInput( nullptr, input_arg ) != nullptr );


#if 0
            operation_t op{ abs_op{} };
            data_type_t data_type{ float_traits{} };
            std::array<float, 28> x{};
            walk_data_array(1, 28, 4, 4, reinterpret_cast<char const *>(p_float + 5), reinterpret_cast<char *>(x.data()), op,
                            data_type);
            expect(x[0] == 1.5_f);
            expect(x[1] == 1.0_f);
            expect(x[2] == 0.5_f);
            expect(x[3] == 0.0_f);
            expect(x[4] == 0.5_f);
            expect(x[5] == 1.0_f);
            expect(x[6] == 1.5_f);
            expect(x[7] == 2.0_f);
#endif       
        };
    };
}
