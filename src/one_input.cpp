#include "one_input.h"
#include "overloaded.h"

#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"

#include <variant>
#include <utility>
#include <cstddef>
#include <type_traits>

namespace internal
{
    template< typename calculation_t >
    void calculate( char* out_ptr, char const * in_ptr, npy_intp const len, int64_t const stride, operation_t const requested_op, calculation_t const in_type )
    {
        using T = typename calculation_t::data_type;
        [[maybe_unused]]T const value{ *reinterpret_cast< T const *>( in_ptr )};
        [[maybe_unused]]typename calculation_t::calculation_type temp;

        decltype( auto ) res = std::visit( overloaded {
                [&value]( abs_op ) -> T { if constexpr( std::is_unsigned_v< T > ) { return value ; } return ( value < T{} ? -value : value ); },
                [&value]( fabs_op arg ) -> double { return value < T{} ? -value : value ; },
                [&value]( sign_op arg ) -> T { if constexpr( std::is_unsigned_v< T > ) { return value > T{}; } return value > T{} ? 1 : value < T{} ? -1 : T{}; },
                [&value]( floatsign_op arg ) -> T { return value > T{} ? 1.0 : ( value < T{} ? -1.0 : ( value == value ? T{} : value ) ); },
                [&value](neg_op arg) -> T { if constexpr( std::is_unsigned_v< T > ) { return value; } return -value; },
                [&value](bitwise_not_op arg) -> T { return ~value; },
                [&value](not_op arg) -> bool { return !!( value == T{} ); },
                [&value](isnotnan_op arg) -> bool { if constexpr( not std::is_floating_point_v< T > ) return false; return not std::isnan( value ); },
                [](isnan_op arg){},
                [](isfinite_op arg){},
                [](isnotfinite_op arg){},
                [](isinf_op arg){},
                [](isnotinf_op arg){},
                [](isnormal_op arg){},
                [](isnotnormal_op arg){},
                [](isnanorzero_op arg){},
                [](round_op arg){},
                [](floor_op arg){},
                [](trunc_op arg){},
                [](ceil_op arg){},
                [](sqrt_op arg){},
                [](long_op arg){},
                [](log2_op arg){},
                [](log10_op arg){},
                [](exp_op arg){},
                [](exp2_op arg){},
                [](cbrt_op arg){},
                [](tan_op arg){},
                [](cos_op arg){},
                [](sin_op arg){},
                [](signbit_op arg){}
            }, requested_op );
    }
}


PyObject * process_one_input( PyArrayObject const* in_array, PyArrayObject * out_object_1, int32_t function_num, int32_t numpy_intype, int32_t numpy_outtype )
{
    int32_t ndim{};
    int64_t stride{};
    
    int32_t direction{ GetStridesAndContig( in_array, ndim, stride ) };
    npy_intp len{ CALC_ARRAY_LENGTH( ndim, PyArray_DIMS( const_cast< PyArrayObject * >( in_array ) ) ) };
    
    internal::chosen_traits_t ops = internal::set_traits( function_num, numpy_intype );
    
    if ( ops.first && ops.second )
    {
        PyArrayObject * result_array{ ( ndim <= 1 ) ? AllocateNumpyArray( 1, &len, numpy_outtype ) : AllocateLikeNumpyArray( in_array, numpy_outtype ) };
        
         if ( result_array )
        {
            char const * in_ptr = PyArray_BYTES( const_cast< PyArrayObject * >( in_array ) );
            char * out_ptr{ PyArray_BYTES( const_cast< PyArrayObject * >( result_array ) ) };
            
            // Loop around the array here? Or a level deeper?
            calculate( out_ptr, in_ptr, len, stride, *ops.first, *ops.second );
        }
        
        return reinterpret_cast< PyObject* >( result_array );
    }
    
    return nullptr;
}

namespace internal
{
    chosen_traits_t set_traits( int32_t const function_num, int32_t const numpy_intype )
    {
        chosen_traits_t retval{};
        
        switch( numpy_intype )
        {
        case NPY_INT8:
            retval.second = data_type_traits< int8_t >{};
            break;
        case NPY_INT16:
            retval.second = data_type_traits< int16_t >{};
            break;
#if RT_COMPILER_MSVC
        case NPY_INT:
#endif
        case NPY_INT32:
            retval.second = data_type_traits< int32_t >{};
            break;
#if ( RT_COMPILER_CLANG || RT_COMPILER_GCC )
        case NPY_LONGLONG:
#endif
        case NPY_INT64:
            retval.second = data_type_traits< int64_t >{};
            break;
        case NPY_UINT8:
            retval.second = data_type_traits< uint16_t >{};
            break;
        case NPY_UINT16:
            retval.second = data_type_traits< uint16_t >{};
            break;
#if RT_COMPILER_MSVC
        case NPY_UINT:
#endif
        case NPY_UINT32:
            retval.second = data_type_traits< uint32_t >{};
            break;
#if ( RT_COMPILER_CLANG || RT_COMPILER_GCC )
        case NPY_ULONGLONG:
#endif
        case NPY_UINT64:
            retval.second = data_type_traits< uint64_t >{};
            break;
        case NPY_FLOAT:
            retval.second = data_type_traits< float >{};
            break;
        case NPY_DOUBLE:
            retval.second = data_type_traits< double >{};
            break;
        }
        
        switch( function_num )
        {
        case MATH_OPERATION::ABS:
            retval.first = abs_op{};
            break;
        case MATH_OPERATION::ISNAN:
            retval.first = isnan_op{};
            break;
            
        case MATH_OPERATION::ISNOTNAN:
            retval.first = isnotnan_op{};
            break;
            
        case MATH_OPERATION::ISFINITE:
            retval.first = isfinite_op{};
            break;
            
        case MATH_OPERATION::ISNOTFINITE:
            retval.first = isnotfinite_op{};
            break;
            
        case MATH_OPERATION::NEG:
            retval.first = bitwise_not_op{};
            break;
            
        case MATH_OPERATION::INVERT:
            retval.first = bitwise_not_op{};
            break;
            
        case MATH_OPERATION::FLOOR:
            retval.first = floor_op{};
            break;
            
        case MATH_OPERATION::CEIL:
            retval.first = ceil_op{};
            break;
            
        case MATH_OPERATION::TRUNC:
            retval.first = trunc_op{};
            break;
            
        case MATH_OPERATION::ROUND:
            retval.first = round_op{};
            break;
            
        case MATH_OPERATION::SQRT:
            retval.first = sqrt_op{};
            break;
        }
        
        return retval;
    }
}
