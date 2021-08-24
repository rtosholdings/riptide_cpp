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
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, abs_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( std::is_unsigned_v< T > == true )
                     {
                        return T{value};
                     }
         else
         {
            return value < T{} ? T(-value) : T(value);
         }
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, fabs_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return T{value};
                     }
         else
         {
            return value < T{} ? T(-value) : T(value);
         }
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, sign_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( std::is_unsigned_v< T > == true )
                     {
                        return T(value) > T{} ? T(1) : T{};
                     }
         else
         {
            return value > T{} ? T(1) : T(value) < T{} ? T(-1) : T{};
         }
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, floatsign_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return T{};
                     }
         else
         {
            return value > T{} ? T(1.0) : ( value < T{} ? T(-1.0) : ( value == value ? T{} : T(value) ) );
         }
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, neg_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( std::is_unsigned_v< T > == true )
                     {
                        return T(value);
                     }
         else
         {
            return T(-value);
         }
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, bitwise_not_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( std::is_floating_point_v< T > == true )
                     {
                        return T(NAN);
                     }
         else
         {
            return T(~value);
         }
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, round_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T(round( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, floor_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
      if ( in_type )
      {
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( floor( value ) );
      }
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, trunc_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( trunc( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, ceil_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( ceil( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, sqrt_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( sqrt( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, log_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( log( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, log2_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( log2( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, log10_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( log10( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, exp_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( exp( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, exp2_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( exp2( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, cbrt_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( cbrt( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, tan_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( tan( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, cos_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( cos( value ) );
   }

   template< typename calculation_t >
   decltype( auto ) calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, sin_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return T( sin( value ) );
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, signbit_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr ( not std::is_floating_point_v< T > == true )
                      {
                         return std::is_signed_v< T > && T( value ) < T{};
                      }
         else
         {
            return std::signbit( T(value) );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, not_op const * requested_op, calculation_t const * in_type )
   {
         using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         return !!( T(value) == T{} );
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isnotnan_op const * requested_op, calculation_t const * in_type )
   {
         using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return false;
                     }
         else
         {
            return not std::isnan( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isnan_op const * requested_op, calculation_t const * in_type )
   {
         using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return false;
                     }
         else
         {
            return std::isnan( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isfinite_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return false;
                     }
         else
         {
            return std::isfinite( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isnotfinite_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return false;
                     }
         else
         {
            return not std::isfinite( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isinf_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return false;
                     }
         else
         {
            return std::isinf( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isnotinf_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return false;
                     }
         else
         {
            return not std::isinf( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isnormal_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return false;
                     }
         else
         {
            return std::isnormal( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isnotnormal_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return value;
                     }
         else
         {
            return not std::isnormal( value );
         }
   }

   template< typename calculation_t >
   bool calculate( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, isnanorzero_op const * requested_op, calculation_t const * in_type )
   {
      using T = typename calculation_t::data_type const;
         [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
         [[maybe_unused]] typename calculation_t::calculation_type temp;

         if constexpr( not std::is_floating_point_v< T > == true )
                     {
                        return T(value) == T{};
                     }
         else
         {
            return T(value) == T{} || std::isnan( value );
         }
   }

   // numpy standard is to treat stride as bytes, but I'm keeping the math simple for now more for exposition than anything else.
   template< typename operation_t, typename data_t >
   void perform_operation( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, operation_t * op_p, data_t * data_type_p, int64_t const out_stride_as_items = 1 )
   {
      if ( op_p )
      {
         // Output cannot be longer than the input
         char const * last_out_p{ out_p + sizeof( data_t ) * len };

         while( out_p < last_out_p )
         {
            calculate( out_p, in_p, len, stride, op_p, data_type_p );

            in_p += stride;
            out_p += sizeof( data_t ) * out_stride_as_items; 
         }
      }
   }

   template< typename operation_variant, typename data_type, size_t... Is >
   void calculate_for_active_operation( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, operation_variant const & requested_op, data_type const * type_p, std::index_sequence< Is... > )
   {
      if ( type_p )
      {
         ( perform_operation( out_p, in_p, len, stride, std::get_if< Is >( &requested_op ), type_p ), ... );
      }
   }
   
   template< typename type_variant, size_t... Is >
   void calculate_for_active_data_type( char * out_p, char const * in_p, npy_intp const len, int64_t const stride, operation_t const & requested_op, type_variant const & in_type, std::index_sequence< Is... > )
   {
      ( calculate_for_active_operation( out_p, in_p, len, stride, requested_op, std::get_if< Is >( &in_type ), std::make_index_sequence< std::variant_size_v< operation_t > >{} ), ... );
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
            char const * in_p = PyArray_BYTES( const_cast< PyArrayObject * >( in_array ) );
            char * out_p{ PyArray_BYTES( const_cast< PyArrayObject * >( result_array ) ) };
            
            internal::calculate_for_active_data_type( out_p, in_p, len, stride, *ops.first, *ops.second, std::make_index_sequence< std::variant_size_v< internal::data_type_t > >{} );
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
