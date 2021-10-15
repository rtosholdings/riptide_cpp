#include "one_input.h"
#include "overloaded.h"

#include "RipTide.h"
#include "ndarray.h"
#include "MathWorker.h"
#include "basic_ops.h"

#include "simd/avx2.h"

#include <variant>
#include <utility>
#include <cstddef>
#include <type_traits>

namespace internal
{
   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, abs_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const * wide_value_p( reinterpret_cast< wide_t const * >( in_p ) );

      if constexpr( std::is_unsigned_v< T > == true )
                  {
                     return T{value};
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.abs( wide_value_p );
                     }
         else
         {
            return T( std::abs( value ) );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, fabs_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return T{value};
                  }
      else
      {
         return value < T{} ? T(-value) : T(value);
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, sign_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( std::is_unsigned_v< T > == true )
                  {
                     return T(value) > T{} ? T(1) : T{};
                  }
      else
      {
         return value > T{} ? T(1) : T(value) < T{} ? T(-1) : T{};
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, floatsign_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return T{};
                  }
      else
      {
         return value > T{} ? T(1.0) : ( value < T{} ? T(-1.0) : value == value ? T{} : T(value) );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, neg_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( std::is_unsigned_v< T > == true )
                  {
                     return T(value);
                  }
      else
      {
         return T(-value);
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, bitwise_not_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( std::is_floating_point_v< T > == true )
                  {
                     return T(NAN);
                  }
      else
      {
         return T(~value);
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, round_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const wide_value( internal::LOADU( reinterpret_cast< wide_t const * >( in_p ) ) );

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return T( std::round( value ) );
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.round( wide_value );
                     }
         else
         {
            return T( std::round( value ) );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, floor_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const wide_value( internal::LOADU( reinterpret_cast< wide_t const * >( in_p ) ) );

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return T( std::floor( value ) );
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.floor( wide_value );
                     }
         else
         {
            return T( std::floor( value ) );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, trunc_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const wide_value( internal::LOADU( reinterpret_cast< wide_t const * >( in_p ) ) );

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return T( std::trunc( value ) );
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.trunc( wide_value );
                     }
         else
         {
            return T( std::trunc( value ) );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, ceil_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const wide_value( internal::LOADU( reinterpret_cast< wide_t const * >( in_p ) ) );

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return T( std::ceil( value ) );
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.ceil( wide_value );
                     }
         else
         {
            return T( std::ceil( value ) );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, sqrt_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const wide_value( internal::LOADU( reinterpret_cast< wide_t const * >( in_p ) ) );

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return T( std::sqrt( value ) );
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.sqrt( wide_value );
                     }
         else
         {
            return T( std::sqrt( value ) );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, log_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( log( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, log2_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( log2( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, log10_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( log10( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, exp_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( exp( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, exp2_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( exp2( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, cbrt_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( cbrt( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, tan_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( tan( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, cos_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( cos( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, sin_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return T( sin( value ) );
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, signbit_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr ( not std::is_floating_point_v< T > == true )
                   {
                      return std::is_signed_v< T > && T( value ) < T{};
                   }
      else
      {
         return std::signbit( T(value) );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, not_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      return !!( T(value) == T{} );
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, isnotnan_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const wide_value( internal::LOADU( reinterpret_cast< wide_t const * >( in_p ) ) );

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return false;
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.isnotnan( wide_value );
                     }
         else
         {
            return not std::isnan( value );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   decltype( auto ) calculate( char const * in_p, isnan_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };
      using wide_t = typename calculation_t::calculation_type;
      [[maybe_unused]] wide_t const wide_value( internal::LOADU( reinterpret_cast< wide_t const * >( in_p ) ) );

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return false;
                  }
      else
      {
         if constexpr( wide_ops.simd_implemented_v )
                     {
                        return wide_ops.isnan( wide_value );
                     }
         else
         {
            return std::isnan( value );
         }
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, isfinite_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return false;
                  }
      else
      {
         return std::isfinite( value );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, isnotfinite_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return false;
                  }
      else
      {
         return not std::isfinite( value );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, isinf_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return false;
                  }
      else
      {
         return std::isinf( value );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, isnotinf_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return false;
                  }
      else
      {
         return not std::isinf( value );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, isnormal_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return false;
                  }
      else
      {
         return std::isnormal( value );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, isnotnormal_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

      if constexpr( not std::is_floating_point_v< T > == true )
                  {
                     return value;
                  }
      else
      {
         return not std::isnormal( value );
      }
   }

   template< typename calculation_t, typename wide_ops_t >
   bool calculate( char const * in_p, isnanorzero_op const * requested_op, calculation_t const * in_type, wide_ops_t wide_ops )
   {
      using T = typename calculation_t::data_type const;
      [[maybe_unused]] T const value{ *reinterpret_cast< T const * >( in_p ) };

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
   void perform_operation( char const * in_p, char * out_p, npy_intp const len, int64_t const stride, operation_t * op_p, data_t * data_type_p, int64_t const out_stride_as_items = 1 )
   {
      // Output cannot be longer than the input
      char const * last_out_p{ out_p + sizeof( data_t ) * len };

      auto calc = [&](auto vectorization_object)
                  {
                     while( out_p < last_out_p )
                     {
                        auto x = calculate( in_p, op_p, data_type_p, vectorization_object );
                        *reinterpret_cast< decltype( x ) * >( out_p ) = x;

                        in_p += stride;
                        out_p += sizeof( decltype( x ) ) * out_stride_as_items;
                     }
                  };

      if ( op_p )
      {
         if constexpr( operation_t::simd_implementation::value )
                     {
                        using wide_type = typename data_t::calculation_type;
                        using wide_sct = typename riptide::simd::avx2::template vec256< typename data_t::data_type >;

                        if ( stride == sizeof( typename data_t::data_type ) && out_stride_as_items == 1 )
                        {
                           calc( wide_sct{} );
                        }
                        else
                        {
                           calc(typename riptide::simd::avx2::template vec256< void >{});
                        }
                     }
         else
         {
            calc( typename riptide::simd::avx2::template vec256< void >{} );
         }
      }
   }

   template< typename operation_variant, typename data_type, size_t... Is >
   void calculate_for_active_operation( char const * in_p, char * out_p, npy_intp const len, int64_t const stride, operation_variant const & requested_op, data_type const * type_p, std::index_sequence< Is... > )
   {
      if ( type_p )
      {
         ( perform_operation( in_p, out_p, len, stride, std::get_if< Is >( &requested_op ), type_p ), ... );
      }
   }

   template< typename type_variant, size_t... Is >
   void calculate_for_active_data_type( char const * in_p, char * out_p, npy_intp const len, int64_t const stride, operation_t const & requested_op, type_variant const & in_type, std::index_sequence< Is... > )
   {
      ( calculate_for_active_operation( in_p, out_p, len, stride, requested_op, std::get_if< Is >( &in_type ), std::make_index_sequence< std::variant_size_v< operation_t > >{} ), ... );
   }

   template< typename operation_variant, size_t... Is >
   bool get_active_value_return( operation_variant v, std::index_sequence< Is... > )
   {
      return ( ( std::get_if< Is >( &v ) ? std::get_if< Is >( &v )->value_return : false ) || ... );
   }

   template< typename operation_trait, typename type_trait >
   void walk_row_major( char const * in_p, char * out_p, int32_t ndim, PyArrayObject const * in_array, int64_t const stride_out, operation_trait const & requested_op, type_trait const & in_type )
   {
      ptrdiff_t inner_len{1};
      for( int32_t i{1}; i < ndim; ++i ) // Is this loop really right? One-based but bounded by < ndim???
      {
         inner_len *= PyArray_DIM( in_array, i );
      }

      ptrdiff_t const outer_len = PyArray_DIM( in_array, 0 );
      ptrdiff_t const outer_stride = PyArray_STRIDE( in_array, 0 );

      for ( ptrdiff_t offset{};  offset < outer_len; ++offset )
      {
         calculate_for_active_data_type( in_p + ( offset * outer_stride ), out_p + ( offset * inner_len * stride_out ), outer_len, outer_stride, requested_op, in_type, std::make_index_sequence< std::variant_size_v< internal::data_type_t > >{} );
      }
   }

   template< typename operation_trait, typename type_trait >
   void walk_column_major( char const * in_p, char * out_p, int32_t ndim, PyArrayObject const * in_array, int64_t const stride_out, operation_trait const & requested_op, type_trait const & in_type )
   {
      ptrdiff_t inner_len{ PyArray_DIM( in_array, 0 ) * PyArray_DIM( in_array, 1 ) };

/*   This loop from UnaryOps.cpp is optimized to the above,
     I'm unsure about it since it only looks at 2 dimensions,
     and then we utilize at ndim below instead.
     for( int32_t i{0}; i < 1; ++i )
     {
     inner_len *= PyArray_DIM( in_array, i );
     }
*/

      ptrdiff_t const outer_len{ PyArray_DIM( in_array, ( ndim - 1 ) ) };
      ptrdiff_t const outer_stride{ PyArray_DIM( in_array, ( ndim - 1 ) ) };

      for( ptrdiff_t offset{}; offset < outer_len; ++offset )
      {
         calculate_for_active_data_type( in_p + ( offset * outer_stride ), out_p + ( offset * inner_len * stride_out ), outer_len, outer_stride, requested_op, in_type, std::make_index_sequence< std::variant_size_v< internal::data_type_t > >{} );
      }
   }
}

PyObject * process_one_input( PyArrayObject const* in_array, PyArrayObject * out_object_1, int32_t function_num, int32_t numpy_intype, int32_t numpy_outtype )
{
   int32_t ndim{};
   int64_t stride{};

   int32_t direction{ GetStridesAndContig( in_array, ndim, stride ) };
   npy_intp len{ CALC_ARRAY_LENGTH( ndim, PyArray_DIMS( const_cast< PyArrayObject * >( in_array ) ) ) };

   auto [ opt_op_trait, opt_type_trait ] = internal::set_traits( function_num, numpy_intype );

   if ( opt_op_trait && opt_type_trait )
   {
      if ( direction == 0 && numpy_outtype == -1 )
      {
         numpy_outtype = get_active_value_return( *opt_op_trait, std::make_index_sequence< std::variant_size_v< internal::operation_t > >{} ) ? numpy_intype : NPY_BOOL;
         PyArrayObject * result_array{ ( ndim <= 1 ) ? AllocateNumpyArray( 1, &len, numpy_outtype ) : AllocateLikeNumpyArray( in_array, numpy_outtype ) };

         if ( result_array )
         {
            char const * in_p = PyArray_BYTES( const_cast< PyArrayObject * >( in_array ) );
            char * out_p{ PyArray_BYTES( const_cast< PyArrayObject * >( result_array ) ) };

            internal::calculate_for_active_data_type( in_p, out_p, len, stride, *opt_op_trait, *opt_type_trait, std::make_index_sequence< std::variant_size_v< internal::data_type_t > >{} );
         }
         else
         {
            Py_INCREF( Py_None );
            return Py_None;
         }

         return reinterpret_cast< PyObject* >( result_array );
      }
      else
      {
         int wanted_outtype = get_active_value_return( *opt_op_trait, std::make_index_sequence< std::variant_size_v< internal::operation_t > >{} ) ? numpy_intype : NPY_BOOL;

         if ( numpy_outtype != -1 && numpy_outtype != wanted_outtype )
         {
            LOGGING( "Wanted output type %d does not match output type %d\n", wanted_outtype, numpy_outtype );
            Py_INCREF( Py_None );
            return Py_None;
         }

         PyArrayObject * result_array { numpy_outtype == -1 ? AllocateLikeNumpyArray( in_array, wanted_outtype ) : out_object_1 };

         if ( ( result_array == nullptr ) ||
              ( result_array == out_object_1 ) && ( len != ArrayLength( result_array ) )
            )
         {
            Py_INCREF( Py_None );
            return Py_None;
         }

         if ( result_array == out_object_1 )
         {
            Py_INCREF( result_array );
         }

         char const *in_p{ PyArray_BYTES( const_cast< PyArrayObject * >( in_array ) ) };
         char * out_p{ PyArray_BYTES( const_cast< PyArrayObject * >( result_array ) ) };

         int num_dims_out{};
         int64_t stride_out{};
         int direction_out = GetStridesAndContig( result_array, num_dims_out, stride_out );

         if ( direction_out == 0 )
         {
            switch( direction )
            {
            case 0:
               internal::calculate_for_active_data_type( in_p, out_p, len, stride, *opt_op_trait, *opt_type_trait, std::make_index_sequence< std::variant_size_v< internal::data_type_t > >{} );
               break;
            case 1:
               internal::walk_row_major( in_p, out_p, ndim, in_array, stride_out, *opt_op_trait, *opt_type_trait );
               break;
            case -1:
               internal::walk_column_major( in_p, out_p, ndim, in_array, stride_out, *opt_op_trait, *opt_type_trait );
               break;
            }
         }
      }
      Py_INCREF( Py_None );
      return Py_None;
   }

   Py_INCREF( Py_None );
   return Py_None;
}

namespace internal
{
   chosen_traits_t set_traits( int32_t const function_num, int32_t const numpy_intype )
   {
      chosen_traits_t retval{};

      switch( numpy_intype )
      {
      case NPY_INT8:
         retval.second = int8_traits{};
         break;
      case NPY_INT16:
         retval.second = int16_traits{};
         break;
#if RT_COMPILER_MSVC
      case NPY_INT:
#endif
      case NPY_INT32:
         retval.second = int32_traits{};
         break;
#if ( RT_COMPILER_CLANG || RT_COMPILER_GCC )
      case NPY_LONGLONG:
#endif
      case NPY_INT64:
         retval.second = int64_traits{};
         break;
      case NPY_UINT8:
         retval.second = uint8_traits{};
         break;
      case NPY_UINT16:
         retval.second = uint16_traits{};
         break;
#if RT_COMPILER_MSVC
      case NPY_UINT:
#endif
      case NPY_UINT32:
         retval.second = uint32_traits{};
         break;
#if ( RT_COMPILER_CLANG || RT_COMPILER_GCC )
      case NPY_ULONGLONG:
#endif
      case NPY_UINT64:
         retval.second = uint64_traits{};
         break;
      case NPY_FLOAT:
         retval.second = float_traits{};
         break;
      case NPY_DOUBLE:
         retval.second = double_traits{};
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
