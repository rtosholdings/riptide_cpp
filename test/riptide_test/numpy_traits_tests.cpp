#include "CommonInc.h"
#include "RipTide.h"
#include "numpy_traits.h"

#include "ut_extensions.h"

#include "boost/ut.hpp"

#include <cmath>
#include <type_traits>

using namespace riptide;
using namespace boost::ut;
using boost::ut::suite;

namespace
{
    template <NPY_TYPES TypeCodeIn>
    struct typecode_to_type
    {
        static constexpr NPY_TYPES value{ TypeCodeIn };
    };

    template <typename T1, typename T2>
    inline constexpr bool is_compatible_v = std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2> &&
                                            std::numeric_limits<T1>::is_integer == std::numeric_limits<T2>::is_integer &&
                                            std::numeric_limits<T1>::is_signed == std::numeric_limits<T2>::is_signed &&
                                            std::numeric_limits<T1>::digits == std::numeric_limits<T2>::digits;

    using SupportedTypeCodeTypes =
        std::tuple<typecode_to_type<NPY_TYPES::NPY_BOOL>, typecode_to_type<NPY_TYPES::NPY_INT8>,
                   typecode_to_type<NPY_TYPES::NPY_INT16>, typecode_to_type<NPY_TYPES::NPY_INT32>,
                   typecode_to_type<NPY_TYPES::NPY_INT64>, typecode_to_type<NPY_TYPES::NPY_UINT8>,
                   typecode_to_type<NPY_TYPES::NPY_UINT16>, typecode_to_type<NPY_TYPES::NPY_UINT32>,
                   typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
                   typecode_to_type<NPY_TYPES::NPY_DOUBLE>, typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>>;

    struct numpy_traits_tester
    {
        template <typename TypeCodeType>
        void operator()()
        {
            constexpr NPY_TYPES typecode_{ TypeCodeType::value };

            using c_type = riptide::numpy_c_type_t<typecode_>;
            using cpp_type = riptide::numpy_cpp_type_t<typecode_>;

            static_assert(is_compatible_v<c_type, cpp_type>);

            constexpr NPY_TYPES remapped_typecode_
            {
#if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
                typecode_ == NPY_LONGDOUBLE ? NPY_DOUBLE :
#endif
                                              typecode_
            };

            static_assert(riptide::numpy_c_type_code_v<c_type> == remapped_typecode_);
            static_assert(riptide::numpy_type_code_v<cpp_type> == remapped_typecode_);
        }
    };

    suite invalids_compatibility = []
    {
        "numpy_traits"_test = numpy_traits_tester{} | SupportedTypeCodeTypes{};
    };
}
