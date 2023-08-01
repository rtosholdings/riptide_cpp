#include "CommonInc.h"
#include "RipTide.h"
#include "numpy_traits.h"

#include "np_util.h"
#include "ut_extensions.h"

#include "boost/ut.hpp"

#include <cmath>
#include <type_traits>

using namespace riptide_utility::internal;
using namespace riptide;
using namespace boost::ut;
using boost::ut::suite;

namespace
{
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

            static_assert(numpy_is_storable_v<cpp_type, c_type>);

            // Fixed C++ types must map to expected fixed typecode (respecting long double ambiguity)
            constexpr NPY_TYPES remapped_typecode_ =
#if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
                typecode_ == NPY_LONGDOUBLE ? NPY_DOUBLE :
#endif
                                              typecode_;

            static_assert(riptide::numpy_type_code_v<cpp_type> == remapped_typecode_);

            // Storage C types must map to storage typecodes that map to same storage C types.
            // NOTE: fixed C++ typecodes may map to larger C storage types (e.g. bool -> uint8).
            constexpr NPY_TYPES remapped_storage_typecode_ = riptide::numpy_c_type_code_v<c_type>;
            using remapped_storage_type = riptide::numpy_c_type_t<remapped_storage_typecode_>;
            static_assert(std::is_same_v<remapped_storage_type, c_type>);
        }
    };

    suite invalids_compatibility = []
    {
        "numpy_traits"_test = numpy_traits_tester{} | SupportedTypeCodeTypes{};
    };
}
