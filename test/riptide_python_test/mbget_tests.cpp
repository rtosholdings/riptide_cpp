#include "riptide_python_test.h"

#include "numpy_traits.h"
#include "Merge.h"

#include "buffer.h"
#include "mem_buffer.h"
#include "np_util.h"
#include "tuple_util.h"
#include "ut_extensions.h"

#include "ut_core.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <variant>

using namespace riptide_python_test::internal;
using namespace riptide_utility::internal;
using namespace boost::ut;
using riptide_utility::ut::file_suite;

namespace
{
    using SupportedTypeCodeInNumTypes =
        std::tuple<typecode_to_type<NPY_TYPES::NPY_BOOL>, typecode_to_type<NPY_TYPES::NPY_INT8>,
                   typecode_to_type<NPY_TYPES::NPY_INT16>, typecode_to_type<NPY_TYPES::NPY_INT32>,
                   typecode_to_type<NPY_TYPES::NPY_INT64>, typecode_to_type<NPY_TYPES::NPY_UINT8>,
                   typecode_to_type<NPY_TYPES::NPY_UINT16>, typecode_to_type<NPY_TYPES::NPY_UINT32>,
                   typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
                   typecode_to_type<NPY_TYPES::NPY_DOUBLE>, typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>>;

    using SupportedTypeCodeIndexNumTypes =
        std::tuple<typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>,
                   typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>,
                   typecode_to_type<NPY_TYPES::NPY_UINT64>>;

    using SupportedParams = decltype(tuple_prod(SupportedTypeCodeInNumTypes{}, SupportedTypeCodeIndexNumTypes{}));

    template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
    struct mbget_tester
    {
        static constexpr auto typecode_out = TypeCodeIn;

        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;
        using cpp_type_index = riptide::numpy_cpp_type_t<TypeCodeIndex>;
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec(const_buffer<cpp_type_in> const in_values, const_buffer<cpp_type_index> const index_values,
                         const_buffer<cpp_type_out> const expected, std::optional<cpp_type_out> const opt_default_value,
                         reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeIn>, typecode_to_type<TypeCodeIndex>>;
            auto const desc{ [&opt_default_value]()
                             {
                                 std::ostringstream str;
                                 str << "default_value: ";
                                 if (opt_default_value.has_value())
                                 {
                                     str << to_out(opt_default_value.value());
                                 }
                                 else
                                 {
                                     str << "<N/A>";
                                 }
                                 return str.str();
                             }() };

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const in_array{ pyarray_from_array<TypeCodeIn>(in_values) };
            typed_expect<desc_type>(desc, (in_array != nullptr) >> fatal) << caller_loc;
            auto const index_array{ pyarray_from_array<TypeCodeIndex>(index_values) };
            typed_expect<desc_type>(desc, (index_array != nullptr) >> fatal) << caller_loc;

            pyobject_ptr retval{ opt_default_value.has_value() ?
                                     PyObject_CallMethod(riptide_module_p, "MBGet", "OOi", in_array.get(), index_array.get(),
                                                         opt_default_value.value()) :
                                     PyObject_CallMethod(riptide_module_p, "MBGet", "OO", in_array.get(), index_array.get()) };
            typed_expect<desc_type>(desc, no_pyerr() >> fatal) << caller_loc;

            auto const actual_values{ cast_pyarray_values_as<typecode_out>(&retval) };
            typed_expect<desc_type>(desc, no_pyerr() >> fatal) << caller_loc;

            for (size_t i{ 0 }; i < expected.size(); ++i)
            {
                auto const expected_value{ expected[i] };
                auto const actual_value{ actual_values[i] };
                typed_expect<desc_type>(desc, equal_to_nan_aware(actual_value, expected_value))
                    << "index:" << i << ", expected:" << to_out(expected_value) << ", actual:" << to_out(actual_value)
                    << caller_loc;
            }
        }
    };

    enum class test_case_id
    {
        VALID,
        OOB,
        INVALID,
        INVALID_DEFAULT,
    };

    struct mbget_tests
    {
        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
        using tester_type = mbget_tester<TypeCodeIn, TypeCodeIndex>;

        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
        struct test_case
        {
            using this_tester_type = tester_type<TypeCodeIn, TypeCodeIndex>;

            using cpp_type_in = typename this_tester_type::cpp_type_in;
            using cpp_type_index = typename this_tester_type::cpp_type_index;
            using cpp_type_out = typename this_tester_type::cpp_type_out;

            any_const_buffer<cpp_type_in> in_values_;
            any_const_buffer<cpp_type_index> index_values_;
            any_const_buffer<cpp_type_out> expected_values_;
            std::optional<cpp_type_out> opt_default_value_;
            reflection::source_location loc_;

            template <template <typename> typename BufferInT, template <typename> typename BufferOutT>
            test_case(BufferInT<cpp_type_in> && in_values, BufferInT<cpp_type_index> && index_values,
                      BufferOutT<cpp_type_out> && expected_values,
                      std::optional<cpp_type_out> const & opt_default_value = std::optional<cpp_type_out>(),
                      reflection::source_location const & loc = reflection::source_location::current())
                : in_values_{ std::move(in_values) }
                , index_values_{ std::move(index_values) }
                , expected_values_{ std::move(expected_values) }
                , opt_default_value_{ opt_default_value }
                , loc_{ loc }
            {
            }
        };

        template <test_case_id Id>
        struct test
        {
            template <typename T>
            void operator()() const
            {
                using TypeCodeInType = std::tuple_element_t<0, T>;
                using TypeCodeIndexType = std::tuple_element_t<1, T>;

                constexpr auto TypeCodeIn = TypeCodeInType::value;
                constexpr auto TypeCodeIndex = TypeCodeIndexType::value;

                using this_tester_type = tester_type<TypeCodeIn, TypeCodeIndex>;

                auto const testcase = get_test_case<Id, TypeCodeIn, TypeCodeIndex>{}();

                this_tester_type::exec(testcase.in_values_, testcase.index_values_, testcase.expected_values_,
                                       testcase.opt_default_value_, testcase.loc_);
            }
        };

        template <test_case_id Id, NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
        struct get_test_case
        {
            auto operator()()
            {
                using test_case_type = test_case<TypeCodeIn, TypeCodeIndex>;
                using cpp_type_in = typename test_case_type::cpp_type_in;
                using cpp_type_index = typename test_case_type::cpp_type_index;
                using cpp_type_out = typename test_case_type::cpp_type_out;

                if constexpr (Id == test_case_id::VALID)
                {
                    return test_case_type{
                        get_iota_values<cpp_type_in>(3, 1),
                        get_iota_values<cpp_type_index>(3, 0),
                        get_iota_values<cpp_type_out>(3, 1),
                    };
                }
                else if constexpr (Id == test_case_id::OOB)
                {
                    constexpr auto maxv{ std::numeric_limits<cpp_type_index>::max() };
                    constexpr auto minv{ std::is_unsigned_v<cpp_type_index> ? maxv : std::numeric_limits<cpp_type_index>::min() };
                    constexpr auto negv{ static_cast<cpp_type_index>(4) };
                    constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                    return test_case_type{
                        get_iota_values<cpp_type_in>(3, 1),
                        make_mem_buffer<cpp_type_index>({ minv, negv, maxv }),
                        get_same_values<cpp_type_out>(3, invalid),
                    };
                }
                else if constexpr (Id == test_case_id::INVALID)
                {
                    constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                    return test_case_type{
                        get_iota_values<cpp_type_in>(3, 1),
                        get_same_values<cpp_type_index>(3, invalid),
                        get_same_values<cpp_type_out>(3, TypeCodeIn == NPY_BOOL ? 1 : invalid),
                    };
                }
                else if constexpr (Id == test_case_id::INVALID_DEFAULT)
                {
                    constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                    constexpr auto default_val{ static_cast<cpp_type_out>(0) };
                    return test_case_type{
                        get_iota_values<cpp_type_in>(3, 1),
                        get_same_values<cpp_type_index>(3, invalid),
                        get_same_values<cpp_type_out>(3, TypeCodeIn == NPY_BOOL ? 1 : default_val),
                        default_val,
                    };
                }
            }
        };
    };

    file_suite mbget_ops = []
    {
        "mbget_valid"_test = mbget_tests::test<test_case_id::VALID>{} | SupportedParams{};
        "mbget_oob"_test = mbget_tests::test<test_case_id::OOB>{} | SupportedParams{};
        //NOTYET: "mbget_invalid"_test = mbget_tests::test<test_case_id::INVALID>{} | SupportedParams{};
        //NOTYET:"mbget_invalid_default"_test = mbget_tests::test<test_case_id::INVALID_DEFAULT>{} | SupportedParams{};
    };
}