#include "riptide_python_test.h"

#include "numpy_traits.h"
#include "Bins.h"

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
#include <random>
#include <utility>
#include <variant>

using namespace riptide_python_test::internal;
using namespace riptide_utility::internal;
using namespace boost::ut;
using riptide_utility::ut::file_suite;

namespace
{
    using SupportedTypeCodeInTypes =
        std::tuple<typecode_to_type<NPY_TYPES::NPY_BOOL>, typecode_to_type<NPY_TYPES::NPY_INT8>,
                   typecode_to_type<NPY_TYPES::NPY_INT16>, typecode_to_type<NPY_TYPES::NPY_INT32>,
                   typecode_to_type<NPY_TYPES::NPY_INT64>, typecode_to_type<NPY_TYPES::NPY_UINT8>,
                   typecode_to_type<NPY_TYPES::NPY_UINT16>, typecode_to_type<NPY_TYPES::NPY_UINT32>,
                   typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
                   typecode_to_type<NPY_TYPES::NPY_DOUBLE>, typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>>;

    using SupportedTypeCodeIndexTypes = std::tuple<typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>>;

    using SupportedParams = decltype(tuple_prod(SupportedTypeCodeInTypes{}, SupportedTypeCodeIndexTypes{}));

    template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
    struct bins_tester
    {
        static constexpr auto typecode_bin = NPY_FLOAT64;
        static constexpr auto typecode_out = NPY_INT8; // TODO: only true if len(array) < 100

        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;
        using cpp_type_bin = riptide::numpy_cpp_type_t<typecode_bin>;
        using cpp_type_index = riptide::numpy_cpp_type_t<TypeCodeIndex>;
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec(const_buffer<cpp_type_in> const in_values, const_buffer<cpp_type_bin> const bin_values,
                         const_buffer<cpp_type_index> const index_values, const_buffer<cpp_type_out> const expected,
                         reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeIn>, typecode_to_type<TypeCodeIndex>>;

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const in_array{ pyarray_from_array<TypeCodeIn>(in_values) };
            typed_expect<desc_type>((in_array != nullptr) >> fatal) << caller_loc;
            auto const bin_array{ pyarray_from_array<typecode_bin>(bin_values) };
            typed_expect<desc_type>((bin_array != nullptr) >> fatal) << caller_loc;
            auto const index_array{ pyarray_from_array<TypeCodeIndex>(index_values) };
            typed_expect<desc_type>((index_array != nullptr) >> fatal) << caller_loc;

            pyobject_ptr count_array{ PyObject_CallMethod(riptide_module_p, "NanInfCountFromSort", "OO", in_array.get(),
                                                          index_array.get()) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            int32_t mode{ 0 };
            pyobject_ptr retval{ PyObject_CallMethod(riptide_module_p, "BinsToCutsSorted", "OOOOi", in_array.get(),
                                                     bin_array.get(), index_array.get(), count_array.get(), mode) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            auto const actual_values{ cast_pyarray_values_as<typecode_out>(&retval) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            for (size_t i{ 0 }; i < expected.size(); ++i)
            {
                auto const expected_value{ expected[i] };
                auto const actual_value{ actual_values[i] };
                typed_expect<desc_type>(equal_to_nan_aware(actual_value, expected_value))
                    << "index:" << i << ", expected:" << expected_value << ", actual:" << actual_value << caller_loc;
            }
        }
    };

    enum class test_case_id
    {
        VALID = 0,
        INVALID = 1,
        MIXED = 2
    };

    struct bins_tests
    {
        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
        using tester_type = bins_tester<TypeCodeIn, TypeCodeIndex>;

        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
        struct test_case
        {
            using this_tester_type = tester_type<TypeCodeIn, TypeCodeIndex>;

            using cpp_type_in = typename this_tester_type::cpp_type_in;
            using cpp_type_bin = typename this_tester_type::cpp_type_bin;
            using cpp_type_index = typename this_tester_type::cpp_type_index;
            using cpp_type_out = typename this_tester_type::cpp_type_out;

            any_const_buffer<cpp_type_in> in_values_;
            any_const_buffer<cpp_type_bin> bin_values_;
            any_const_buffer<cpp_type_index> index_values_;
            any_const_buffer<cpp_type_out> expected_values_;
            reflection::source_location loc_;

            template <template <typename> typename BufferInT, template <typename> typename BufferOutT>
            test_case(BufferInT<cpp_type_in> && in_values, BufferInT<cpp_type_bin> && bin_values,
                      any_const_buffer<cpp_type_index> index_values, BufferOutT<cpp_type_out> && expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : in_values_{ std::move(in_values) }
                , bin_values_{ std::move(bin_values) }
                , index_values_{ std::move(index_values) }
                , expected_values_{ std::move(expected_values) }
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

                auto const testcase = get_test_case<Id, TypeCodeIn, TypeCodeIndex>();

                this_tester_type::exec(testcase.in_values_, testcase.bin_values_, testcase.index_values_,
                                       testcase.expected_values_, testcase.loc_);
            }
        };

        template <test_case_id Id, NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeIndex>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCodeIn, TypeCodeIndex>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_bin = typename test_case_type::cpp_type_bin;
            using cpp_type_index = typename test_case_type::cpp_type_index;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            // NPY_BOOL has no invalid, so always treat it as the VALID test.
            if constexpr (Id == test_case_id::VALID || TypeCodeIn == NPY_TYPES::NPY_BOOL)
            {
                if constexpr (TypeCodeIn == NPY_TYPES::NPY_BOOL)
                {
                    return test_case_type{ make_mem_buffer<cpp_type_in>({ 0, 1 }), make_mem_buffer<cpp_type_bin>({ 0.0, 1.0 }),
                                           make_mem_buffer<cpp_type_index>({ 0, 1 }), make_mem_buffer<cpp_type_out>({ 2, 2 }) };
                }

                else
                {
                    return test_case_type{ make_mem_buffer<cpp_type_in>({ 0, 1, 2, 3, 4 }),
                                           make_mem_buffer<cpp_type_bin>({ 0.0, 1.0, 2.0, 3.0, 4.0 }),
                                           make_mem_buffer<cpp_type_index>({ 0, 1, 2, 3, 4 }),
                                           make_mem_buffer<cpp_type_out>({ 2, 2, 3, 4, 5 }) };
                }
            }
            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_in>::value };
                return test_case_type{ make_mem_buffer<cpp_type_in>({ invalid, invalid, invalid, invalid, invalid }),
                                       make_mem_buffer<cpp_type_bin>({ 0.0, 1.0, 2.0, 3.0, 4.0 }),
                                       make_mem_buffer<cpp_type_index>({ 0, 1, 2, 3, 4 }),
                                       make_mem_buffer<cpp_type_out>({ 0, 0, 0, 0, 0 }) };
            }
        }
    };

    file_suite bins_ops = []
    {
        "bins_to_cuts_sorted_valid"_test = bins_tests::test<test_case_id::VALID>{} | SupportedParams{};
        "bins_to_cuts_sorted_invalid"_test = bins_tests::test<test_case_id::INVALID>{} | SupportedParams{};
    };
}
