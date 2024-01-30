#include "riptide_python_test.h"

#include "numpy_traits.h"

#include "buffer.h"
#include "mem_buffer.h"
#include "np_util.h"
#include "tuple_util.h"
#include "ut_extensions.h"
#include "vector_util.h"

#include "ut_core.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <numeric>
#include <vector>
#include <random>
#include <utility>
#include <variant>

using namespace riptide_python_test::internal;
using namespace riptide_utility::internal;
using namespace boost::ut;
using riptide_utility::ut::file_suite;

namespace
{
    enum class test_case_id : int32_t
    {
        VALID = 0,
        INVALID = 1
    };

    enum class sort_direction : int32_t
    {
        ASCENDING = 0,
        DESCENDING = 1,
        MIXED = 2
    };

    template <sort_direction Order>
    using sort_direction_to_type = std::integral_constant<sort_direction, Order>;

    using SupportedIntegerTypes = std::tuple<typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>,
                                             typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>,
                                             typecode_to_type<NPY_TYPES::NPY_UINT8>, typecode_to_type<NPY_TYPES::NPY_UINT16>,
                                             typecode_to_type<NPY_TYPES::NPY_UINT32>, typecode_to_type<NPY_TYPES::NPY_UINT64>>;

    using SupportedFloatTypes = std::tuple<typecode_to_type<NPY_TYPES::NPY_FLOAT>, typecode_to_type<NPY_TYPES::NPY_DOUBLE>,
                                           typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>>;

    using SupportedSortOrder =
        std::tuple<sort_direction_to_type<sort_direction::ASCENDING>, sort_direction_to_type<sort_direction::DESCENDING>,
                   sort_direction_to_type<sort_direction::MIXED>>;

    // TODO: support testing string types as inputs.
    // using SupportedStringTypes =
    //     std::tuple<std::tuple<typecode_to_type<NPY_TYPES::NPY_STRING>, typecode_to_type<NPY_TYPES::NPY_STRING>>,
    //                std::tuple<typecode_to_type<NPY_TYPES::NPY_UNICODE>, typecode_to_type<NPY_TYPES::NPY_UNICODE>>>;

    using SupportedParamsValid =
        decltype(tuple_prod(std::tuple_cat(SupportedIntegerTypes{}, SupportedFloatTypes{}), SupportedSortOrder{}));
    using SupportedParamsInvalid = decltype(tuple_prod(SupportedIntegerTypes{}, SupportedSortOrder{}));

    template <NPY_TYPES TypeCodeIn>
    struct lexsort_tester
    {
        // always true for LexSort32
        static constexpr auto typecode_out = NPY_INT32;

        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec(const std::vector<any_const_buffer<cpp_type_in>> && in_vals, const const_buffer<bool> sort_direction,
                         const const_buffer<cpp_type_out> expected,
                         reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeIn>>;

            const auto caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            const pyobject_ptr lst{ PyList_New(in_vals.size()) };

            for (size_t i = 0; i < in_vals.size(); i++)
            {
                const_buffer<cpp_type_in> inval = in_vals[i];
                const auto arr{ pyarray_from_array<TypeCodeIn>(inval) };
                typed_expect<desc_type>((arr != nullptr) >> fatal) << caller_loc;
                Py_IncRef(arr.get());
                PyList_SetItem(lst.get(), i, arr.get());
            }
            // PyObject_Print(lst.get(), stdout, 0);

            const pyobject_ptr pysort_direction{ pyarray_from_array<NPY_BOOL>(sort_direction) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            const pyobject_ptr args{ Py_BuildValue("(O)", lst.get()) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;
            const pyobject_ptr kwargs{ PyDict_New() };
            PyDict_SetItemString(kwargs.get(), "ascending", pysort_direction.get());
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            const pyobject_ptr lexsort{ PyObject_GetAttrString(riptide_module_p, "LexSort32") };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            pyobject_ptr retval{ PyObject_Call(lexsort.get(), args.get(), kwargs.get()) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            const auto actual_values{ cast_pyarray_values_as<typecode_out>(&retval) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            for (size_t i{ 0 }; i < expected.size(); ++i)
            {
                auto const expected_value{ expected[i] };
                auto const actual_value{ actual_values[i] };
                typed_expect<desc_type>(equal_to_nan_aware(actual_value, expected_value))
                    << "index:" << i << ", expected:" << to_out(expected_value) << ", actual:" << to_out(actual_value)
                    << caller_loc;
            }
        }
    };

    struct lexsort_tests
    {
        template <NPY_TYPES TypeCodeIn>
        using tester_type = lexsort_tester<TypeCodeIn>;

        template <NPY_TYPES TypeCodeIn>
        struct test_case
        {
            using this_tester_type = tester_type<TypeCodeIn>;

            using cpp_type_in = this_tester_type::cpp_type_in;
            using cpp_type_out = this_tester_type::cpp_type_out;

            std::vector<any_const_buffer<cpp_type_in>> vals_;
            any_const_buffer<bool> sort_direction_;
            any_const_buffer<cpp_type_out> expected_values_;
            reflection::source_location loc_;

            template <template <typename> typename BufferInT, template <typename> typename BufferOutT>
            test_case(std::vector<any_const_buffer<cpp_type_in>> && nvals, BufferInT<bool> && sort_direction,
                      BufferOutT<cpp_type_out> && expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : sort_direction_{ std::move(sort_direction) }
                , expected_values_{ std::move(expected_values) }
                , loc_{ loc }
            {
                vals_.swap(nvals);
            }
        };

        template <test_case_id Id>
        struct test
        {
            template <typename T>
            void operator()()
            {
                // (in, order)
                constexpr auto TypeCodeIn = std::tuple_element_t<0, T>::value;
                constexpr auto Order = std::tuple_element_t<1, T>::value;

                const auto testcase = get_test_case<Id, TypeCodeIn, Order>{}();
                tester_type<TypeCodeIn>::exec(std::move(testcase.vals_), testcase.sort_direction_, testcase.expected_values_,
                                              testcase.loc_);
            }
        };

        template <test_case_id Id, NPY_TYPES TypeCodeIn, sort_direction Order>
        struct get_test_case;

        template <NPY_TYPES TypeCodeIn, sort_direction Order>
        struct get_test_case<test_case_id::VALID, TypeCodeIn, Order>
        {
            auto operator()()
            {
                using test_case_type = test_case<TypeCodeIn>;
                using cpp_type_in = typename test_case_type::cpp_type_in;
                using cpp_type_out = typename test_case_type::cpp_type_out;

                return test_case_type{
                    make_vector<any_const_buffer<cpp_type_in>>(make_mem_buffer<cpp_type_in>({ 1, 2, 3, 4, 5, 6 }),
                                                               make_mem_buffer<cpp_type_in>({ 2, 2, 1, 1, 3, 3 })),
                    (Order == sort_direction::ASCENDING  ? get_same_values<bool>(2, true) :
                     Order == sort_direction::DESCENDING ? get_same_values<bool>(2, false) :
                                                           make_mem_buffer<bool>({ true, false })),
                    (Order == sort_direction::ASCENDING  ? make_mem_buffer<cpp_type_out>({ 2, 3, 0, 1, 4, 5 }) :
                     Order == sort_direction::DESCENDING ? make_mem_buffer<cpp_type_out>({ 5, 4, 1, 0, 3, 2 }) :
                                                           make_mem_buffer<cpp_type_out>({ 4, 5, 0, 1, 2, 3 }))
                };
            }
        };

        template <NPY_TYPES TypeCodeIn, sort_direction Order>
        struct get_test_case<test_case_id::INVALID, TypeCodeIn, Order>
        {
            auto operator()()
            {
                using test_case_type = test_case<TypeCodeIn>;
                using cpp_type_in = typename test_case_type::cpp_type_in;
                using cpp_type_out = typename test_case_type::cpp_type_out;

                constexpr auto is_signed{ std::is_signed_v<cpp_type_in> };
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_in>::value };
                return test_case_type{
                    make_vector<any_const_buffer<cpp_type_in>>(make_mem_buffer<cpp_type_in>({ 1, 2, invalid, invalid, 5, 6 }),
                                                               make_mem_buffer<cpp_type_in>({ 2, invalid, 1, 1, invalid, 3 })),
                    (Order == sort_direction::ASCENDING  ? get_same_values<bool>(2, true) :
                     Order == sort_direction::DESCENDING ? get_same_values<bool>(2, false) :
                                                           make_mem_buffer<bool>({ false, true })),
                    (Order == sort_direction::ASCENDING  ? (is_signed ? make_mem_buffer<cpp_type_out>({ 1, 4, 2, 3, 0, 5 }) :
                                                                        make_mem_buffer<cpp_type_out>({ 2, 3, 0, 5, 1, 4 })) :
                     Order == sort_direction::DESCENDING ? (is_signed ? make_mem_buffer<cpp_type_out>({ 5, 0, 2, 3, 4, 1 }) :
                                                                        make_mem_buffer<cpp_type_out>({ 4, 1, 5, 0, 2, 3 })) :
                                                           (is_signed ? make_mem_buffer<cpp_type_out>({ 4, 1, 2, 3, 0, 5 }) :
                                                                        make_mem_buffer<cpp_type_out>({ 2, 3, 0, 5, 4, 1 })))
                };
            }
        };
    };

    file_suite lex_sort = []
    {
        "lexsort_valid"_test = lexsort_tests::test<test_case_id::VALID>{} | SupportedParamsValid{};
        "lexsort_invalid"_test = lexsort_tests::test<test_case_id::INVALID>{} | SupportedParamsInvalid{};
    };
}