#include "riptide_python_test.h"

#include "numpy_traits.h"
#include "Convert.h"

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
#if 1
        std::tuple<typecode_to_type<NPY_TYPES::NPY_INT8>>;
#else
        std::tuple<typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>,
                   typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>>;
#endif

    using SupportedParams = SupportedTypeCodeInNumTypes;

    template <NPY_TYPES TypeCodeIn>
    struct combine_accum_tester
    {
        static constexpr auto typecode_filter = NPY_TYPES::NPY_BOOL;
        static constexpr auto typecode_out = TypeCodeIn;

        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;
        using cpp_type_filter = riptide::numpy_cpp_type_t<typecode_filter>;
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec(const_buffer<cpp_type_in> const in_values,
                         std::optional<const_buffer<cpp_type_filter>> const filter_values,
                         const_buffer<cpp_type_out> const expected,
                         reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = typecode_to_type<TypeCodeIn>;
            auto const desc{ "" };

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const in_array{ pyarray_from_array<TypeCodeIn>(in_values) };
            typed_expect<desc_type>(desc, (in_array != nullptr) >> fatal) << caller_loc;
            auto const num_uniques{ expected.size() };
            pyobject_ptr const filter_array{ filter_values.has_value() ?
                                                 pyarray_from_array<typecode_filter>(filter_values.value()) :
                                                 pyobject_ptr{ Py_None } };

            pyobject_ptr retval{ PyObject_CallMethod(riptide_module_p, "CombineAccum1Filter", "OiO", in_array.get(), num_uniques,
                                                     filter_array.get()) };
            typed_expect<desc_type>(desc, no_pyerr() >> fatal) << caller_loc;

            typed_expect<desc_type>(desc, (PyList_Check(retval.get()) && PyList_Size(retval.get()) == 3) >> fatal) << caller_loc;
            pyobject_ptr out_reval{ pyobject_newref(PyList_GetItem(retval.get(), 0)) };
            auto * const first_retval{ PyList_GetItem(retval.get(), 1) };
            auto const uniques_count_retval{ PyLong_AsLongLong(PyList_GetItem(retval.get(), 2)) };
            typed_expect<desc_type>(desc, no_pyerr() >> fatal) << caller_loc;

            auto const actual_values{ cast_pyarray_values_as<typecode_out>(&out_reval) };
            typed_expect<desc_type>(desc, no_pyerr() >> fatal) << caller_loc;
            typed_expect<desc_type>(desc, (actual_values.size() == expected.size()) >> fatal) << caller_loc;

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
        ALL_UNIQUE,
        ALL_SAME,
        OOB,
        ALL_UNIQUE_FILT,
        ALL_SAME_FILT,
        OOB_FILT,
    };

    struct combine_accum_tests
    {
        template <NPY_TYPES TypeCodeIn>
        using tester_type = combine_accum_tester<TypeCodeIn>;

        template <NPY_TYPES TypeCodeIn>
        struct test_case
        {
            using this_tester_type = tester_type<TypeCodeIn>;

            using cpp_type_in = typename this_tester_type::cpp_type_in;
            using cpp_type_filter = typename this_tester_type::cpp_type_filter;
            using cpp_type_out = typename this_tester_type::cpp_type_out;

            any_const_buffer<cpp_type_in> in_values_;
            std::optional<any_const_buffer<cpp_type_filter>> filter_values_;
            any_const_buffer<cpp_type_out> expected_values_;
            reflection::source_location loc_;

            template <template <typename> typename BufferInT, template <typename> typename BufferOutT>
            test_case(BufferInT<cpp_type_in> && in_values, BufferOutT<cpp_type_out> && expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : in_values_{ std::move(in_values) }
                , expected_values_{ std::move(expected_values) }
                , loc_{ loc }
            {
            }

            template <template <typename> typename BufferInT, template <typename> typename BufferFilterT,
                      template <typename> typename BufferOutT>
            test_case(BufferInT<cpp_type_in> && in_values, BufferFilterT<cpp_type_filter> && filter_values,
                      BufferOutT<cpp_type_out> && expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : in_values_{ std::move(in_values) }
                , filter_values_{ std::move(filter_values) }
                , expected_values_{ std::move(expected_values) }
                , loc_{ loc }
            {
            }
        };

        template <test_case_id Id>
        struct test
        {
            template <typename TypeCodeInType>
            void operator()() const
            {
                constexpr auto TypeCodeIn = TypeCodeInType::value;

                using this_tester_type = tester_type<TypeCodeIn>;

                auto const testcase = get_test_case<Id, TypeCodeIn>{}();

                this_tester_type::exec(testcase.in_values_, testcase.filter_values_, testcase.expected_values_, testcase.loc_);
            }
        };

        template <test_case_id Id, NPY_TYPES TypeCodeIn>
        struct get_test_case
        {
            auto operator()()
            {
                using test_case_type = test_case<TypeCodeIn>;
                using cpp_type_in = typename test_case_type::cpp_type_in;
                using cpp_type_filter = typename test_case_type::cpp_type_filter;
                using cpp_type_out = typename test_case_type::cpp_type_out;

                if constexpr (Id == test_case_id::ALL_UNIQUE)
                {
                    return test_case_type{
                        get_iota_values<cpp_type_in>(10, 0),
                        get_iota_values<cpp_type_out>(10, 0),
                    };
                }
                if constexpr (Id == test_case_id::ALL_SAME)
                {
                    return test_case_type{
                        get_same_values<cpp_type_in>(10, 8),
                        get_same_values<cpp_type_out>(10, 1),
                    };
                }
                else if constexpr (Id == test_case_id::OOB)
                {
                    constexpr auto maxv{ std::numeric_limits<cpp_type_in>::max() };
                    constexpr auto minv{ std::is_unsigned_v<cpp_type_in> ? maxv : std::numeric_limits<cpp_type_in>::min() };
                    constexpr auto negv{ static_cast<cpp_type_in>(4) };
                    return test_case_type{
                        make_mem_buffer<cpp_type_in>({ minv, negv, maxv }),
                        get_same_values<cpp_type_out>(3, 0),
                    };
                }
                if constexpr (Id == test_case_id::ALL_UNIQUE_FILT)
                {
                    return test_case_type{
                        get_iota_values<cpp_type_in>(5, 0),
                        make_mem_buffer<cpp_type_filter>({ true, true, false, true, false }),
                        make_mem_buffer<cpp_type_out>({ 0, 1, 0, 2, 0 }),
                    };
                }
                if constexpr (Id == test_case_id::ALL_SAME_FILT)
                {
                    return test_case_type{
                        get_same_values<cpp_type_in>(5, 3),
                        make_mem_buffer<cpp_type_filter>({ true, true, false, true, false }),
                        make_mem_buffer<cpp_type_out>({ 1, 1, 0, 1, 0 }),
                    };
                }
                else if constexpr (Id == test_case_id::OOB_FILT)
                {
                    constexpr auto maxv{ std::numeric_limits<cpp_type_in>::max() };
                    constexpr auto minv{ std::is_unsigned_v<cpp_type_in> ? maxv : std::numeric_limits<cpp_type_in>::min() };
                    constexpr auto negv{ static_cast<cpp_type_in>(4) };
                    return test_case_type{
                        make_mem_buffer<cpp_type_in>({ minv, negv, maxv }),
                        make_mem_buffer<cpp_type_filter>({ true, true, true }),
                        get_same_values<cpp_type_out>(3, 0),
                    };
                }
            }
        };
    };

    file_suite combine_accum_ops = []
    {
        "combine_accum_all_unique"_test = combine_accum_tests::test<test_case_id::ALL_UNIQUE>{} | SupportedParams{};
        "combine_accum_all_same"_test = combine_accum_tests::test<test_case_id::ALL_SAME>{} | SupportedParams{};
        "combine_accum_oob"_test = combine_accum_tests::test<test_case_id::OOB>{} | SupportedParams{};
        "combine_accum_all_unique_filt"_test = combine_accum_tests::test<test_case_id::ALL_UNIQUE_FILT>{} | SupportedParams{};
        "combine_accum_all_same_filt"_test = combine_accum_tests::test<test_case_id::ALL_SAME_FILT>{} | SupportedParams{};
        "combine_accum_oob_filt"_test = combine_accum_tests::test<test_case_id::OOB_FILT>{} | SupportedParams{};
    };
}