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
    enum class search_mode : int32_t
    {
        LEFTPLUS = 0, // per rt.searchsorted(): 'leftplus' is a new option in riptable where values > get a 0
        LEFT = 1,     // leftmost
        RIGHT = 2,    // rightmost
    };

    std::string to_string(search_mode const mode)
    {
        return mode == search_mode::LEFTPLUS ? "leftplus" :
               mode == search_mode::LEFT     ? "left" :
               mode == search_mode::RIGHT    ? "right" :
                                               std::to_string(static_cast<int>(mode)) + '?';
    }

    template <search_mode Mode>
    using search_mode_to_type = std::integral_constant<search_mode, Mode>;

    using SupportedTypeCodeInNumTypes = std::tuple<typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>,
                                                   typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>,
                                                   typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
                                                   typecode_to_type<NPY_TYPES::NPY_DOUBLE>>;

    using SupportedTypeCodeBinNumTypes =
        std::tuple<typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>,
                   typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>,
                   typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
                   typecode_to_type<NPY_TYPES::NPY_DOUBLE>>;

    // string in,bin pairs must have same types.
    // TODO: Need to add support for testing string/unicode types, mapping fixed-length strings.
    using SupportedTypeCodeInBinStrTypes =
        std::tuple< //TODO//std::tuple<typecode_to_type<NPY_TYPES::NPY_STRING>, typecode_to_type<NPY_TYPES::NPY_STRING>>
                    //TODO//,std::tuple<typecode_to_type<NPY_TYPES::NPY_UNICODE>, typecode_to_type<NPY_TYPES::NPY_UNICODE>>
            >;

    using SupportedTypeCodeInBinTypes =
        decltype(std::tuple_cat(tuple_prod(SupportedTypeCodeInNumTypes{}, SupportedTypeCodeBinNumTypes{})),
                 SupportedTypeCodeInBinStrTypes{});

    using SupportedSearchModeTypes = std::tuple< //BROKEN//search_mode_to_type<search_mode::LEFTPLUS>,
        search_mode_to_type<search_mode::LEFT>, search_mode_to_type<search_mode::RIGHT>>;

    using SupportedParams = decltype(tuple_prod(SupportedTypeCodeInBinTypes{}, SupportedSearchModeTypes{}));

    template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeBin>
    struct bins_tester
    {
        static constexpr auto typecode_out = NPY_INT8; // TODO: only true if len(array) < 100

        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;
        using cpp_type_bin = riptide::numpy_cpp_type_t<TypeCodeBin>;
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec(const_buffer<cpp_type_in> const in_values, const_buffer<cpp_type_bin> const bin_values,
                         search_mode const mode, const_buffer<cpp_type_out> const expected,
                         reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeIn>, typecode_to_type<TypeCodeBin>>;
            auto const desc = std::string("search_mode=") + to_string(mode);

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const in_array{ pyarray_from_array<TypeCodeIn>(in_values) };
            typed_expect<desc_type>(desc, (in_array != nullptr) >> fatal) << caller_loc;
            auto const bin_array{ pyarray_from_array<TypeCodeBin>(bin_values) };
            typed_expect<desc_type>(desc, (bin_array != nullptr) >> fatal) << caller_loc;

            pyobject_ptr retval{ PyObject_CallMethod(riptide_module_p, "BinsToCutsBSearch", "OOi", in_array.get(), bin_array.get(),
                                                     mode) };
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
        BASIC,
        DUPS,
        OOBS,
    };

    struct bins_tests
    {
        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeBin>
        using tester_type = bins_tester<TypeCodeIn, TypeCodeBin>;

        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeBin, search_mode SearchMode>
        struct test_case
        {
            using this_tester_type = tester_type<TypeCodeIn, TypeCodeBin>;

            using cpp_type_in = typename this_tester_type::cpp_type_in;
            using cpp_type_bin = typename this_tester_type::cpp_type_bin;
            using cpp_type_out = typename this_tester_type::cpp_type_out;

            any_const_buffer<cpp_type_in> in_values_;
            any_const_buffer<cpp_type_bin> bin_values_;
            any_const_buffer<cpp_type_out> expected_values_;
            reflection::source_location loc_;

            template <template <typename> typename BufferInT, template <typename> typename BufferOutT>
            test_case(BufferInT<cpp_type_in> && in_values, BufferInT<cpp_type_bin> && bin_values,
                      BufferOutT<cpp_type_out> && expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : in_values_{ std::move(in_values) }
                , bin_values_{ std::move(bin_values) }
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
                using T2 = std::tuple_element_t<0, T>;
                using TypeCodeInType = std::tuple_element_t<0, T2>;
                using TypeCodeBinType = std::tuple_element_t<1, T2>;
                using SearchModeType = std::tuple_element_t<1, T>;

                constexpr auto TypeCodeIn = TypeCodeInType::value;
                constexpr auto TypeCodeBin = TypeCodeBinType::value;
                constexpr auto SearchMode = SearchModeType::value;

                using this_tester_type = tester_type<TypeCodeIn, TypeCodeBin>;

                auto const testcase = get_test_case<Id, TypeCodeIn, TypeCodeBin, SearchMode>{}();

                this_tester_type::exec(testcase.in_values_, testcase.bin_values_, SearchMode, testcase.expected_values_,
                                       testcase.loc_);
            }
        };

        template <test_case_id Id, NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeBin, search_mode SearchMode>
        struct get_test_case;

        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeBin, search_mode SearchMode>
        struct get_test_case<test_case_id::BASIC, TypeCodeIn, TypeCodeBin, SearchMode>
        {
            auto operator()()
            {
                using test_case_type = test_case<TypeCodeIn, TypeCodeBin, SearchMode>;
                using cpp_type_in = typename test_case_type::cpp_type_in;
                using cpp_type_bin = typename test_case_type::cpp_type_bin;
                using cpp_type_out = typename test_case_type::cpp_type_out;

                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 0, 1, 2, 3, 4, 5 }),
                    make_mem_buffer<cpp_type_bin>({ 0, 1, 2, 3, 4, 5 }),
                    (SearchMode != search_mode::RIGHT ? make_mem_buffer<cpp_type_out>({ 0, 1, 2, 3, 4, 5 }) :
                                                        make_mem_buffer<cpp_type_out>({ 1, 2, 3, 4, 5, 6 })),
                };
            }
        };

        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeBin, search_mode SearchMode>
        struct get_test_case<test_case_id::DUPS, TypeCodeIn, TypeCodeBin, SearchMode>
        {
            auto operator()()
            {
                using test_case_type = test_case<TypeCodeIn, TypeCodeBin, SearchMode>;
                using cpp_type_in = typename test_case_type::cpp_type_in;
                using cpp_type_bin = typename test_case_type::cpp_type_bin;
                using cpp_type_out = typename test_case_type::cpp_type_out;

                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 0, 1, 2, 3, 4, 5 }),
                    make_mem_buffer<cpp_type_bin>({ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 }),
                    (SearchMode != search_mode::RIGHT ? make_mem_buffer<cpp_type_out>({ 0, 2, 4, 6, 8, 10 }) :
                                                        make_mem_buffer<cpp_type_out>({ 2, 4, 6, 8, 10, 12 })),
                };
            }
        };

        template <NPY_TYPES TypeCodeIn, NPY_TYPES TypeCodeBin, search_mode SearchMode>
        struct get_test_case<test_case_id::OOBS, TypeCodeIn, TypeCodeBin, SearchMode>
        {
            auto operator()()
            {
                using test_case_type = test_case<TypeCodeIn, TypeCodeBin, SearchMode>;
                using cpp_type_in = typename test_case_type::cpp_type_in;
                using cpp_type_bin = typename test_case_type::cpp_type_bin;
                using cpp_type_out = typename test_case_type::cpp_type_out;

                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 0, 1, 2, 3, 4, 5 }),
                    make_mem_buffer<cpp_type_bin>({ 2, 3 }),
                    (SearchMode == search_mode::LEFT     ? make_mem_buffer<cpp_type_out>({ 0, 0, 0, 1, 2, 2 }) :
                     SearchMode == search_mode::LEFTPLUS ? make_mem_buffer<cpp_type_out>({ 0, 0, 0, 1, 0, 0 }) :
                                                           make_mem_buffer<cpp_type_out>({ 0, 0, 1, 2, 2, 2 })),
                };
            }
        };
    };

    file_suite bins_ops = []
    {
        "bins_to_cuts_bsearch_basic"_test = bins_tests::test<test_case_id::BASIC>{} | SupportedParams{};
        "bins_to_cuts_bsearch_dups"_test = bins_tests::test<test_case_id::DUPS>{} | SupportedParams{};
        "bins_to_cuts_bsearch_oobs"_test = bins_tests::test<test_case_id::OOBS>{} | SupportedParams{};
    };
}