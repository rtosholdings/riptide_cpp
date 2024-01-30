#include "riptide_python_test.h"

#include "MultiKey.h"

#include "buffer.h"
#include "mem_buffer.h"
#include "np_util.h"
#include "tuple_util.h"
#include "ut_extensions.h"

#include "ut_core.h"

#include <array>
#include <tuple>
#include <type_traits>

using namespace riptide_python_test::internal;
using namespace riptide_utility::internal;
using namespace boost::ut;
using riptide_utility::ut::file_suite;

namespace
{
    enum class Direction
    {
        FORWARD,
        BACKWARD
    };

    template <Direction Dir>
    using direction_to_type = std::integral_constant<Direction, Dir>;

    using SupportedTypeCodeKeyTypes = std::tuple<
        typecode_to_type<NPY_TYPES::NPY_BOOL>, typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>,
        typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>, typecode_to_type<NPY_TYPES::NPY_UINT8>,
        typecode_to_type<NPY_TYPES::NPY_UINT16>, typecode_to_type<NPY_TYPES::NPY_UINT32>, typecode_to_type<NPY_TYPES::NPY_UINT64>,
        typecode_to_type<NPY_TYPES::NPY_FLOAT>, typecode_to_type<NPY_TYPES::NPY_DOUBLE>
// NPY_LONGDOUBLE doesn't hash properly with GCC on Linux (see riptide_cpp#46)
#ifndef __GNUC__
        ,
        typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>
#endif
        >;

    using SupportedTypeCodeValTypes =
        std::tuple<typecode_to_type<NPY_TYPES::NPY_INT32>, typecode_to_type<NPY_TYPES::NPY_INT64>,
                   typecode_to_type<NPY_TYPES::NPY_FLOAT32>, typecode_to_type<NPY_TYPES::NPY_FLOAT64>>;

    using SupportedDirectionTypes = std::tuple<direction_to_type<Direction::FORWARD>, direction_to_type<Direction::BACKWARD>>;

    using SupportedParams =
        decltype(tuple_prod(SupportedTypeCodeKeyTypes{}, tuple_prod(SupportedTypeCodeValTypes{}, SupportedDirectionTypes{})));
}

namespace
{
    template <NPY_TYPES TypeCodeKeyIn, NPY_TYPES TypeCodeValIn, Direction Dir>
    struct mkalign_tester
    {
        // Matches test in riptable.
        using cpp_type_key_in = riptide::numpy_cpp_type_t<TypeCodeKeyIn>;
        using cpp_type_val_in = riptide::numpy_cpp_type_t<TypeCodeValIn>;

        static constexpr auto typecode_out = NPY_INT32; // TODO: only true if len(array) < 2M!
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec(const_buffer<cpp_type_key_in> const keys1, const_buffer<cpp_type_key_in> const keys2,
                         const_buffer<cpp_type_val_in> const vals1, const_buffer<cpp_type_val_in> const vals2,
                         const_buffer<cpp_type_out> const expected_values,
                         reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeKeyIn>, typecode_to_type<TypeCodeValIn>, direction_to_type<Dir>>;

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const keys1_array{ pyarray_from_array<TypeCodeKeyIn>(keys1) };
            auto const keys2_array{ pyarray_from_array<TypeCodeKeyIn>(keys2) };
            auto const vals1_array{ pyarray_from_array<TypeCodeValIn>(vals1) };
            auto const vals2_array{ pyarray_from_array<TypeCodeValIn>(vals2) };
            auto * is_forward{ Dir == Direction::FORWARD ? Py_True : Py_False };
            auto * allow_exact{ Py_True };

            pyobject_ptr retval{ PyObject_CallMethod(riptide_module_p, "MultiKeyAlign32", "(O)(O)OOOO", keys1_array.get(),
                                                     keys2_array.get(), vals1_array.get(), vals2_array.get(), is_forward,
                                                     allow_exact) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            typed_expect<desc_type>(PyArray_Check(retval.get()) >> fatal) << caller_loc;

            auto const actual_values{ cast_pyarray_values_as<typecode_out>(&retval) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            typed_expect<desc_type>(actual_values.size() == expected_values.size() >> fatal) << caller_loc;

            for (size_t i{ 0 }; i < expected_values.size(); ++i)
            {
                auto const expected_value{ expected_values[i] };
                auto const actual_value{ actual_values[i] };
                typed_expect<desc_type>(equal_to_nan_aware(actual_value, expected_value))
                    << "index:" << i << ", expected:" << to_out(expected_value) << ", actual:" << to_out(actual_value)
                    << caller_loc;
            }
        }
    };

    enum class test_case_id
    {
        VALID,
        MIXED,
        INVALID,
    };

    struct mkalign_tests
    {
        template <NPY_TYPES TypeCodeKey, NPY_TYPES TypeCodeVal, Direction Dir>
        using tester_type = mkalign_tester<TypeCodeKey, TypeCodeVal, Dir>;

        template <NPY_TYPES TypeCodeKey, NPY_TYPES TypeCodeVal, Direction Dir>
        struct test_case
        {
            using this_tester_type = tester_type<TypeCodeKey, TypeCodeVal, Dir>;

            using cpp_type_key_in = typename this_tester_type::cpp_type_key_in;
            using cpp_type_val_in = typename this_tester_type::cpp_type_val_in;
            using cpp_type_out = typename this_tester_type::cpp_type_out;

            any_const_buffer<cpp_type_key_in> keys1_;
            any_const_buffer<cpp_type_key_in> keys2_;
            any_const_buffer<cpp_type_val_in> vals1_;
            any_const_buffer<cpp_type_val_in> vals2_;
            any_const_buffer<cpp_type_out> expected_values_;
            reflection::source_location loc_;

            template <template <typename> typename BufferKeyInT, template <typename> typename BufferValInT,
                      template <typename> typename BufferOutT>
            test_case(BufferKeyInT<cpp_type_key_in> && keys1, BufferKeyInT<cpp_type_key_in> && keys2,
                      BufferValInT<cpp_type_val_in> && vals1, BufferValInT<cpp_type_val_in> && vals2,
                      BufferOutT<cpp_type_out> && expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : keys1_{ std::move(keys1) }
                , keys2_{ std::move(keys2) }
                , vals1_{ std::move(vals1) }
                , vals2_{ std::move(vals2) }
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
                using TypeCodeKeyType = std::tuple_element_t<0, T>;
                using T2 = std::tuple_element_t<1, T>;
                using TypeCodeValType = std::tuple_element_t<0, T2>;
                using DirectionType = std::tuple_element_t<1, T2>;

                constexpr auto TypeCodeKey = TypeCodeKeyType::value;
                constexpr auto TypeCodeVal = TypeCodeValType::value;
                constexpr auto Dir = DirectionType::value;

                auto const testcase = get_test_case<Id, TypeCodeKey, TypeCodeVal, Dir>();
                tester_type<TypeCodeKey, TypeCodeVal, Dir>::exec(testcase.keys1_, testcase.keys2_, testcase.vals1_,
                                                                 testcase.vals2_, testcase.expected_values_, testcase.loc_);
            }
        };

        template <test_case_id Id, NPY_TYPES TypeCodeKey, NPY_TYPES TypeCodeVal, Direction Dir>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCodeKey, TypeCodeVal, Dir>;
            using cpp_type_key_in = typename test_case_type::cpp_type_key_in;
            using cpp_type_val_in = typename test_case_type::cpp_type_val_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            constexpr auto first_idx{ Dir == Direction::FORWARD ? 0 : 2 };

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    get_same_values<cpp_type_key_in>(3, 1),      get_same_values<cpp_type_key_in>(3, 1),
                    get_same_values<cpp_type_val_in>(3, 1),      get_same_values<cpp_type_val_in>(3, 1),
                    get_same_values<cpp_type_out>(3, first_idx),
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid_idx{ riptide::invalid_for_type<cpp_type_out>::value };
                constexpr auto invalid_val{ riptide::invalid_for_type<cpp_type_val_in>::value };
                return test_case_type{
                    get_same_values<cpp_type_key_in>(3, 1),
                    get_same_values<cpp_type_key_in>(3, 1),
                    make_mem_buffer<cpp_type_val_in>({ 1, invalid_val, 1 }),
                    make_mem_buffer<cpp_type_val_in>({ 1, invalid_val, 1 }),
                    make_mem_buffer<cpp_type_out>({ first_idx, invalid_idx, first_idx }),
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid_idx{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_same_values<cpp_type_key_in>(3, 1),
                    get_same_values<cpp_type_key_in>(3, 1),
                    get_invalid_values<cpp_type_val_in>(3),
                    get_invalid_values<cpp_type_val_in>(3),
                    make_mem_buffer<cpp_type_out>({ invalid_idx, invalid_idx, invalid_idx }),
                };
            }
        }
    };

    file_suite mkalign_ops = []
    {
        "mkalign_valid"_test = mkalign_tests::test<test_case_id::VALID>{} | SupportedParams{};
        "mkalign_mixed"_test = mkalign_tests::test<test_case_id::MIXED>{} | SupportedParams{};
        "mkalign_invalid"_test = mkalign_tests::test<test_case_id::INVALID>{} | SupportedParams{};
    };
}
