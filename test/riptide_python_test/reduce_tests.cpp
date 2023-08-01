#include "riptide_python_test.h"

#include "CommonInc.h"
#include "Reduce.h"

#include "ut_extensions.h"

#include "MathWorker.h"
#include "numpy_traits.h"
#include "missing_values.h"
#include "simd/avx2.h"

#include "buffer.h"
#include "mem_buffer.h"
#include "np_util.h"
#include "ut_extensions.h"

#include "boost/ut.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <numeric>
#include <random>
#include <optional>
#include <utility>
#include <vector>

using namespace riptide_python_test::internal;
using namespace riptide_utility::internal;
using namespace boost::ut;
using boost::ut::suite;

namespace
{
    template <REDUCE_FUNCTIONS Fn>
    using reducefn_to_type = std::integral_constant<REDUCE_FUNCTIONS, Fn>;

    std::random_device dev{};

    // At this time, only floating types are supported, but not long double.
    using SupportedTypes = std::tuple</*bool, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,*/ float,
                                      double /*, long double*/>;

    using SupportedTypeCodeTypes =
        std::tuple<typecode_to_type<NPY_TYPES::NPY_BOOL>, typecode_to_type<NPY_TYPES::NPY_INT8>,
                   typecode_to_type<NPY_TYPES::NPY_INT16>, typecode_to_type<NPY_TYPES::NPY_INT32>,
                   typecode_to_type<NPY_TYPES::NPY_INT64>, typecode_to_type<NPY_TYPES::NPY_UINT8>,
                   typecode_to_type<NPY_TYPES::NPY_UINT16>, typecode_to_type<NPY_TYPES::NPY_UINT32>,
                   typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
                   typecode_to_type<NPY_TYPES::NPY_DOUBLE>, typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>>;

    template <NPY_TYPES TypeCode>
    auto get_prim_value(PyObject * const obj)
    {
        if constexpr (TypeCode == NPY_UINT64)
        {
            uint64_t result{};
            if (PyLong_Check(obj))
            {
                result = PyLong_AsUnsignedLongLong(obj);
            }
            return result;
        }
        else if constexpr (TypeCode == NPY_INT64)
        {
            int64_t result{};
            if (PyLong_Check(obj))
            {
                result = PyLong_AsLongLong(obj);
            }
            return result;
        }
        else if constexpr (TypeCode == NPY_DOUBLE)
        {
            double result{};
            if (PyFloat_Check(obj))
            {
                result = PyFloat_AsDouble(obj);
            }
            return result;
        }
        else
        {
            static_assert(std::is_void_v<TypeCode>, "Unexpected typecode");
        }
    }

    struct min_with_nan_passthru_tester
    {
        template <typename T>
        void operator()()
        {
            auto const invalid{ riptide::invalid_for_type<T>::value };
            T const valid{ 0 };

            {
                auto const result{ riptide::math::min_with_nan_passthru(invalid, valid) };
                typed_expect<T>(std::is_same_v<T, bool> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "invalid,valid";
            }
            {
                auto const result{ riptide::math::min_with_nan_passthru(valid, invalid) };
                typed_expect<T>(std::is_same_v<T, bool> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "valid,invalid";
            }
        }
    };

    struct max_with_nan_passthru_tester
    {
        template <typename T>
        void operator()()
        {
            auto const invalid{ riptide::invalid_for_type<T>::value };
            T const valid{ 0 };

            {
                auto const result{ riptide::math::max_with_nan_passthru(invalid, valid) };
                typed_expect<T>(std::is_same_v<T, bool> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "invalid,valid";
            }
            {
                auto const result{ riptide::math::max_with_nan_passthru(valid, invalid) };
                typed_expect<T>(std::is_same_v<T, bool> ? (result == valid) : ! riptide::invalid_for_type<T>::is_valid(result))
                    << "valid,invalid";
            }
        }
    };

    constexpr NPY_TYPES get_output_typecode(NPY_TYPES const typecode_in, REDUCE_FUNCTIONS const fn)
    {
        if (fn >= REDUCE_ARGMAX && fn <= REDUCE_NANARGMAX)
        {
            return NPY_UINT64;
        }

        if (fn >= 100 && fn < REDUCE_MIN)
        {
            return NPY_DOUBLE;
        }

        switch (typecode_in)
        {
        case NPY_UINT64:
            return NPY_UINT64;

        case NPY_FLOAT:
        case NPY_DOUBLE:
        case NPY_LONGDOUBLE:
            return NPY_DOUBLE;

        default:
            return NPY_INT64;
        }
    }

    template <NPY_TYPES TypeCodeIn, REDUCE_FUNCTIONS ReduceFn>
    struct reduce_tester
    {
        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;

        static constexpr auto typecode_out = get_output_typecode(TypeCodeIn, ReduceFn);
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec(const_buffer<cpp_type_in> const test_values, cpp_type_out const expected_value,
                         std::optional<int64_t> arg,
                         reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeIn>, reducefn_to_type<ReduceFn>>;

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const input_array{ pyarray_from_array<TypeCodeIn>(test_values) };
            typed_expect<desc_type>((input_array != nullptr) >> fatal) << caller_loc;

            pyobject_ptr const retval{ arg.has_value() ?
                                           PyObject_CallMethod(riptide_module_p, "Reduce", "Oii", input_array.get(), ReduceFn,
                                                               arg.value()) :
                                           PyObject_CallMethod(riptide_module_p, "Reduce", "Oi", input_array.get(), ReduceFn) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            auto const actual_value{ get_prim_value<typecode_out>(retval.get()) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            bool is_equal;
            if constexpr (std::is_floating_point_v<cpp_type_out>)
            {
                is_equal = equal_to_nan_aware(actual_value, expected_value, equal_within<cpp_type_out>{ 1e6 });
            }
            else
            {
                is_equal = equal_to_nan_aware(actual_value, expected_value);
            }

            typed_expect<desc_type>(is_equal)
                << "expected : " << to_out(expected_value) << ", actual : " << to_out(actual_value) << caller_loc;
        }
    };

    enum class test_case_id
    {
        VALID,
        MIXED,
        INVALID,
    };

    template <REDUCE_FUNCTIONS>
    struct reduce_tests;

    template <REDUCE_FUNCTIONS ReduceFn>
    struct reduce_tests_base
    {
        using Derived = reduce_tests<ReduceFn>;

        static constexpr REDUCE_FUNCTIONS reduce_fn = ReduceFn;

        template <NPY_TYPES TypeCode>
        using tester_type = reduce_tester<TypeCode, ReduceFn>;

        template <NPY_TYPES TypeCode>
        struct test_case
        {
            using cpp_type_in = typename tester_type<TypeCode>::cpp_type_in;
            using cpp_type_out = typename tester_type<TypeCode>::cpp_type_out;

            any_const_buffer<cpp_type_in> test_values_;
            cpp_type_out expected_value_;
            std::optional<int64_t> arg_;
            reflection::source_location loc_;

            template <template <typename> typename BufferInT>
            test_case(BufferInT<cpp_type_in> && test_values, cpp_type_out const expected_value, int64_t const arg,
                      reflection::source_location const & loc = reflection::source_location::current())
                : test_values_{ std::move(test_values) }
                , expected_value_{ expected_value }
                , arg_{ arg }
                , loc_{ loc }
            {
            }

            template <template <typename> typename BufferInT>
            test_case(BufferInT<cpp_type_in> && test_values, cpp_type_out const expected_value,
                      reflection::source_location const & loc = reflection::source_location::current())
                : test_values_{ std::move(test_values) }
                , expected_value_{ expected_value }
                , loc_{ loc }
            {
            }
        };

        template <test_case_id Id>
        struct test
        {
            template <typename T, NPY_TYPES TypeCode = T::value>
            void operator()() const
            {
                auto const testcase = Derived::template get_test_case<Id, TypeCode>();
                tester_type<TypeCode>::exec(testcase.test_values_, testcase.expected_value_, testcase.arg_, testcase.loc_);
            }
        };
    };

    //static constexpr size_t big_size = 2 * CMathWorker::WORK_ITEM_BIG;
    static constexpr size_t big_size = 3;

    template <>
    struct reduce_tests<REDUCE_SUM> : reduce_tests_base<REDUCE_SUM>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    get_same_values<cpp_type_in>(big_size, 1),
                    big_size,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(big_size),
                    TypeCode == NPY_BOOL ? true : invalid,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(big_size),
                    TypeCode == NPY_BOOL ? true : invalid,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_NANSUM> : reduce_tests_base<REDUCE_NANSUM>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    get_same_values<cpp_type_in>(3, 1),
                    3,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    0,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_MEAN> : reduce_tests_base<REDUCE_MEAN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    get_same_values<cpp_type_in>(3, 1),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 1. / 3. : invalid,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_NANMEAN> : reduce_tests_base<REDUCE_NANMEAN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    get_same_values<cpp_type_in>(3, 1),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 1. / 3. : 1. / 2.,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_VAR> : reduce_tests_base<REDUCE_VAR>
    {
        static constexpr int64_t ddof = 0; // numpy

        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    2. / 9.,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 2. / 9. : invalid,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                    ddof,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_NANVAR> : reduce_tests_base<REDUCE_NANVAR>
    {
        static constexpr int64_t ddof = 0; // numpy

        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    2. / 9.,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 2. / 9. : 1. / 4.,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                    ddof,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_STD> : reduce_tests_base<REDUCE_STD>
    {
        static constexpr int64_t ddof = 0; // numpy

        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    0.4714045207910317,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 0.4714045207910317 : invalid,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                    ddof,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_NANSTD> : reduce_tests_base<REDUCE_NANSTD>
    {
        static constexpr int64_t ddof = 0; // numpy

        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    0.4714045207910317,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 0.4714045207910317 : 0.5,
                    ddof,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                    ddof,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_MIN> : reduce_tests_base<REDUCE_MIN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    0,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 0 : invalid,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_NANMIN> : reduce_tests_base<REDUCE_NANMIN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    0,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    0,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_MAX> : reduce_tests_base<REDUCE_MAX>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 1 : invalid,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_NANMAX> : reduce_tests_base<REDUCE_NANMAX>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    invalid,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_ARGMIN> : reduce_tests_base<REDUCE_ARGMIN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    TypeCode == NPY_BOOL ? 0 : 1,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    0,
                };
            }
        }
    };

    template <>
    struct reduce_tests<REDUCE_NANARGMIN> : reduce_tests_base<REDUCE_NANARGMIN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_in = typename test_case_type::cpp_type_in;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    make_mem_buffer<cpp_type_in>({ 1, 0, 1 }),
                    1,
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    0,
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    0,
                };
            }
        }
    };

    suite reduce_ops = []
    {
        "min_with_nan_passthru"_test = min_with_nan_passthru_tester{} | SupportedTypes{};
        "max_with_nan_passthru"_test = max_with_nan_passthru_tester{} | SupportedTypes{};

        "reduce_sum_valid"_test = reduce_tests<REDUCE_SUM>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        //badimpl//"reduce_sum_mixed"_test = reduce_tests<REDUCE_SUM>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_sum_invalid"_test = reduce_tests<REDUCE_SUM>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nansum_valid"_test = reduce_tests<REDUCE_NANSUM>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "reduce_nansum_mixed"_test = reduce_tests<REDUCE_NANSUM>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_nansum_invalid"_test = reduce_tests<REDUCE_NANSUM>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_mean_valid"_test = reduce_tests<REDUCE_MEAN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        //badimpl//"reduce_mean_mixed"_test = reduce_tests<REDUCE_MEAN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_mean_invalid"_test = reduce_tests<REDUCE_MEAN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nanmean_valid"_test = reduce_tests<REDUCE_NANMEAN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "reduce_nanmean_mixed"_test = reduce_tests<REDUCE_NANMEAN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_nanmean_invalid"_test = reduce_tests<REDUCE_NANMEAN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_var_valid"_test = reduce_tests<REDUCE_VAR>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        //badimpl//"reduce_var_mixed"_test = reduce_tests<REDUCE_VAR>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_var_invalid"_test = reduce_tests<REDUCE_VAR>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nanvar_valid"_test = reduce_tests<REDUCE_NANVAR>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "reduce_nanvar_mixed"_test = reduce_tests<REDUCE_NANVAR>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_nanvar_invalid"_test = reduce_tests<REDUCE_NANVAR>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_std_valid"_test = reduce_tests<REDUCE_STD>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        //badimpl//"reduce_std_mixed"_test = reduce_tests<REDUCE_STD>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_std_invalid"_test = reduce_tests<REDUCE_STD>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nanstd_valid"_test = reduce_tests<REDUCE_NANSTD>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "reduce_nanstd_mixed"_test = reduce_tests<REDUCE_NANSTD>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_nanstd_invalid"_test = reduce_tests<REDUCE_NANSTD>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_min_valid"_test = reduce_tests<REDUCE_MIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        //badimpl//"reduce_min_mixed"_test = reduce_tests<REDUCE_MIN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_min_invalid"_test = reduce_tests<REDUCE_MIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nanmin_valid"_test = reduce_tests<REDUCE_NANMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "reduce_nanmin_mixed"_test = reduce_tests<REDUCE_NANMIN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_nanmin_invalid"_test = reduce_tests<REDUCE_NANMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_max_valid"_test = reduce_tests<REDUCE_MAX>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        //badimpl//"reduce_max_mixed"_test = reduce_tests<REDUCE_MAX>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_max_invalid"_test = reduce_tests<REDUCE_MAX>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nanmax_valid"_test = reduce_tests<REDUCE_NANMAX>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "reduce_nanmax_mixed"_test = reduce_tests<REDUCE_NANMAX>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_nanmax_invalid"_test = reduce_tests<REDUCE_NANMAX>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_argmin_valid"_test = reduce_tests<REDUCE_ARGMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        //badimpl//"reduce_argmin_mixed"_test = reduce_tests<REDUCE_ARGMIN>::test<test_case_id::MIXED>{} |
        //SupportedTypeCodeTypes{};
        "reduce_argmin_invalid"_test = reduce_tests<REDUCE_ARGMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nanargmin_valid"_test = reduce_tests<REDUCE_NANARGMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "reduce_nanargmin_mixed"_test = reduce_tests<REDUCE_NANARGMIN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "reduce_nanargmin_invalid"_test = reduce_tests<REDUCE_NANARGMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};

        "reduce_nanmean_char"_test = [&]
        {
            std::array<char, 5> test_values{ 'a', 'a', 'a', 'b', 'c' };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_BYTE, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_NANMEAN) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
        };

        "reduce_mean_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, 2.0f, 0.0f, 3.0f, 4.0f };

            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_MEAN) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            expect(no_pyerr() >> fatal);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 2.0_f);
            expect(is_float_type == 1_i);
        };

        "reduce_nanmean_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, 2.0f, NAN, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_NANMEAN) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 2.5_f);
            expect(is_float_type == 1_i);
        };

        "reduce_nanmax"_test = [&]
        {
            std::array<double, 5> test_values{ 3.0, NAN, 2.0, 3.142, -5.0 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_DOUBLE, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_NANMAX) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            double ret_val = PyFloat_AsDouble(retval);
            expect(ret_val == 3.142_d);
            expect(is_float_type == 1_i);
        };

        "reduce_min"_test = [&]
        {
            std::array<double, 5> test_values{ 3.0, -42.5, 2.0, 3.142, -5.0 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_DOUBLE, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_MIN) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            double ret_val = PyFloat_AsDouble(retval);
            expect(ret_val == -42.5_d);
            expect(is_float_type == 1_i);
        };

        "reduce_nanmin"_test = [&]
        {
            std::array<double, 5> test_values{ 3.0, NAN, 2.0, 3.142, -5.0 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_DOUBLE, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_NANMIN) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            double ret_val = PyFloat_AsDouble(retval);
            expect(ret_val == -5.0_d);
            expect(is_float_type == 1_i);
        };

        "reduce_nanmean_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, 2.0f, NAN, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_NANMEAN) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 2.5_f);
            expect(is_float_type == 1_i);
        };

        "reduce_sum"_test = [&]
        {
            std::array<int32_t, 5> test_values{ 1, 2, 42, 3, 4 };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_INT32, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_SUM) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int32_t ret_val = PyLong_AsLong(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 52_i);
            expect(is_float_type == 0_i);
        };

        "reduce_sum_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, 2.0f, 42.5, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_SUM) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 52.5_f);
            expect(is_float_type == 1_i);
        };

        "reduce_nansum_float"_test = [&]
        {
            std::array<float, 5> test_values{ 1.0f, NAN, 42.5, 3.0f, 4.0f };
            npy_intp const dim_len{ 5 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_values.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_NANSUM) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            expect(ret_val == 50.5_f);
            expect(is_float_type == 1_i);
        };

        "reduce_float_long"_test = [&]
        {
            auto const d{ dev() };
            std::vector<float> test_data(65536);
            std::iota(std::begin(test_data), std::end(test_data), -10000.0f);
            std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ d });
            npy_intp const dim_len{ 65536 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_FLOAT, test_data.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_sum{ Py_BuildValue("i", REDUCE_SUM) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_sum, NULL);
            double ret_val = PyFloat_AsDouble(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_sum != nullptr);
            expect(retval != nullptr);
            float x = 1492090880.0f;
            expect(ret_val == x) << "d=" << d << "We got " << ret_val << " but wanted " << x << "\n";
            expect(is_float_type == 1_i);

            PyObject * reduce_fn_mean{ Py_BuildValue("i", REDUCE_MEAN) };
            retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_mean, NULL);
            double calc_avg = PyFloat_AsDouble(retval);
            double actual_avg = x / test_data.size();
            is_float_type = PyFloat_Check(retval);

            expect(reduce_fn_mean != nullptr);
            expect(retval != nullptr);
            expect(calc_avg == actual_avg);
            expect(is_float_type == 1_i);

            PyObject * reduce_fn_var{ Py_BuildValue("i", REDUCE_VAR) };
            retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_var, NULL);
            ret_val = PyFloat_AsDouble(retval);
            is_float_type = PyFloat_Check(retval);

            expect(reduce_fn_var != nullptr);
            expect(retval != nullptr);
            double y = 357919402.0 + 2.0 / 3.0;
            expect(ret_val == y) << "Calculated variance [" << std::setprecision(21) << ret_val << "] expected was [" << y
                                 << "]\n";
            ;
            expect(is_float_type == 1_i);

            PyObject * reduce_fn_std{ Py_BuildValue("i", REDUCE_STD) };
            retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_std, NULL);
            ret_val = PyFloat_AsDouble(retval);
            is_float_type = PyFloat_Check(retval);

            expect(reduce_fn_var != nullptr);
            expect(retval != nullptr);
            y = std::sqrt(y);
            expect(ret_val == y) << "Calculated population standard deviation [" << std::setprecision(21) << ret_val
                                 << "] expected was [" << y << "]\n";
            ;
            expect(is_float_type == 1_i);
        };

        "reduce_sum_int32_long"_test = [&]
        {
            std::vector<int32_t> test_data(65535);
            std::iota(std::begin(test_data), std::end(test_data), -10000);
            std::shuffle(std::begin(test_data), std::end(test_data), std::mt19937{ dev() });
            npy_intp const dim_len{ 65535 };
            PyObject * array{ PyArray_SimpleNewFromData(1, &dim_len, NPY_INT32, test_data.data()) };
            PyObject * function_object = get_named_function(riptide_module_p, "Reduce");
            PyObject * reduce_fn_num{ Py_BuildValue("i", REDUCE_SUM) };
            PyObject * retval = PyObject_CallFunctionObjArgs(function_object, array, reduce_fn_num, NULL);
            int32_t ret_val = PyLong_AsLong(retval);
            int is_float_type = PyFloat_Check(retval);

            expect(array != nullptr);
            expect(function_object != nullptr);
            expect(reduce_fn_num != nullptr);
            expect(retval != nullptr);
            auto const expected{ std::accumulate(std::begin(test_data), std::end(test_data), 0) };
            expect(ret_val == expected) << "expected=" << expected << "actual=" << ret_val;
            expect(is_float_type == 0_i);
        };
    };
}