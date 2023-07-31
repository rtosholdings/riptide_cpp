#include "riptide_python_test.h"

#include "Ema.h"

#include "buffer.h"
#include "mem_buffer.h"
#include "np_util.h"
#include "ut_extensions.h"

#include "boost/ut.hpp"

#include <array>
#include <type_traits>

using namespace riptide_python_test::internal;
using namespace riptide_utility::internal;
using namespace boost::ut;
using boost::ut::suite;

namespace
{
    static constexpr int GB_BASE_INDEX = 1; // comes from RipTide.h, which is too cumbersome to include...

    template <EMA_FUNCTIONS Fn>
    using emafn_to_type = std::integral_constant<EMA_FUNCTIONS, Fn>;

    using SupportedTypeCodeTypes = std::tuple<
        //NOTIMPL: typecode_to_type<NPY_TYPES::NPY_BOOL>,
        typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>, typecode_to_type<NPY_TYPES::NPY_INT32>,
        typecode_to_type<NPY_TYPES::NPY_INT64>, typecode_to_type<NPY_TYPES::NPY_UINT8>, typecode_to_type<NPY_TYPES::NPY_UINT16>,
        typecode_to_type<NPY_TYPES::NPY_UINT32>, typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
        typecode_to_type<NPY_TYPES::NPY_DOUBLE>, typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>>;
}

namespace
{
    constexpr NPY_TYPES get_output_typecode(NPY_TYPES const typecode_in, EMA_FUNCTIONS const fn)
    {
        if (fn == EMA_CUMSUM)
        {
            switch (typecode_in)
            {
            case NPY_INT8:
            case NPY_INT16:
            case NPY_INT32:
            case NPY_INT64:
                return NPY_INT64;

            case NPY_UINT8:
            case NPY_UINT16:
            case NPY_UINT32:
            case NPY_UINT64:
                return NPY_UINT64;

            case NPY_FLOAT:
                return NPY_FLOAT32;
            case NPY_DOUBLE:
                return NPY_FLOAT64;
            case NPY_LONGDOUBLE:
                return NPY_LONGDOUBLE;

            default:
                throw std::runtime_error("Unexpected typecode_in");
            }
        }

        else if (fn == EMA_CUMNANMAX || fn == EMA_CUMMAX || fn == EMA_CUMNANMIN || fn == EMA_CUMMIN)
        {
            return typecode_in;
        }

        throw std::runtime_error("Unexpected fn");
    }

    template <NPY_TYPES TypeCodeIn, EMA_FUNCTIONS EmaFn>
    struct ema_tester
    {
        // Matches test in riptable.
        static_assert(EmaFn >= 300 && EmaFn <= 309);

        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;

        static constexpr auto typecode_out = get_output_typecode(TypeCodeIn, EmaFn);
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static void exec_packed(const_buffer<cpp_type_in> const test_values, const_buffer<cpp_type_out> const expected_values,
                                reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeIn>, emafn_to_type<EmaFn>>;

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const N{ test_values.size() };
            typed_expect<desc_type>(expected_values.size() == N >> fatal) << caller_loc;

            auto const input_array{ pyarray_from_array<TypeCodeIn>(test_values) };
            auto const key_array{ pyarray_from_array<NPY_INT32>(get_same_values<int32_t>(N, GB_BASE_INDEX)) };
            auto const unique_rows{ 1 };
            double const double_param{ 0.0 };
            auto * in_time{ Py_None };
            auto * include_mask{ Py_None };
            auto * reset_mask{ Py_None };

            pyobject_ptr const retval{ PyObject_CallMethod(riptide_module_p, "EmaAll32", "[O]Oii(dOOO)", input_array.get(),
                                                           key_array.get(), unique_rows, EmaFn, double_param, in_time,
                                                           include_mask, reset_mask) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            typed_expect<desc_type>(PyTuple_Check(retval.get()) >> fatal) << caller_loc;
            typed_expect<desc_type>(PyTuple_Size(retval.get()) == 1 >> fatal) << caller_loc;

            pyobject_ptr retarr{ [&retval]
                                 {
                                     auto * const ptr{ PyTuple_GetItem(retval.get(), 0) };
                                     Py_XINCREF(ptr);
                                     return ptr;
                                 }() };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            auto const actual_values{ cast_pyarray_values_as<typecode_out>(&retarr) };
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

    template <EMA_FUNCTIONS>
    struct ema_tests;

    template <EMA_FUNCTIONS GbFn>
    struct ema_tests_base
    {
        using Derived = ema_tests<GbFn>;

        static constexpr EMA_FUNCTIONS gb_fn = GbFn;

        template <NPY_TYPES TypeCode>
        using tester_type = ema_tester<TypeCode, GbFn>;

        template <NPY_TYPES TypeCode>
        struct test_case
        {
            using cpp_type_in = typename tester_type<TypeCode>::cpp_type_in;
            using cpp_type_out = typename tester_type<TypeCode>::cpp_type_out;

            any_const_buffer<cpp_type_in> test_values_;
            any_const_buffer<cpp_type_out> expected_values_;
            reflection::source_location loc_;

            template <template <typename> typename BufferInT, template <typename> typename BufferOutT>
            test_case(BufferInT<cpp_type_in> && test_values, BufferOutT<cpp_type_out> && expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : test_values_{ std::move(test_values) }
                , expected_values_{ std::move(expected_values) }
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
                tester_type<TypeCode>::exec_packed(testcase.test_values_, testcase.expected_values_, testcase.loc_);
            }
        };
    };

    template <>
    struct ema_tests<EMA_CUMSUM> : ema_tests_base<EMA_CUMSUM>
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
                    make_mem_buffer<cpp_type_out>({ 1, 2, 3 }),
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ 0, invalid, invalid }),
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ invalid, invalid, invalid }),
                };
            }
        }
    };

    template <>
    struct ema_tests<EMA_CUMMIN> : ema_tests_base<EMA_CUMMIN>
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
                    make_mem_buffer<cpp_type_out>({ 1, 0, 0 }),
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ 0, invalid, invalid }),
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ invalid, invalid, invalid }),
                };
            }
        }
    };

    template <>
    struct ema_tests<EMA_CUMMAX> : ema_tests_base<EMA_CUMMAX>
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
                    make_mem_buffer<cpp_type_in>({ 0, 1, 1 }),
                    make_mem_buffer<cpp_type_out>({ 0, 1, 1 }),
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ 0, invalid, invalid }),
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ invalid, invalid, invalid }),
                };
            }
        }
    };

    template <>
    struct ema_tests<EMA_CUMNANMIN> : ema_tests_base<EMA_CUMNANMIN>
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
                    make_mem_buffer<cpp_type_out>({ 1, 0, 0 }),
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ 0, 0, 0 }),
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ invalid, invalid, invalid }),
                };
            }
        }
    };

    template <>
    struct ema_tests<EMA_CUMNANMAX> : ema_tests_base<EMA_CUMNANMAX>
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
                    make_mem_buffer<cpp_type_in>({ 0, 1, 1 }),
                    make_mem_buffer<cpp_type_out>({ 0, 1, 1 }),
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                return test_case_type{
                    get_mixed_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ 0, 0, 1 }),
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<cpp_type_in>(3),
                    make_mem_buffer<cpp_type_out>({ invalid, invalid, invalid }),
                };
            }
        }
    };

    suite ema_ops = []
    {
        // TODO: Add all the other tests for >GB_FIRST and <300.
        "ema_cumsum_valid"_test = ema_tests<EMA_CUMSUM>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
#if 0 // fail: not nan-aware yet
        "ema_cumsum_mixed"_test = ema_tests<EMA_CUMSUM>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "ema_cumsum_invalid"_test = ema_tests<EMA_CUMSUM>::test<test_case_id::INVALID>{} | SupportedTypeCodeTypes{};
#endif

        "ema_cummin_valid"_test = ema_tests<EMA_CUMMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "ema_cummin_mixed"_test = ema_tests<EMA_CUMMIN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "ema_cummin_invalid"_test = ema_tests<EMA_CUMMIN>::test<test_case_id::INVALID>{} | SupportedTypeCodeTypes{};

        "ema_cummax_valid"_test = ema_tests<EMA_CUMMAX>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "ema_cummax_mixed"_test = ema_tests<EMA_CUMMAX>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "ema_cummax_invalid"_test = ema_tests<EMA_CUMMAX>::test<test_case_id::INVALID>{} | SupportedTypeCodeTypes{};

        "ema_cumnanmin_valid"_test = ema_tests<EMA_CUMNANMIN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "ema_cumnanmin_mixed"_test = ema_tests<EMA_CUMNANMIN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "ema_cumnanmin_invalid"_test = ema_tests<EMA_CUMNANMIN>::test<test_case_id::INVALID>{} | SupportedTypeCodeTypes{};

        "ema_cumnanmax_valid"_test = ema_tests<EMA_CUMNANMAX>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "ema_cumnanmax_mixed"_test = ema_tests<EMA_CUMNANMAX>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "ema_cumnanmax_invalid"_test = ema_tests<EMA_CUMNANMAX>::test<test_case_id::INVALID>{} | SupportedTypeCodeTypes{};

        "ema_decay_riptable_67"_test = []
        {
            /*
              x = rt.ones(10)
              t = np.arange(10)
              f = rt.FA([True] * 5 + [False] * 4 + [True])
              test = rt.Dataset({
              'x': np.tile(x, reps=2),
              't': np.tile(t, reps=2)
              'f': np.tile(f, reps=2),
              'c': rt.FA(["A"] * 10 + ["B"] * 10)
              })
              test['ema_default'] = test.cat('c').ema_decay(test.x, time=test.t, decay_rate=np.log(2), filter=test.f)
              correct_ema = rt.FA.ema_decay(x, t, np.log(2), filter=f)
              test['ema_correct'] = np.tile(correct_ema, reps=2)
            */
            PyObject * ones{ riptide_python_test::internal::get_named_function(riptable_module_p, "ones") };
            PyObject * func_param{ Py_BuildValue("i", 20) };
            PyObject * x{ PyObject_CallFunctionObjArgs(ones, func_param, NULL) };
            PyObject * arange{ riptide_python_test::internal::get_named_function(riptable_module_p, "arange") };
            PyObject * func_param_10{ Py_BuildValue("i", 10) };
            PyObject * t0{ PyObject_CallFunctionObjArgs(arange, func_param_10, NULL) };
            PyObject * tile{ riptide_python_test::internal::get_named_function(riptable_module_p, "tile") };
            PyObject * reps{ Py_BuildValue("i", 2) };
            PyObject * t{ PyObject_CallFunctionObjArgs(tile, t0, reps, NULL) };
            PyObject * bool_objs{ Py_BuildValue("NNNNNNNNNNNNNNNNNNNN", Py_True, Py_True, Py_True, Py_True, Py_True, Py_False,
                                                Py_False, Py_False, Py_False, Py_True, Py_True, Py_True, Py_True, Py_True, Py_True,
                                                Py_False, Py_False, Py_False, Py_False, Py_True) };
            PyObject * fa{ riptide_python_test::internal::get_named_function(riptable_module_p, "FA") };
            PyObject * f{ PyObject_CallFunctionObjArgs(fa, bool_objs, NULL) };
            PyObject * categories{ Py_BuildValue("CCCCCCCCCCCCCCCCCCCC", 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B',
                                                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B') };
            PyObject * c{ PyObject_CallFunctionObjArgs(fa, categories, NULL) };
            PyObject * test_dict{ Py_BuildValue("{COCOCOCO}", 'x', x, 't', t, 'f', f, 'c', c) };
            PyObject * dataset{ riptide_python_test::internal::get_named_function(riptable_module_p, "Dataset") };
            PyObject * test{ PyObject_CallFunctionObjArgs(dataset, test_dict, NULL) };
            PyObject * return1{ Py_BuildValue("d", 1.0) };
            PyObject * return2{ Py_BuildValue("d", 2.0) };
            PyObject * global_vars{ Py_BuildValue("{sO}", "test", test) };
            PyObject * local_vars{ Py_BuildValue("{sOsO}", "return1", return1, "return2", return2) };

            PyRun_String(
                "test['result']=test.cat('c').ema_decay(test.x, time=test.t, decay_rate=0.693, "
                "filter=test.f);return1=test['result'][4];return2=test['result'][5];",
                Py_file_input, global_vars, local_vars);

            PyObject * return_val1{ PyDict_GetItemString(local_vars, "return1") };
            PyObject * return_val2{ PyDict_GetItemString(local_vars, "return2") };
            double val4{ PyFloat_AsDouble(return_val1) };
            double val5{ PyFloat_AsDouble(return_val2) };

            expect(val4 > 1.0_d);
            expect(val5 < 1.0_d);
        };
    };
}
