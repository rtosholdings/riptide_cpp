#include "riptide_python_test.h"

#include "GroupBy.h"
#include "numpy_traits.h"

#include "span.h"
#include "ut_extensions.h"
#include "boost/ut.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace riptide_python_test::internal;
using namespace riptide_utility::internal;
using namespace boost::ut;
using boost::ut::suite;

template <NPY_TYPES TypeCodeIn>
struct typecode_to_type
{
    static constexpr NPY_TYPES value{ TypeCodeIn };
};

template <GB_FUNCTIONS Fn>
struct gbfn_to_type
{
    static constexpr GB_FUNCTIONS value{ Fn };
};

namespace
{
#if 1
    using SupportedTypeCodeTypes = std::tuple<
        //NOTIMPL: typecode_to_type<NPY_TYPES::NPY_BOOL>,
        typecode_to_type<NPY_TYPES::NPY_INT8>, typecode_to_type<NPY_TYPES::NPY_INT16>, typecode_to_type<NPY_TYPES::NPY_INT32>,
        typecode_to_type<NPY_TYPES::NPY_INT64>, typecode_to_type<NPY_TYPES::NPY_UINT8>, typecode_to_type<NPY_TYPES::NPY_UINT16>,
        typecode_to_type<NPY_TYPES::NPY_UINT32>, typecode_to_type<NPY_TYPES::NPY_UINT64>, typecode_to_type<NPY_TYPES::NPY_FLOAT>,
        typecode_to_type<NPY_TYPES::NPY_DOUBLE>, typecode_to_type<NPY_TYPES::NPY_LONGDOUBLE>>;
#else
    using SupportedTypeCodeTypes = std::tuple<typecode_to_type<NPY_TYPES::NPY_FLOAT>>;
#endif

    template <NPY_TYPES TypeCodeIn>
    auto get_mixed_values(size_t const N = 3)
    {
        using cpp_type = riptide::numpy_cpp_type_t<TypeCodeIn>;
        std::vector<cpp_type> arr(N);

        size_t const midpoint{ N / 2 };
        std::fill_n(arr.data(), midpoint, cpp_type{ 0 });
        *(arr.data() + midpoint) = riptide::invalid_for_type<cpp_type>::value;
        std::fill_n(arr.data() + midpoint + 1, N - midpoint - 1, cpp_type{ 1 });

        return arr;
    }

    template <NPY_TYPES TypeCodeIn, typename VT>
    auto get_same_values(size_t const N, VT const V)
    {
        using cpp_type = riptide::numpy_cpp_type_t<TypeCodeIn>;
        std::vector<cpp_type> arr(N, static_cast<cpp_type>(V));
        return arr;
    }

    template <NPY_TYPES TypeCodeIn>
    auto get_zeroes_values(size_t const N)
    {
        return get_same_values<TypeCodeIn>(N, 0);
    }

    template <NPY_TYPES TypeCodeIn>
    auto get_invalid_values(size_t const N)
    {
        using cpp_type = riptide::numpy_cpp_type_t<TypeCodeIn>;
        return get_same_values<TypeCodeIn>(N, riptide::invalid_for_type<cpp_type>::value);
    }

    template <NPY_TYPES TypeCodeIn, typename VT>
    auto get_iota_values(size_t const N, VT const V)
    {
        using cpp_type = riptide::numpy_cpp_type_t<TypeCodeIn>;
        std::vector<cpp_type> arr(N);

        std::iota(arr.data(), arr.data() + N, V);
        return arr;
    }

    template <typename CppType>
    inline CppType get_cpp_value(PyObject * const obj)
    {
        if constexpr (std::is_same_v<CppType, unsigned long long>)
        {
            return PyLong_AsUnsignedLongLong(obj);
        }
        else if constexpr (std::is_same_v<CppType, long long>)
        {
            return PyLong_AsLongLong(obj);
        }
        else
        {
            return PyFloat_AsDouble(obj);
        }
    }

    template <NPY_TYPES TypeCode, typename CppType = riptide::numpy_cpp_type_t<TypeCode>>
    inline span<CppType> get_array_values(pyobject_ptr * const ptr)
    {
        if (! PyArray_Check(ptr->get()))
        {
            PyErr_SetString(PyExc_ValueError, "not a PyArray");
            return span<CppType>{};
        }
        if (PyArray_TYPE(reinterpret_cast<PyArrayObject *>(ptr->get())) != TypeCode)
        {
            PyErr_SetString(PyExc_ValueError, "Not expected typenum");
            return span<CppType>{};
        }

        pyobject_ptr temp{ std::move(*ptr) };

        auto * obj_ptr{ temp.get() };
        CppType * data{ nullptr };
        npy_intp dims{};

        pyobject_any_ptr<PyArray_Descr> desc_ptr{ PyArray_DescrNewFromType(TypeCode) };

        auto const retval{ PyArray_AsCArray(&obj_ptr, &data, &dims, 1, desc_ptr.get()) };
        if (retval < 0)
        {
            return span<CppType>{};
        }

        desc_ptr.release();
        temp.release();
        ptr->reset(obj_ptr);

        span<CppType> result{ data, static_cast<size_t>(dims) };
        return result;
    }

    constexpr NPY_TYPES get_output_typecode(NPY_TYPES const typecode_in, GB_FUNCTIONS const fn)
    {
        if (fn == GB_FUNCTIONS::GB_ROLLING_MEAN)
        {
            return NPY_FLOAT64;
        }
        else
        {
            return NPY_FLOAT64;
        }
    }

    template <NPY_TYPES TypeCodeIn, typename CppType = riptide::numpy_cpp_type_t<TypeCodeIn>, typename Container>
    pyobject_ptr pyarray_from_array(Container const & values)
    {
        auto const dim_len{ static_cast<npy_intp>(values.size()) };
        pyobject_ptr input_array{ PyArray_SimpleNewFromData(1, &dim_len, TypeCodeIn,
                                                            const_cast<void *>(static_cast<void const *>(values.data()))) };
        pyobject_ptr result_array{ PyArray_SimpleNew(1, &dim_len, TypeCodeIn) };
        PyArray_CopyInto(reinterpret_cast<PyArrayObject *>(result_array.get()),
                         reinterpret_cast<PyArrayObject *>(input_array.get()));
        return std::move(result_array);
    }

    template <NPY_TYPES TypeCodeIn, GB_FUNCTIONS GroupByFn>
    struct groupby_tester
    {
        // Matches test in riptable.
        static_assert(GroupByFn >= GB_FUNCTIONS::GB_FIRST && GroupByFn < 300);

        using cpp_type_in = riptide::numpy_cpp_type_t<TypeCodeIn>;

        static constexpr auto typecode_out = get_output_typecode(TypeCodeIn, GroupByFn);
        using cpp_type_out = riptide::numpy_cpp_type_t<typecode_out>;

        static cpp_type_out as_cpp_type_out(cpp_type_in const input)
        {
            return riptide::cast_nan_aware<cpp_type_out>(input);
        }

        static void packed(std::vector<cpp_type_in> const & test_values, std::vector<cpp_type_out> const & expected_values,
                           reflection::source_location const & loc = reflection::source_location::current())
        {
            using desc_type = std::tuple<typecode_to_type<TypeCodeIn>, gbfn_to_type<GroupByFn>>;

            auto const caller_loc{ [&loc]
                                   {
                                       std::ostringstream stream;
                                       stream << "; caller: " << loc.file_name() << ':' << loc.line();
                                       return stream.str();
                                   }() };

            auto const N{ test_values.size() };
            typed_expect<desc_type>(expected_values.size() == N >> fatal) << caller_loc;

            auto const input_array{ pyarray_from_array<TypeCodeIn>(test_values) };
            auto const key_array{ pyarray_from_array<NPY_INT32>(get_zeroes_values<NPY_INT32>(N)) };
            auto const group_array{ pyarray_from_array<NPY_INT32>(get_iota_values<NPY_INT32>(N, 0)) };
            auto const uniques_array{ pyarray_from_array<NPY_INT32>(get_zeroes_values<NPY_INT32>(2)) };
            auto const uniques_counts_array{ pyarray_from_array<NPY_INT32>(get_same_values<NPY_INT32>(2, N)) };
            auto const unique_rows{ 2 };
            auto const bin_low{ 0 };
            auto const bin_high{ 2 };
            auto const window_size{ 1 };

            pyobject_ptr const retval{ PyObject_CallMethod(
                riptide_module_p, "GroupByAllPack32", "[O]OOOOi[i][i][i]i", input_array.get(), key_array.get(), group_array.get(),
                uniques_array.get(), uniques_counts_array.get(), unique_rows, GroupByFn, bin_low, bin_high, window_size) };
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

            auto const actual_values{ get_array_values<typecode_out>(&retarr) };
            typed_expect<desc_type>(no_pyerr() >> fatal) << caller_loc;

            typed_expect<desc_type>(actual_values.size() == expected_values.size() >> fatal) << caller_loc;

            for (size_t i{ 0 }; i < expected_values.size(); ++i)
            {
                auto const expected_val{ expected_values[i] };
                auto const actual_val{ actual_values[i] };
                typed_expect<desc_type>(equal_to_nan_aware(actual_val, expected_val))
                    << "index:" << i << ", expected:" << expected_val << ", actual:" << actual_val << caller_loc;
            }
        }
    };

    enum class test_case_id
    {
        VALID,
        MIXED,
        INVALID,
    };

    template <GB_FUNCTIONS>
    struct groupby_tests;

    template <GB_FUNCTIONS GbFn>
    struct groupby_tests_base
    {
        using Derived = groupby_tests<GbFn>;

        static constexpr GB_FUNCTIONS gb_fn = GbFn;

        template <NPY_TYPES TypeCode>
        struct test_case
        {
            using cpp_type_in = typename groupby_tester<TypeCode, GbFn>::cpp_type_in;
            using cpp_type_out = typename groupby_tester<TypeCode, GbFn>::cpp_type_out;

            std::vector<cpp_type_in> test_values_;
            std::vector<cpp_type_out> expected_values_;
            reflection::source_location loc_;

            test_case(std::vector<cpp_type_in> const & test_values, std::vector<cpp_type_out> expected_values,
                      reflection::source_location const & loc = reflection::source_location::current())
                : test_values_{ test_values }
                , expected_values_{ expected_values }
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
                using tester_type = groupby_tester<TypeCode, GbFn>;
                tester_type::packed(testcase.test_values_, testcase.expected_values_, testcase.loc_);
            }
        };
    };

    template <>
    struct groupby_tests<GB_ROLLING_MEAN> : groupby_tests_base<GB_ROLLING_MEAN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    get_same_values<TypeCode>(3, 1),
                    std::vector<cpp_type_out>{ 1, 1, 1 },
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<TypeCode>(3),
                    std::vector<cpp_type_out>{ 0, invalid, invalid },
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<TypeCode>(3),
                    std::vector<cpp_type_out>{ invalid, invalid, invalid },
                };
            }
        }
    };

    template <>
    struct groupby_tests<GB_ROLLING_NANMEAN> : groupby_tests_base<GB_ROLLING_NANMEAN>
    {
        template <test_case_id Id, NPY_TYPES TypeCode>
        static auto get_test_case()
        {
            using test_case_type = test_case<TypeCode>;
            using cpp_type_out = typename test_case_type::cpp_type_out;

            if constexpr (Id == test_case_id::VALID)
            {
                return test_case_type{
                    get_same_values<TypeCode>(3, 1),
                    std::vector<cpp_type_out>{ 1, 1, 1 },
                };
            }

            else if constexpr (Id == test_case_id::MIXED)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_mixed_values<TypeCode>(3),
                    std::vector<cpp_type_out>{ 0, invalid, 1 },
                };
            }

            else if constexpr (Id == test_case_id::INVALID)
            {
                constexpr auto invalid{ riptide::invalid_for_type<cpp_type_out>::value };
                return test_case_type{
                    get_invalid_values<TypeCode>(3),
                    std::vector<cpp_type_out>{ invalid, invalid, invalid },
                };
            }
        }
    };

    suite groupby_ops = []
    {
        // TODO: Add all the other tests for >GB_FIRST and <300.

        "groupby_accum_rolling_mean_valid"_test =
            groupby_tests<GB_ROLLING_MEAN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
#if 0 // fail: not nan-aware yet
        "groupby_accum_rolling_mean_mixed"_test =
            groupby_tests<GB_ROLLING_MEAN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "groupby_accum_rolling_mean_invalid"_test =
            groupby_tests<GB_ROLLING_MEAN>::test<test_case_id::INVALID>{} | SupportedTypeCodeTypes{};
#endif
        "groupby_accum_rolling_nanmean_valid"_test =
            groupby_tests<GB_ROLLING_NANMEAN>::test<test_case_id::VALID>{} | SupportedTypeCodeTypes{};
        "groupby_accum_rolling_nanmean_mixed"_test =
            groupby_tests<GB_ROLLING_NANMEAN>::test<test_case_id::MIXED>{} | SupportedTypeCodeTypes{};
        "groupby_accum_rolling_nanmean_invalid"_test =
            groupby_tests<GB_ROLLING_NANMEAN>::test<test_case_id::INVALID>{} | SupportedTypeCodeTypes{};
    };
}
