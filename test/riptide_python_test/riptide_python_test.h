#ifndef RIPTIDE_PYTHON_TEST_H
#define RIPTIDE_PYTHON_TEST_H

// Undo the damage we're about to do by undefining a reserved macro name
#if defined(_MSC_VER) && defined(_DEBUG) && _MSC_VER >= 1930
    #include <corecrt.h>
#endif

// Hack because debug builds force python36_d.lib
#define MS_NO_COREDLL    // don't add import libs by default
#define Py_ENABLE_SHARED // but do enable shared libs

#include <pyconfig.h>
#undef Py_DEBUG // don't use debug Python APIs

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//#define PY_ARRAY_UNIQUE_SYMBOL riptide_python_test_global
#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API
#ifndef PYTHON_TEST_MAIN
    #define NO_IMPORT_ARRAY (1)
#else
    #undef NO_IMPORT
    #undef NO_IMPORT_ARRAY
#endif

#include "missing_values.h"
#include "numpy_traits.h"

#include "buffer.h"
#include "mem_buffer.h"
#include "any_const_buffer.h"

#include <numpy/arrayobject.h>

#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <variant>

namespace riptide_python_test::internal
{
    extern PyObject * get_named_function(PyObject * module_p, char const * name_p);

    extern void pyobject_printer(PyObject * printable);

    extern bool no_pyerr(bool print = true);
}

namespace riptide_python_test::internal
{
    namespace details
    {
        template <typename PyT>
        struct pyobject_deleter
        {
            void operator()(PyT * obj) const
            {
                Py_XDECREF(obj);
            }
        };
    }

    template <typename PyT>
    using pyobject_any_ptr = std::unique_ptr<PyT, details::pyobject_deleter<PyT>>;

    using pyobject_ptr = pyobject_any_ptr<PyObject>;
}

namespace riptide_python_test::internal
{
    template <typename T>
    auto to_out(T && t)
    {
        return std::forward<T>(t);
    }

    inline auto to_out(char const t)
    {
        return static_cast<int>(t);
    }

    inline auto to_out(signed char const t)
    {
        return static_cast<int>(t);
    }

    inline auto to_out(unsigned char const t)
    {
        return static_cast<unsigned int>(t);
    }
}

namespace riptide_python_test::internal
{
    template <typename T>
    struct equal_within
    {
        T const tolerance_{};

        bool operator()(T const & x, T const & y) const
        {
            T const delta{ x - y };
            T const abs_delta{ delta < 0 ? -delta : delta };
            return abs_delta < tolerance_;
        }
    };

    template <typename T, typename Predicate = std::equal_to<T>>
    constexpr bool equal_to_nan_aware(T const & x, T const & y, Predicate const pred = {})
    {
        using invalid_for_type = riptide::invalid_for_type<T>;

        auto const x_valid{ invalid_for_type::is_valid(x) };
        auto const y_valid{ invalid_for_type::is_valid(y) };

        if (x_valid ^ y_valid)
        {
            return false;
        }
        return ! x_valid || pred(x, y);
    }
}

namespace riptide_python_test::internal
{
    template <NPY_TYPES TypeCode, typename Container>
    pyobject_ptr pyarray_from_array(Container const & data)
    {
        using cpp_type = riptide::numpy_cpp_type_t<TypeCode>;
        using storage_type = riptide::numpy_c_type_t<TypeCode>;
        static_assert(sizeof(cpp_type) == sizeof(storage_type));

        auto const * const data_array{ data.data() };
        auto const data_size{ data.size() };
        using data_type = std::decay_t<decltype(*data_array)>;

        static_assert(std::is_same_v<data_type, cpp_type> || std::is_same_v<data_type, storage_type>);

        auto const dim_len{ static_cast<npy_intp>(data_size) };
        pyobject_ptr result_array{ PyArray_SimpleNew(1, &dim_len, TypeCode) };
        if (! result_array)
        {
            return {};
        }

        auto * const storage_array{ reinterpret_cast<storage_type *>(
            PyArray_BYTES(reinterpret_cast<PyArrayObject *>(result_array.get()))) };

        if constexpr (std::is_same_v<data_type, storage_type>)
        {
            std::memcpy(storage_array, data_array, data_size * sizeof(storage_type));
        }

        else
        {
            std::copy(data_array, data_array + data_size, storage_array);
        }

        return result_array;
    }
}

namespace riptide_python_test::internal
{
    template <NPY_TYPES TypeCode, typename CppType = riptide::numpy_cpp_type_t<TypeCode>>
    riptide_utility::internal::const_buffer<CppType> cast_pyarray_values_as(pyobject_ptr * const ptr)
    {
        using result_t = riptide_utility::internal::const_buffer<CppType>;

        if (! PyArray_Check(ptr->get()))
        {
            PyErr_SetString(PyExc_ValueError, "not a PyArray");
            return result_t{};
        }
        if (PyArray_TYPE(reinterpret_cast<PyArrayObject *>(ptr->get())) != TypeCode)
        {
            PyErr_SetString(PyExc_ValueError, "Not expected typenum");
            return result_t{};
        }

        pyobject_ptr temp{ std::move(*ptr) };

        auto * obj_ptr{ temp.get() };
        CppType * data{ nullptr };
        npy_intp dims{};

        pyobject_any_ptr<PyArray_Descr> desc_ptr{ PyArray_DescrNewFromType(TypeCode) };

        auto const retval{ PyArray_AsCArray(&obj_ptr, &data, &dims, 1, desc_ptr.get()) };
        if (retval < 0)
        {
            return result_t{};
        }

        desc_ptr.release();
        temp.release();
        ptr->reset(obj_ptr);

        result_t result{ data, static_cast<size_t>(dims) };
        return result;
    }
}

namespace riptide_python_test::internal
{
    template <typename CppType>
    auto get_mixed_values(size_t const N = 3)
    {
        riptide_utility::internal::mem_buffer<CppType> arr(N);
        size_t const midpoint{ N / 2 };
        std::fill_n(arr.data(), midpoint, CppType{ 0 });
        *(arr.data() + midpoint) = riptide::invalid_for_type<CppType>::value;
        std::fill_n(arr.data() + midpoint + 1, N - midpoint - 1, CppType{ 1 });
        return arr;
    }

    template <typename CppType, typename VT>
    auto get_same_values(size_t const N, VT const V)
    {
        riptide_utility::internal::mem_buffer<CppType> arr(N);
        std::fill_n(arr.data(), N, static_cast<CppType>(V));
        return arr;
    }

    template <typename CppType>
    auto get_zeroes_values(size_t const N)
    {
        return get_same_values<CppType>(N, 0);
    }

    template <typename CppType>
    auto get_invalid_values(size_t const N)
    {
        return get_same_values<CppType>(N, riptide::invalid_for_type<CppType>::value);
    }

    template <typename CppType, typename VT>
    auto get_iota_values(size_t const N, VT const V)
    {
        riptide_utility::internal::mem_buffer<CppType> arr(N);
        std::iota(arr.data(), arr.data() + N, V);
        return arr;
    }
}

extern PyObject * riptide_module_p;
extern PyObject * riptable_module_p;

enum struct hash_choice_t
{
    hash_linear,
    tbb,
};

inline hash_choice_t runtime_hash_choice;
#endif
