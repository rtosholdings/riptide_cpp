#include "riptide_python_test.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <array>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    suite ema_ops = []
    {
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
            PyObject * bool_objs{ Py_BuildValue("NNNNNNNNNNNNNNNNNNNN", Py_True, Py_True, Py_True, Py_True, Py_True, Py_False, Py_True, Py_True, Py_True, Py_True, Py_True, Py_True, Py_True, Py_True, Py_True, Py_False, Py_True, Py_True, Py_True, Py_True) };
            PyObject * fa{ riptide_python_test::internal::get_named_function(riptable_module_p, "FA") };
            PyObject * f{ PyObject_CallFunctionObjArgs(fa, bool_objs, NULL) };
            PyObject * categories{ Py_BuildValue("CCCCCCCCCCCCCCCCCCCC", 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B') };
            PyObject * c{ PyObject_CallFunctionObjArgs(fa, categories, NULL ) };
            PyObject * test_dict{ Py_BuildValue("{COCOCOCO}", 'x', x, 't', t, 'f', f, 'c', c) };
            PyObject * dataset{ riptide_python_test::internal::get_named_function(riptable_module_p, "Dataset") };
            PyObject * test{ PyObject_CallFunctionObjArgs(dataset, test_dict, NULL ) };
            PyObject * arg_c{ Py_BuildValue("C", 'c') };
            PyObject * cat_name{ Py_BuildValue("s", "cat") };
            PyObject * cat{ PyObject_CallMethodObjArgs( test, cat_name, arg_c, NULL ) };
            PyObject * ema_decay_name{ Py_BuildValue("s", "ema_decay") };

            expect(cat != nullptr);
        };
    };
}
