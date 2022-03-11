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
        "failing"_test = []
        {
//            expect( 1 == 2_i );
        };
        
        "ema_decay_riptable_67"_test = []
        {
            PyObject * ones{ riptide_python_test::internal::get_named_function(riptable_module_p, "ones") };
            PyObject * func_param{ Py_BuildValue("i", 10) };
            PyObject * x{ PyObject_CallFunctionObjArgs(ones, func_param, NULL) };
            PyObject * arange{ riptide_python_test::internal::get_named_function(riptable_module_p, "arange") };
            PyObject * t{ PyObject_CallFunctionObjArgs(arange, func_param, NULL) };
            PyObject * bool_objs{ Py_BuildValue("OOOOOOOOOO", Py_True, Py_True, Py_True, Py_True, Py_True, Py_False, Py_True, Py_True, Py_True, Py_True) };
            PyObject * fa{ riptide_python_test::internal::get_named_function(riptable_module_p, "FA") };
            PyObject * f{ PyObject_CallFunctionObjArgs(fa, bool_objs, NULL) };
            
            expect(ones != nullptr);
            expect(func_param != nullptr);
            expect(x != nullptr);
            expect(arange != nullptr);
            expect(t != nullptr);
            expect(bool_objs != nullptr);
            expect(fa != nullptr);
            expect(f != nullptr);
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
        };
    };
}
