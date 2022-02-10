#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL sharedata_ARRAY_API
#include "numpy/arrayobject.h"

#include "Python.h"

#include <iostream>

int main()
{
    std::cout << "In main\n";
    Py_Initialize();
    import_array();
}
