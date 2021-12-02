// hack for now - headers should be self-inclusive
#include "RipTide.h"
#include <gtest/gtest.h>

PyObject* CompareNumpyMemAddress(PyObject *self, PyObject *args);

TEST(riptide_tests, test_CompareNumpyMemAddress)
{
    PyObject * self{Py_None};
    Py_IncRef(self);

    PyObject * args{Py_None};
    Py_IncRef(args);

    PyObject const * const actual{nullptr/*CompareNumpyMemAddress(self, args)*/};
    EXPECT_NE(nullptr, actual);

    Py_DecRef(self);
    Py_DecRef(args);
}