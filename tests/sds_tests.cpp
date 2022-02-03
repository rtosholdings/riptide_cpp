// hack for now - headers should be self-inclusive
#include "RipTide.h"
#include "SDSFile.h"
#include <gtest/gtest.h>

TEST(sds_tests, test_SDSGetLastError)
{
    char const * actual{SDSGetLastError()};
    EXPECT_NE(nullptr, actual);
    EXPECT_EQ(0, actual[0]);
}
