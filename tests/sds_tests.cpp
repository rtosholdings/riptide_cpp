// hack for now - headers should be self-inclusive
#include "src/RipTide.h"
#include "src/SDSFile.h"
#include <gtest/gtest.h>

TEST(sds_tests, test_SDSGetLastError)
{
    char const * actual{SDSGetLastError()};
    EXPECT_NE(nullptr, actual);
    EXPECT_EQ(0, actual[0]);
}