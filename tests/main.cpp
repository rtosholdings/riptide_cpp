#include <gtest/gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // This lets c++ exceptions and windows SEH (i.e. null pointer/segv)
    // ripple up so we can write tests to document when we expect this
    // to happen.
    ::testing::GTEST_FLAG(catch_exceptions) = false;
    auto const ret{RUN_ALL_TESTS()};
    return ret;
}