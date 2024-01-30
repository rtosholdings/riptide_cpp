#include "ut_core.h"

int main(int argc, char const ** argv)
{
    auto const ut_options{ riptide_utility::ut::parse_options(argc, argv) };

    boost::ut::cfg<boost::ut::override> = ut_options;

    auto result{ boost::ut::cfg<boost::ut::override>.run() };

    return result;
}
