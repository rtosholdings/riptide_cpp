#include "boost/ut.hpp"

namespace
{
    struct options
    {
        bool list_tests{ false };
        std::string test_filter{};
    };

    options parse_options(int const argc, char const ** const argv)
    {
        options options;

        for (int ac{ 1 }; ac < argc; ++ac)
        {
            std::string_view const arg{ argv[ac] };

            if (arg == "--ut_list_tests")
            {
                options.list_tests = true;
            }

            else if (arg == "--ut_filter")
            {
                if (ac == argc - 1)
                {
                    fprintf(stderr, "Fatal error: missing filter argument");
                    exit(1);
                }

                options.test_filter = argv[++ac];
            }

            else
            {
                fprintf(stderr, "Fatal error: unrecognized option, %s\n", std::string(arg).c_str());
                exit(1);
            }
        }

        return options;
    }
}

int main(int argc, char const ** argv)
{
    auto const options{ parse_options(argc, argv) };

    boost::ut::options ut_options;
    if (options.list_tests)
    {
        ut_options.dry_run = true;
    }
    if (! options.test_filter.empty())
    {
        ut_options.filter = options.test_filter;
    }
    boost::ut::cfg<> = ut_options;

    auto result{ boost::ut::cfg<>.run() };

    return result;
}
