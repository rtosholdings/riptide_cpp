#pragma once

#include "boost/ut.hpp"

#include <iostream>
#include <filesystem>
#include <set>
#include <source_location>
#include <string>

namespace riptide_utility::ut::details
{
    /// Sanitize the name to make suitable for gtest purposes
    inline std::string sanitize_name(std::string_view const name)
    {
        std::string result{ name };
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c)
                       {
                           return (c == '_' || c == '.' || std::isalnum(c)) ? c : '_';
                       });
        return result;
    }

    class test_matcher
    {
    public:
        test_matcher() = default;

        test_matcher(std::string_view const input_pattern)
            : full_pattern_{ input_pattern }
        {
            std::string_view const pattern{ full_pattern_ };
            std::string_view::size_type start{ pattern.empty() ? pattern.npos : 0 };
            while (start != pattern.npos)
            {
                auto const next{ pattern.find_first_of(':', start) };
                offsets_.emplace_back(offsets_type{ start, next });
                start = next == pattern.npos ? next : next + 1;
            }
        }

        [[nodiscard]] bool operator()(std::string_view const input) const
        {
            std::string_view const full_pattern{ full_pattern_ };
            for (auto const & offset : offsets_)
            {
                auto const pattern{ full_pattern.substr(offset.first, offset.second) };
                if (boost::ut::utility::is_match(input, pattern))
                {
                    return true;
                }
            }
            return false;
        }

    private:
        using offsets_type = std::pair<std::string_view::size_type, std::string_view::size_type>;

        std::string full_pattern_;
        std::vector<offsets_type> offsets_{};
    };

    class test_filter
    {
    public:
        test_filter() = default;

        test_filter(std::string_view const filter_str)
        {
            auto const sep{ filter_str.find_first_of('-') };
            auto const positive_str = filter_str.substr(0, sep);
            if (! positive_str.empty())
            {
                positive_matcher_ = test_matcher{ positive_str };
            }
            if (sep != filter_str.npos)
            {
                negative_matcher_ = test_matcher{ filter_str.substr(sep + 1) };
            }
        }

        [[nodiscard]] bool operator()(std::string_view const input) const
        {
            return positive_matcher_(input) && ! negative_matcher_(input);
        }

    private:
        test_matcher positive_matcher_{ "*" };
        test_matcher negative_matcher_{};
    };
}

namespace cfg
{
    using base_reporter_type = boost::ut::reporter<boost::ut::printer>;

    /// Reporter that maintains active suite name.
    struct gtest_reporter : base_reporter_type
    {
        static inline char const * const global_suite_{ "global" };

        using base_reporter_type::on;
        using base_reporter_type::operator=;

        void on(boost::ut::events::suite_begin suite)
        {
            active_suite_ = riptide_utility::ut::details::sanitize_name(suite.name);
        }

        void on(boost::ut::events::suite_end suite)
        {
            active_suite_ = global_suite_;
        }

        void on(boost::ut::events::test_begin test)
        {
            std::string test_fqn{ active_suite_ };
            test_fqn += '.';
            test_fqn += test.name;

            if (active_test_ != test_fqn)
            {
                active_test_ = test_fqn;
                std::cout << "Running " << test_fqn << "...\n";
            }

            base_reporter_type::on(test);
        }

        std::string active_suite_{ global_suite_ };
        std::string active_test_{};
    };

    using base_runner_type = boost::ut::runner<gtest_reporter>;

    /// Runner that supports gtest queries and test fully-qualified name.
    struct gtest_runner : base_runner_type
    {
        using base_runner_type::on;

        void operator=(boost::ut::options options)
        {
            filter_ = options.filter;
            options.filter = {};                 // disable ut filtering
            options.colors = { "", "", "", "" }; // disable colors

            base_runner_type::operator=(options);
        }

        template <class... Ts>
        auto on(boost::ut::events::test<Ts...> test)
        {
            std::string test_fqn{ reporter_.active_suite_ };
            test_fqn += '.';
            test_fqn += test.name;

            if (! filter_(test_fqn))
            {
                return;
            }

            // If dry_run, we are simply querying all test names.
            // Emit the suite-based full name and stop further processing.
            if (dry_run_)
            {
                if (reporter_.active_suite_ != cur_suite_)
                {
                    cur_suite_ = reporter_.active_suite_;
                    std::cout << cur_suite_ << ".\n";
                }

                auto const result{ found_names_.emplace(test.name) };
                if (result.second)
                {
                    std::cout << "  " << test.name << '\n';
                }
                return;
            }

            base_runner_type::on(test);
        }

        std::set<std::string> found_names_;
        std::string cur_suite_{};
        riptide_utility::ut::details::test_filter filter_{};
    };
}

template <class... Ts>
inline auto boost::ut::cfg<boost::ut::override, Ts...> = cfg::gtest_runner{};

namespace riptide_utility::ut
{
    /// Suite that names itself as "<parent-name>.<file-stem>".
    struct file_suite
    {
        std::source_location const location;
        std::string const name;

        template <class TSuite>
        file_suite(TSuite _suite, std::source_location loc = std::source_location::current())
            : location{ loc }
            , name{ to_name(loc) }
        {
            static_assert(1 == sizeof(_suite));
            boost::ut::detail::on<decltype(+_suite)>(boost::ut::events::suite<decltype(+_suite)>{ .run = +_suite, .name = name });
        }

        [[nodiscard]] static std::string to_name(std::source_location const & loc)
        {
            auto const path{ std::filesystem::path{ loc.file_name() } };
            auto const parents_str{ path.parent_path().generic_string() };
            auto const parent_pos{ parents_str.find_last_of('/') };
            auto const parent_str{ parent_pos == std::string::npos ? parents_str : parents_str.substr(parent_pos + 1) };
            auto const parent_name{ details::sanitize_name(parent_str) };
            auto const stem{ path.stem() };
            auto const stem_name{ details::sanitize_name(stem.generic_string()) };
            return parent_name + "." + stem_name;
        }
    };

    /// Parse command-line options for ut options and gest options emulation.
    [[nodiscard]] inline boost::ut::options parse_options(int const argc, char const ** const argv)
    {
        boost::ut::options options;

        for (int ac{ 1 }; ac < argc; ++ac)
        {
            std::string_view const arg{ argv[ac] };

            if (arg == "--ut_list_tests" || arg == "--gtest_list_tests")
            {
                options.dry_run = true;
            }

            else if (arg == "--ut_filter")
            {
                if (ac == argc - 1)
                {
                    std::cerr << "Fatal error: missing filter argument\n";
                    exit(1);
                }

                options.filter = argv[++ac];
            }

            else if (arg.starts_with("--gtest_filter="))
            {
                options.filter = arg.substr(15);
            }

            else if (arg.starts_with("--gtest"))
            {
                // ignore any gtest options we don't recognize
            }

            else
            {
                std::cerr << "Fatal error: unrecognized option, " << arg << '\n';
                exit(1);
            }
        }

        return options;
    }
}