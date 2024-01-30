#pragma once

#include "ut_core.h"

#include <sstream>
#include <string_view>

/// @brief Decorates any failed expect's with the specified type and optional description
template <typename Type, typename TExpr>
auto typed_expect(std::string_view const desc, TExpr const & expr,
                  boost::ut::reflection::source_location const & sloc = boost::ut::reflection::source_location::current())
{
    using namespace boost::ut;

    // A fatal assertion will throw an exception when evaluating expr,
    // bypassing any message insertion done afterwards.
    // So we catch that exception, insert our message, then throw an
    // unexpected exception to stop the test.
    // It's clumsy (2 test failure messages and an unexcepted exception message)
    // but gets the job done.
    bool ok{ false };
    bool exception{ false };
    try
    {
        ok = expr;
    }
    catch (...)
    {
        exception = true;
    }
    std::ostringstream msg;
    if (! ok)
    {
        msg << "(Type:" << reflection::type_name<Type>();
        if (! desc.empty())
        {
            msg << "; " << desc;
        }
        msg << ')';
    }
    auto result{ expect(ok, sloc) << (exception ? "*fatal*" : "") << msg.str() };
    if (exception)
    {
        throw std::runtime_error("stopping test due to fatal failure");
    }
    return result;
}

/// @brief Decorates any failed expect's with the specified type
template <typename Type, typename TExpr>
auto typed_expect(TExpr const & expr,
                  boost::ut::reflection::source_location const & sloc = boost::ut::reflection::source_location::current())
{
    return typed_expect<Type, TExpr>(std::string_view{}, expr, sloc);
}