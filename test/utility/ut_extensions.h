#pragma once

#include "boost/ut.hpp"

/// @brief Decorates any failed expect's with the specified type
template <typename Type, typename TExpr>
constexpr auto
typed_expect(TExpr const & expr,
             boost::ut::reflection::source_location const & sloc = boost::ut::reflection::source_location::current())
{
    return boost::ut::expect(expr, sloc) << "(Type:" << boost::ut::reflection::type_name<Type>() << ')';
}