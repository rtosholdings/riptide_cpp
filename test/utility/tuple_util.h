#pragma once

#include <tuple>

namespace riptide_utility::internal
{
    // Cartesian product of two tuples.
    // Adapted from https://stackoverflow.com/a/70405807.
    auto tuple_prod(auto ts0, auto ts1)
    {
        return std::apply(
            [&](auto... t0s)
            {
                return std::tuple_cat([&](auto t0) { // glue (t0) x (t1s...) together
                    return std::apply(
                        [&](auto... t1s)
                        {
                            return std::make_tuple(std::make_tuple(t0, t1s)...); // ((t0,t10),(t0,t11),...)
                        },
                        ts1); // turning ts1 into t1s...
                }(t0s)...);   // "laundering" the t0s... into t0s non-pack, then gluing
            },
            ts0); // turning ts0 into t0s...
    }
}