#ifndef RIPTABLECPP_OVERLOADED_H
#define RIPTABLECPP_OVERLOADED_H

namespace internal
{
    template <typename... Ts>
    struct overloaded : Ts...
    {
        using Ts::operator()...;
    };

    template <typename... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;
} // namespace internal

#endif
