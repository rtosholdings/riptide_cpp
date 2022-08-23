#ifndef RIPTIDECPP_OVERLOAD_H
#define RIPTIDECPP_OVERLOAD_H

template <class... Fs>
struct overload : Fs...
{
    template <class... Ts>
    overload(Ts &&... ts)
        : Fs{ std::forward<Ts>(ts) }...
    {
    }

    using Fs::operator()...;
};

template <class... Ts>
overload(Ts &&...) -> overload<std::remove_reference_t<Ts>...>;

#endif
