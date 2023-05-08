#pragma once
#include <string_view>

// From https://stackoverflow.com/a/56766138

template <typename T>
constexpr auto to_type_str()
{
    std::string_view name, prefix, suffix;
#ifdef __clang__
    name = __PRETTY_FUNCTION__;
    prefix = "auto to_type_str() [T = ";
    suffix = "]";
#elif defined(__GNUC__)
    name = __PRETTY_FUNCTION__;
    prefix = "constexpr auto to_type_str() [with T = ";
    suffix = "]";
#elif defined(_MSC_VER)
    name = __FUNCSIG__;
    prefix = "auto __cdecl to_type_str<";
    suffix = ">(void)";
#else
    #error "Unrecognized compiler"
#endif
    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}
