#include <vector>

namespace riptide_utility::internal
{
    template <typename T, typename... Args>
    std::vector<T> make_vector(Args &&... args)
    {
        std::vector<T> vec;
        vec.reserve(sizeof...(Args));
        (vec.emplace_back(std::forward<Args>(args)), ...);
        return vec;
    }
}