#pragma once

#include "buffer.h"

#include <algorithm>
#include <memory>
#include <memory_resource>

namespace riptide_utility::internal
{
    namespace details
    {
        struct pmr_memory_deleter
        {
            std::pmr::memory_resource * allocator_{};
            size_t size_{};

            void operator()(void * const ptr) const
            {
                allocator_->deallocate(ptr, size_);
            }
        };
    }

    template <typename T>
    using pmr_memory_pointer = std::unique_ptr<T, details::pmr_memory_deleter>;

    template <typename T>
    inline pmr_memory_pointer<T> allocate_pmr_memory(std::pmr::memory_resource * const allocator, size_t const size)
    {
        auto const size_bytes{ size * sizeof(T) };
        return pmr_memory_pointer<T>{ static_cast<T *>(allocator->allocate(size_bytes)),
                                      details::pmr_memory_deleter{ allocator, size_bytes } };
    }

    inline std::pmr::memory_resource * get_memory_resource_or_default(std::pmr::memory_resource * const res) noexcept
    {
        return res ? res : std::pmr::get_default_resource();
    }

    template <typename T>
    class mem_buffer
    {
    public:
        mem_buffer() noexcept = default;

        explicit mem_buffer(size_t const size, std::pmr::memory_resource * const resource = nullptr)
            : storage_{ allocate_pmr_memory<T>(get_memory_resource_or_default(resource), size) }
        {
        }

        [[nodiscard]] T const * data() const noexcept
        {
            return storage_.get();
        }

        [[nodiscard]] T * data() noexcept
        {
            return storage_.get();
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return size_bytes() / sizeof(T);
        }

        [[nodiscard]] size_t size_bytes() const noexcept
        {
            return storage_.get_deleter().size_;
        }

    private:
        pmr_memory_pointer<T> storage_{};
    };

    /// Creates mem_buffer from an array.
    template <typename T, size_t N>
    [[nodiscard]] auto make_mem_buffer(T const (&arr)[N])
    {
        mem_buffer<T> result(N);
        std::copy_n(arr, N, result.data());
        return result;
    }

    /// Creates mem_buffer from an initializer_list.
    template <typename T>
    [[nodiscard]] auto make_mem_buffer(std::initializer_list<T> const & init)
    {
        mem_buffer<T> result(init.size());
        std::copy(init.begin(), init.end(), result.data());
        return result;
    }
}
