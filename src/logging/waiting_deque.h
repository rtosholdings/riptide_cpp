#include <thread>
#include <deque>

namespace riptide::logging::details
{
    template <class T, class Allocator = std::allocator<T>>
    class waiting_deque
    {
    public:
        std::optional<T> pop_front()
        {
            std::unique_lock lk{ lock_ };

            notify_.wait(lk,
                         [&]
                         {
                             return ! deque_.empty() or ! wait_;
                         });

            if (deque_.empty())
            {
                wait_ = true;
                return std::nullopt;
            }

            std::optional<T> ret{ std::move(deque_.front()) };
            deque_.pop_front();
            return ret;
        }

        void push_back(T && t)
        {
            {
                std::lock_guard guard{ lock_ };

                if (deque_.size() >= max_size_)
                    return;

                deque_.push_back(std::forward<T>(t));
            }
            notify_.notify_one();
        }

        bool empty()
        {
            std::lock_guard guard{ lock_ };
            return deque_.empty();
        }

        void clear()
        {
            std::lock_guard guard{ lock_ };
            deque_.clear();
        }

        void cancel_wait() noexcept
        {
            wait_ = false;
            notify_.notify_one();
        }

        void set_max_size(uint32_t size) noexcept
        {
            max_size_ = size;
        }

    private:
        bool wait_ = true;
        uint32_t max_size_;
        std::mutex lock_;
        std::condition_variable notify_;
        std::deque<T, Allocator> deque_;
    };
}