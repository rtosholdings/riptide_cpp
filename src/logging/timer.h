#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace riptide::logging::details
{
    class timer
    {
        using interval = std::chrono::milliseconds;

    public:
        ~timer()
        {
            cancel();
        }

        template <typename Function>
        void set_interval(Function function, interval delay)
        {
            if (active_)
                return;

            active_ = true;
            static_assert(! std::is_move_constructible_v<timer>, "Can't be movable because enclosing thread captures by ref");
            static_assert(! std::is_move_assignable_v<timer>, "Can't be movable because enclosing thread captures by ref");
            static_assert(! std::is_copy_constructible_v<timer>, "Can't be copyable because enclosing thread captures by ref");
            static_assert(! std::is_copy_assignable_v<timer>, "Can't be copyable because enclosing thread captures by ref");

            auto run = [=, this]()
            {
                while (active_)
                {
                    {
                        std::unique_lock lk{ lock_ };
                        wake_.wait_for(lk, delay);
                    }

                    if (! active_)
                        return;

                    function();
                }
            };

            runner_ = std::move(std::thread{ run });
        }

        void cancel()
        {
            if (! active_)
                return;

            {
                std::lock_guard guard{ lock_ };
                active_ = false;
            }
            wake_.notify_all();
            runner_.join();
        }

    private:
        std::mutex lock_;
        std::condition_variable wake_;
        bool active_ = false;
        std::thread runner_;
    };
}