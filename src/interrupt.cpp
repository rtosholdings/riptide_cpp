#include "interrupt.h"

#include <csignal>

namespace riptide
{
    namespace internal
    {
        std::atomic<bool> interrupted_{ false };

        void signal_handler(int signal)
        {
            interrupted_.store(true);
        }
    }

    bool interruptible_section(std::function<void(void)> function)
    {
        // Install our handler and save the previous handler
        auto previous_handler = std::signal(SIGINT, internal::signal_handler);
        internal::interrupted_.store(false);

        // Invoke the user-defined code
        function();

        // Restore the previous handler
        std::signal(SIGINT, previous_handler);
        return internal::interrupted_.exchange(false);
    }
}