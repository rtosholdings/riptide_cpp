#include "riptide_python_test.h"

#include "MathWorker.h"

#define BOOST_UT_DISABLE_MODULE
#include "../ut/include/boost/ut.hpp"

#include <mutex>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    struct CALLBACK_INFO
    {
        std::mutex mutex_;

        enum class WorkerState
        {
            INITIAL,
            READY,
            DONE
        } workerState_{WorkerState::INITIAL};
        std::condition_variable workerSignal_;

        enum class MainState
        {
            INITIAL,
            DONE
        } mainState_{MainState::INITIAL};
        std::condition_variable mainSignal_;
    };

    static bool Callback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
    {
        auto & callbackInfo{*static_cast<CALLBACK_INFO *>(pstWorkerItem->WorkCallbackArg)};

        if (workIndex == 0) // main thread
        {
            // Wait for worker to be ready.
            {
                std::unique_lock lock{callbackInfo.mutex_};
                callbackInfo.workerSignal_.wait(lock, [&]
                {
                    return callbackInfo.workerState_ == CALLBACK_INFO::WorkerState::READY;
                });
            }

            bool didSomeWork{false};
            int64_t lenX;
            int64_t workBlock;

            // As long as there is work to do
            while ((lenX = pstWorkerItem->GetNextWorkBlock(&workBlock)) > 0)
            {
                // Indicate we completed a block
                didSomeWork = true;

                // tell others we completed this work block
                pstWorkerItem->CompleteWorkBlock();
            }

            return didSomeWork;
        }

        else // worker thread(s)
        {
            {
                std::lock_guard _{callbackInfo.mutex_};
                if (callbackInfo.workerState_ != CALLBACK_INFO::WorkerState::INITIAL)
                {
                    return false;
                }
                callbackInfo.workerState_ = CALLBACK_INFO::WorkerState::READY;
            }
            callbackInfo.workerSignal_.notify_all();

            {
                std::unique_lock lock{callbackInfo.mutex_};
                callbackInfo.mainSignal_.wait(lock, [&]
                {
                    return callbackInfo.mainState_ == CALLBACK_INFO::MainState::DONE;
                });
            }

            {
                std::lock_guard _{callbackInfo.mutex_};
                callbackInfo.workerState_ = CALLBACK_INFO::WorkerState::DONE;
            }
            callbackInfo.workerSignal_.notify_all();

            return false;
        }
    }

    suite math_worker_ops = []
    {
        "work_main_waits"_test = [&]
        {
            expect(g_cMathWorker != nullptr);

            CALLBACK_INFO callbackInfo{};

            auto * workItem{g_cMathWorker->GetWorkItem(CMathWorker::WORK_ITEM_BIG)};
            workItem->DoWorkCallback = Callback;
            workItem->WorkCallbackArg = &callbackInfo;

            int32_t const threadWakeup{0};
            int64_t const len{1};
            g_cMathWorker->WorkMain(workItem, len, threadWakeup);

            // Ensure the worker thread is not running.

            // Unblock worker if it's still running.
            {
                std::lock_guard _{callbackInfo.mutex_};
                callbackInfo.mainState_ = CALLBACK_INFO::MainState::DONE;
            }
            callbackInfo.mainSignal_.notify_all();

            {
                std::unique_lock lock{callbackInfo.mutex_};
                callbackInfo.workerSignal_.wait(lock, [&]
                {
                    return callbackInfo.workerState_ == CALLBACK_INFO::WorkerState::DONE;
                });
            }

            int i{};
            ++i;
        };
    };
}
