#include "MathWorker.h"

#include "boost/ut.hpp"

#include <mutex>
#include <condition_variable>
#include <thread>

using namespace boost::ut;
using boost::ut::suite;

namespace
{
    std::once_flag initialized_math_workers_;
}

void once_start_math_workers()
{
    std::call_once(initialized_math_workers_,
                   []()
                   {
                       g_cMathWorker->StartWorkerThreads(0);
                   });
}

namespace
{
    struct JoinTestCallbackInfo
    {
        std::mutex mutex_;

        enum class WorkerState
        {
            INITIAL,
            READY,
            DONE
        } workerState_{ WorkerState::INITIAL };
        std::condition_variable workerSignal_;

        enum class MainState
        {
            INITIAL,
            DONE
        } mainState_{ MainState::INITIAL };
        std::condition_variable mainSignal_;
    };

    static bool JoinTestCallback(struct stMATH_WORKER_ITEM * pstWorkerItem, int core, int64_t workIndex)
    {
        auto & callbackInfo{ *static_cast<JoinTestCallbackInfo *>(pstWorkerItem->WorkCallbackArg) };

        if (workIndex == 0) // main thread
        {
            std::unique_lock lock{ callbackInfo.mutex_ };
            // Wait for worker to be ready.
            callbackInfo.workerSignal_.wait(lock,
                                            [&]
                                            {
                                                return callbackInfo.workerState_ == JoinTestCallbackInfo::WorkerState::READY;
                                            });
            lock.unlock();

            bool didSomeWork{ false };
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

            // Notify worker thread we're completed.
            lock.lock();
            callbackInfo.mainState_ = JoinTestCallbackInfo::MainState::DONE;
            lock.unlock();
            callbackInfo.mainSignal_.notify_all();

            return didSomeWork;
        }

        else // worker thread(s)
        {
            std::unique_lock lock{ callbackInfo.mutex_ };
            // Immediately complete all worker threads but the first.
            if (callbackInfo.workerState_ != JoinTestCallbackInfo::WorkerState::INITIAL)
            {
                return false;
            }

            // Notify the main thread we're ready.
            callbackInfo.workerState_ = JoinTestCallbackInfo::WorkerState::READY;
            lock.unlock();
            callbackInfo.workerSignal_.notify_all();

            // Wait for main thread to complete.
            lock.lock();
            callbackInfo.mainSignal_.wait(lock,
                                          [&]
                                          {
                                              return callbackInfo.mainState_ == JoinTestCallbackInfo::MainState::DONE;
                                          });

            // Give main thread opportunity to exit (in buggy case)
            lock.unlock();
            std::this_thread::yield();

            // Notify main thread we're done.
            lock.lock();
            callbackInfo.workerState_ = JoinTestCallbackInfo::WorkerState::DONE;
            lock.unlock();
            callbackInfo.workerSignal_.notify_all();

            return false;
        }
    }

    suite math_worker_ops = []
    {
        "work_main_joins_workers"_test = [&]
        {
            expect(g_cMathWorker != nullptr);
            once_start_math_workers();

            JoinTestCallbackInfo callbackInfo{};

            auto * workItem{ g_cMathWorker->GetWorkItem(CMathWorker::WORK_ITEM_BIG) };
            workItem->DoWorkCallback = JoinTestCallback;
            workItem->WorkCallbackArg = &callbackInfo;

            int32_t const threadWakeup{ 0 };
            int64_t const len{ 1 };
            g_cMathWorker->WorkMain(workItem, len, threadWakeup);

            // Verify that the worker thread is not running.
            std::unique_lock lock{ callbackInfo.mutex_ };
            expect(callbackInfo.workerState_ == JoinTestCallbackInfo::WorkerState::DONE);

            // Unblock worker if it's still running (in buggy case).
            callbackInfo.mainState_ = JoinTestCallbackInfo::MainState::DONE;
            lock.unlock();
            callbackInfo.mainSignal_.notify_all();

            lock.lock();
            callbackInfo.workerSignal_.wait(lock,
                                            [&]
                                            {
                                                return callbackInfo.workerState_ == JoinTestCallbackInfo::WorkerState::DONE;
                                            });

            lock.unlock();
        };
    };
}
