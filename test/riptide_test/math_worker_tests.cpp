#include "MathWorker.h"

#include "ut_core.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>

using namespace boost::ut;
using riptide_utility::ut::file_suite;

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

    struct GetNextWorkIndexTestState
    {
        std::mutex mutex;
        std::condition_variable wakeup;
        std::unordered_set<int32_t> cores;
    };

    static bool GetNextWorkIndexTestCallback(void * callbackArg, int32_t core, int64_t workIndex)
    {
        auto * state = static_cast<GetNextWorkIndexTestState *>(callbackArg);
        std::unique_lock lock(state->mutex);
        state->cores.insert(core);

        if (state->cores.size() == 1)
        {
            // This is the first thread, wait for the other
            state->wakeup.wait(lock);
        }
        else
        {
            // This is the second thread, wake up the first
            state->wakeup.notify_all();
        }

        lock.unlock();
        return true;
    }

    struct ThreadWakeUpTestState
    {
        std::mutex mutex;
        std::condition_variable condition;
        int64_t count;
        int64_t expected;
    };

    static bool ThreadWakeUpTestCallback(struct stMATH_WORKER_ITEM * worker_item, int core, int64_t work_index)
    {
        auto * state = static_cast<ThreadWakeUpTestState *>(worker_item->WorkCallbackArg);
        std::unique_lock lock(state->mutex);

        int64_t work_block;
        worker_item->GetNextWorkBlock(&work_block);

        if (++state->count < state->expected)
        {
            state->condition.notify_all();
        }
        else
        {
            auto pred = [&]
            {
                return state->count == state->expected;
            };

            // Wait for up to 10s
            state->condition.wait_for(lock, std::chrono::seconds(10), pred);
        }

        worker_item->CompleteWorkBlock();

        lock.unlock();
        return false;
    }

    file_suite math_worker_ops = []
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

        "get_next_work_index"_test = [&]
        {
            expect(fatal(g_cMathWorker != nullptr));
            once_start_math_workers();

            int64_t const numItems{ 2 };     // Two work items
            int32_t const threadWakeup{ 1 }; // One worker thread (plus main thread)
            GetNextWorkIndexTestState state;

            // This function uses GetNextWorkIndex internally to dispatch work
            g_cMathWorker->DoMultiThreadedWork(numItems, GetNextWorkIndexTestCallback, (void *)&state, threadWakeup);

            // Two threads should have done work
            expect(state.cores.size() == 2) << "state.cores=" << state.cores;

            // The main thread should be one of them
            expect(state.cores.contains(0)) << "state.cores=" << state.cores;
        };

        "thread_wakeup"_test = [&](int32_t thread_wakeup)
        {
            switch (thread_wakeup)
            {
            case 1:
                thread_wakeup = 1;
                break;
            case 2:
                thread_wakeup = g_cMathWorker->WorkerThreadCount / 2;
                break;
            case 3:
                thread_wakeup = g_cMathWorker->WorkerThreadCount;
                break;
            }

            expect(fatal(g_cMathWorker != nullptr));
            once_start_math_workers();

            auto * work_item{ g_cMathWorker->GetWorkItem(CMathWorker::WORK_ITEM_BIG) };
            work_item->DoWorkCallback = ThreadWakeUpTestCallback;

            ThreadWakeUpTestState state;
            state.count = 0;
            state.expected = thread_wakeup;
            work_item->WorkCallbackArg = &state;

            work_item->BlocksCompleted = 0;
            work_item->BlockNext = 0;
            work_item->BlockSize = 1;
            work_item->BlockLast = thread_wakeup;

            // Wake up threads
            g_cMathWorker->pWorkerRing->SetWorkItem(thread_wakeup);

            // Wait for all work items to be complete and all thread to go back to sleep
            while (work_item->BlocksCompleted < work_item->BlockLast || g_cMathWorker->pWorkerRing->AnyThreadsAwakened())
            {
                YieldProcessor();
            }

            // Check that the correct number of threads woke up
            expect(state.count == state.expected) << "expected" << state.expected << "thread(s) to wake up, got" << state.count;

            g_cMathWorker->pWorkerRing->CompleteWorkItem();
        } | std::vector<int32_t>{ 1, 2, 3 };
    };
}
