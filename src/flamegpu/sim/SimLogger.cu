#include "flamegpu/sim/SimLogger.h"

#include "flamegpu/io/LoggerFactory.h"
#include "flamegpu/sim/RunPlanVector.h"

// If earlier than VS 2019
#if defined(_MSC_VER) && _MSC_VER < 1920
#include <filesystem>
using std::tr2::sys::exists;
using std::tr2::sys::path;
using std::tr2::sys::create_directory;
#else
// VS2019 requires this macro, as building pre c++17 cant use std::filesystem
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
using std::experimental::filesystem::v1::exists;
using std::experimental::filesystem::v1::path;
using std::experimental::filesystem::v1::create_directory;
#endif

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace flamegpu {

SimLogger::SimLogger(const std::vector<RunLog> &_run_logs,
        const RunPlanVector &_run_plans,
        const std::string &_out_directory,
        const std::string &_out_format,
        std::queue<unsigned int> &_log_export_queue,
        std::mutex &_log_export_queue_mutex,
        std::condition_variable &_log_export_queue_cdn)
    : run_logs(_run_logs)
    , run_plans(_run_plans)
    , out_directory(_out_directory)
    , out_format(_out_format)
    , log_export_queue(_log_export_queue)
    , log_export_queue_mutex(_log_export_queue_mutex)
    , log_export_queue_cdn(_log_export_queue_cdn) {
    this->thread = std::thread(&SimLogger::start, this);
    // Attempt to name the thread
#ifdef _MSC_VER
    std::wstringstream thread_name;
    thread_name << L"SimLogger";
    // HRESULT hr =
    SetThreadDescription(this->thread.native_handle(), thread_name.str().c_str());
    // if (FAILED(hr)) {
    //     fprintf(stderr, "Failed to name thread 'SimLogger'\n");
    // }
#else
    std::stringstream thread_name;
    thread_name << "SimLogger";
    // int hr =
    pthread_setname_np(this->thread.native_handle(), thread_name.str().c_str());
    // if (hr) {
    //     fprintf(stderr, "Failed to name thread 'SimLogger'\n");
    // }
#endif
}
void SimLogger::start() {
    const path p_out_directory = out_directory;
    unsigned int logs_processed = 0;
    while (logs_processed < run_plans.size()) {
        std::unique_lock<std::mutex> lock(log_export_queue_mutex);
        log_export_queue_cdn.wait(lock, [this]{ return !log_export_queue.empty(); });
        do {
            // Pop item to be logged from queue
            const unsigned int target_log = log_export_queue.front();
            log_export_queue.pop();
            lock.unlock();

            // Check item isn't telling us to exit early
            if (target_log == UINT_MAX) {
                logs_processed = UINT_MAX;
                break;
            }
            // Log items
            const path exit_path = p_out_directory/path(run_plans[target_log].getOutputSubdirectory())/path("exit." + out_format);
            const auto exit_logger = LoggerFactory::createLogger(exit_path.generic_string(), false, false);
            exit_logger->log(run_logs[target_log], true, false, true);
            const path step_path = p_out_directory/path(run_plans[target_log].getOutputSubdirectory())/path(std::to_string(target_log)+"."+out_format);
            const auto step_logger = LoggerFactory::createLogger(step_path.generic_string(), false, false);
            step_logger->log(run_logs[target_log], true, true, false);

            // Continue
            ++logs_processed;
            lock.lock();
        } while (!log_export_queue.empty());
    }
}

}  // namespace flamegpu
