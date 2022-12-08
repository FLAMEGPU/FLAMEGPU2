#include "flamegpu/simulation/detail/SimLogger.h"

#include <filesystem>

#include "flamegpu/io/LoggerFactory.h"
#include "flamegpu/simulation/RunPlanVector.h"

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace flamegpu {
namespace detail {

SimLogger::SimLogger(const std::vector<RunLog> &_run_logs,
        const RunPlanVector &_run_plans,
        const std::string &_out_directory,
        const std::string &_out_format,
        std::queue<unsigned int> &_log_export_queue,
        std::mutex &_log_export_queue_mutex,
        std::condition_variable &_log_export_queue_cdn,
        bool _export_step,
        bool _export_exit,
        bool _export_step_time,
        bool _export_exit_time)
    : run_logs(_run_logs)
    , run_plans(_run_plans)
    , out_directory(_out_directory)
    , out_format(_out_format)
    , log_export_queue(_log_export_queue)
    , log_export_queue_mutex(_log_export_queue_mutex)
    , log_export_queue_cdn(_log_export_queue_cdn)
    , export_step(_export_step)
    , export_exit(_export_exit)
    , export_step_time(_export_step_time)
    , export_exit_time(_export_exit_time) {
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
    const std::filesystem::path p_out_directory = out_directory;
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
            if (export_exit) {
                const std::filesystem::path exit_path = p_out_directory / std::filesystem::path(run_plans[target_log].getOutputSubdirectory()) / std::filesystem::path("exit." + out_format);
                const auto exit_logger = io::LoggerFactory::createLogger(exit_path.generic_string(), false, false);
                exit_logger->log(run_logs[target_log], run_plans[target_log], false, true, false, export_exit_time);
            }
            if (export_step) {
                const std::filesystem::path step_path = p_out_directory/std::filesystem::path(run_plans[target_log].getOutputSubdirectory())/std::filesystem::path(std::to_string(target_log)+"."+out_format);
                const auto step_logger = io::LoggerFactory::createLogger(step_path.generic_string(), false, true);
                step_logger->log(run_logs[target_log], run_plans[target_log], true, false, export_step_time, false);
            }
            // Continue
            ++logs_processed;
            lock.lock();
        } while (!log_export_queue.empty());
    }
}

}  // namespace detail
}  // namespace flamegpu
