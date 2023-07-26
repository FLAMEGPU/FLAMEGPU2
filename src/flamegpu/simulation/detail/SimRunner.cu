#include "flamegpu/simulation/detail/SimRunner.h"

#include "flamegpu/simulation/RunPlanVector.h"

namespace flamegpu {
namespace detail {

SimRunner::SimRunner(const std::shared_ptr<const ModelData> _model,
    std::atomic<unsigned int> &_err_ct,
    std::atomic<unsigned int> &_next_run,
    const RunPlanVector &_plans,
    std::shared_ptr<const StepLoggingConfig> _step_log_config,
    std::shared_ptr<const LoggingConfig> _exit_log_config,
    int _device_id,
    unsigned int _runner_id,
    flamegpu::Verbosity _verbosity,
    bool _fail_fast,
    std::map<unsigned int, RunLog> &_run_logs,
    std::queue<unsigned int> &_log_export_queue,
    std::mutex &_log_export_queue_mutex,
    std::condition_variable &_log_export_queue_cdn,
    std::vector<ErrorDetail> &_err_detail,
    const unsigned int _total_runners,
    bool _isSWIG)
    : AbstractSimRunner(
        _model,
        _err_ct,
        _next_run,
        _plans,
        _step_log_config,
        _exit_log_config,
        _device_id,
        _runner_id,
        _verbosity,
        _run_logs,
        _log_export_queue,
        _log_export_queue_mutex,
        _log_export_queue_cdn,
        _err_detail,
        _total_runners,
        _isSWIG)
    , fail_fast(_fail_fast) { }


void SimRunner::main() {
    unsigned int run_id = 0;
    // While there are still plans to process
    while ((run_id = next_run++) < plans.size()) {
        try {
            runSimulation(run_id);
            // Print progress to console
            if (verbosity >= Verbosity::Default) {
                const int progress = static_cast<int>(next_run.load()) - static_cast<int>(total_runners) + 1;
                fprintf(stdout, "\rCUDAEnsemble progress: %d/%u", progress < 1 ? 1 : progress, static_cast<unsigned int>(plans.size()));
                fflush(stdout);
            }
        } catch(std::exception &e) {
            ++err_ct;
            if (this->fail_fast) {
                // Kill the other workers early
                next_run += static_cast<unsigned int>(plans.size());
                {
                    // log_export_mutex is treated as our protection for race conditions on err_detail
                    std::lock_guard<std::mutex> lck(log_export_queue_mutex);
                    log_export_queue.push(UINT_MAX);
                    // Build the error detail (fixed len char array for string)
                    err_detail.push_back(ErrorDetail{run_id, static_cast<unsigned int>(device_id), runner_id, });
                    strncpy(err_detail.back().exception_string, e.what(), sizeof(ErrorDetail::exception_string)-1);
                    err_detail.back().exception_string[sizeof(ErrorDetail::exception_string) - 1] = '\0';
                }
                return;
            } else {
                // Progress flush
                if (verbosity >= Verbosity::Default) {
                    fprintf(stdout, "\n");
                    fflush(stdout);
                }
                if (verbosity > Verbosity::Quiet)
                    fprintf(stderr, "Warning: Run %u failed on device %d, thread %u with exception: \n%s\n", run_id, device_id, runner_id, e.what());
            }
        }
    }
}

}  // namespace detail
}  // namespace flamegpu
