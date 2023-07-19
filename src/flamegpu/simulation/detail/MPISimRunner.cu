#include "flamegpu/simulation/detail/MPISimRunner.h"

#include <utility>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/simulation/RunPlanVector.h"

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace flamegpu {
namespace detail {

MPISimRunner::MPISimRunner(const std::shared_ptr<const ModelData> _model,
    std::atomic<unsigned int>& _err_ct,
    std::atomic<unsigned int>& _next_run,
    const RunPlanVector& _plans,
    std::shared_ptr<const StepLoggingConfig> _step_log_config,
    std::shared_ptr<const LoggingConfig> _exit_log_config,
    int _device_id,
    unsigned int _runner_id,
    flamegpu::Verbosity _verbosity,
    std::vector<RunLog>& _run_logs,
    std::queue<unsigned int>& _log_export_queue,
    std::mutex& _log_export_queue_mutex,
    std::condition_variable& _log_export_queue_cdn,
    ErrorDetail& _fast_err_detail,
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
        _fast_err_detail,
        _total_runners,
        _isSWIG) { }


void MPISimRunner::main() {
    // While there are still plans to process
    while (true) {
        const unsigned int run_id = next_run.load();
        if (run_id < plans.size()) {
            // Process the assigned job
            try {
                runSimulation(run_id);
                if (next_run.exchange(Signal::RequestJob) >= plans.size()) {
                    break;
                }
                // MPI Worker's don't print progress
            } catch(std::exception&) {
                ++err_ct;
                // Need to notify manager that run failed
                if (next_run.exchange(Signal::RequestJob) >= plans.size()) {
                    break;
                }
            }
        } else if (run_id == Signal::RequestJob || run_id == Signal::RunFailed) {
            std::this_thread::yield();
        } else {
            break;
        }
    }
}

}  // namespace detail
}  // namespace flamegpu
