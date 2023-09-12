#include "flamegpu/simulation/detail/AbstractSimRunner.h"

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "flamegpu/model/ModelData.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/simulation/RunPlanVector.h"

namespace flamegpu {
namespace detail {

AbstractSimRunner::AbstractSimRunner(const std::shared_ptr<const ModelData> _model,
    std::atomic<unsigned int> &_err_ct,
    std::atomic<unsigned int> &_next_run,
    const RunPlanVector &_plans,
    std::shared_ptr<const StepLoggingConfig> _step_log_config,
    std::shared_ptr<const LoggingConfig> _exit_log_config,
    int _device_id,
    unsigned int _runner_id,
    flamegpu::Verbosity _verbosity,
    std::map<unsigned int, RunLog> &_run_logs,
    std::queue<unsigned int> &_log_export_queue,
    std::mutex &_log_export_queue_mutex,
    std::condition_variable &_log_export_queue_cdn,
    std::vector<ErrorDetail> &_err_detail,
    const unsigned int _total_runners,
    bool _isSWIG)
      : model(_model->clone())
      , device_id(_device_id)
      , runner_id(_runner_id)
      , total_runners(_total_runners)
      , verbosity(_verbosity)
      , err_ct(_err_ct)
      , next_run(_next_run)
      , plans(_plans)
      , step_log_config(std::move(_step_log_config))
      , exit_log_config(std::move(_exit_log_config))
      , run_logs(_run_logs)
      , log_export_queue(_log_export_queue)
      , log_export_queue_mutex(_log_export_queue_mutex)
      , log_export_queue_cdn(_log_export_queue_cdn)
      , err_detail(_err_detail)
      , isSWIG(_isSWIG) {
}
void AbstractSimRunner::start() {
    this->thread = std::thread(&AbstractSimRunner::main, this);
    // Attempt to name the thread
#ifdef _MSC_VER
    std::wstringstream thread_name;
    thread_name << L"CUDASim D" << device_id << L"T" << runner_id;
    // HRESULT hr =
    SetThreadDescription(this->thread.native_handle(), thread_name.str().c_str());
    // if (FAILED(hr)) {
    //     fprintf(stderr, "Failed to name thread 'CUDASim D%dT%u'\n", device_id, runner_id);
    // }
#else
    std::stringstream thread_name;
    thread_name << "CUDASim D" << device_id << "T" << runner_id;
    // int hr =
    pthread_setname_np(this->thread.native_handle(), thread_name.str().c_str());
    // if (hr) {
    //     fprintf(stderr, "Failed to name thread 'CUDASim D%dT%u'\n", device_id, runner_id);
    // }
#endif
}
void AbstractSimRunner::join() {
    if (this->thread.joinable()) {
        this->thread.join();
    }
}

void AbstractSimRunner::runSimulation(int plan_id) {
    // Update environment (this might be worth moving into CUDASimulation)
    auto &prop_map = model->environment->properties;
    for (auto &ovrd : plans[plan_id].property_overrides) {
        auto &prop = prop_map.at(ovrd.first);
        memcpy(prop.data.ptr, ovrd.second.ptr, prop.data.length);
    }
    // Set simulation device
    std::unique_ptr<CUDASimulation> simulation = std::unique_ptr<CUDASimulation>(new CUDASimulation(model, isSWIG));
    // Copy steps and seed from runplan
    simulation->SimulationConfig().steps = plans[plan_id].getSteps();
    simulation->SimulationConfig().random_seed = plans[plan_id].getRandomSimulationSeed();
    simulation->SimulationConfig().verbosity = Verbosity::Default;
    if (verbosity == Verbosity::Quiet)  // Use quiet verbosity for sims if set in ensemble but never verbose
        simulation->SimulationConfig().verbosity = Verbosity::Quiet;
    simulation->SimulationConfig().telemetry = false;   // Never any telemtry for individual runs inside an ensemble
    simulation->SimulationConfig().timing = false;
    simulation->CUDAConfig().device_id = this->device_id;
    simulation->CUDAConfig().is_ensemble = true;
    simulation->CUDAConfig().ensemble_run_id = plan_id;
    simulation->applyConfig();
    // Set the step config directly, to bypass validation
    simulation->step_log_config = step_log_config;
    simulation->exit_log_config = exit_log_config;
    // Don't need to set pop, this must be done via init function within ensembles
    // Execute simulation
    simulation->simulate();
    {
        std::lock_guard<std::mutex> lck(log_export_queue_mutex);
        // Store results in run_log
        run_logs.emplace(plan_id, simulation->getRunLog());
        // Notify logger
        log_export_queue.push(plan_id);
    }
    log_export_queue_cdn.notify_one();
}

}  // namespace detail
}  // namespace flamegpu
