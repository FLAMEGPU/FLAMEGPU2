#include "flamegpu/sim/SimRunner.h"

#include <utility>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/sim/RunPlanVector.h"

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace flamegpu {

SimRunner::SimRunner(const std::shared_ptr<const ModelData> _model,
    std::atomic<unsigned int> &_err_ct,
    std::atomic<unsigned int> &_next_run,
    const RunPlanVector &_plans,
    std::shared_ptr<const StepLoggingConfig> _step_log_config,
    std::shared_ptr<const LoggingConfig> _exit_log_config,
    int _device_id,
    unsigned int _runner_id,
    bool _verbose,
    std::vector<RunLog> &_run_logs,
    std::queue<unsigned int> &_log_export_queue,
    std::mutex &_log_export_queue_mutex,
    std::condition_variable &_log_export_queue_cdn)
      : model(_model->clone())
      , run_id(0)
      , device_id(_device_id)
      , runner_id(_runner_id)
      , verbose(_verbose)
      , err_ct(_err_ct)
      , next_run(_next_run)
      , plans(_plans)
      , step_log_config(std::move(_step_log_config))
      , exit_log_config(std::move(_exit_log_config))
      , run_logs(_run_logs)
      , log_export_queue(_log_export_queue)
      , log_export_queue_mutex(_log_export_queue_mutex)
      ,  log_export_queue_cdn(_log_export_queue_cdn) {
    this->thread = std::thread(&SimRunner::start, this);
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


void SimRunner::start() {
    // While there are still plans to process
    while ((this->run_id = next_run++) < plans.size()) {
        try {
            // Update environment (this might be worth moving into CUDASimulation)
            auto &prop_map = model->environment->properties;
            for (auto &ovrd : plans[run_id].property_overrides) {
                auto &prop = prop_map.at(ovrd.first);
                memcpy(prop.data.ptr, ovrd.second.ptr, prop.data.length);
            }
            // Set simulation device
            std::unique_ptr<CUDASimulation> simulation = std::unique_ptr<CUDASimulation>(new CUDASimulation(model));
            // Copy steps and seed from runplan
            simulation->SimulationConfig().steps = plans[run_id].getSteps();
            simulation->SimulationConfig().random_seed = plans[run_id].getRandomSimulationSeed();
            simulation->SimulationConfig().verbose = false;
            simulation->SimulationConfig().timing = false;
            simulation->CUDAConfig().device_id = this->device_id;
            simulation->applyConfig();
            // Set the step config directly, to bypass validation
            simulation->step_log_config = step_log_config;
            simulation->exit_log_config = exit_log_config;
            // TODO Set population?
            // Execute simulation
            simulation->simulate();
            // Store results in run_log (use placement new because const members)
            run_logs[this->run_id] = simulation->getRunLog();
            // Notify logger
            {
                std::lock_guard<std::mutex> lck(log_export_queue_mutex);
                log_export_queue.push(this->run_id);
            }
            log_export_queue_cdn.notify_one();
            // Print progress to console
            if (verbose) {
                fprintf(stdout, "\rCUDAEnsemble progress: %u/%u", run_id + 1, static_cast<unsigned int>(plans.size()));
                fflush(stdout);
            }
        } catch(std::exception &e) {
            fprintf(stderr, "\nRun %u failed on device %d, thread %u with exception: \n%s\n", run_id, device_id, runner_id, e.what());
        }
    }
}

}  // namespace flamegpu
