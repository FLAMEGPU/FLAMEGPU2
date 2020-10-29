#ifndef INCLUDE_FLAMEGPU_SIM_SIMRUNNER_H_
#define INCLUDE_FLAMEGPU_SIM_SIMRUNNER_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include <vector>
#include "flamegpu/sim/LogFrame.h"

struct ModelData;
class LoggingConfig;
class StepLoggingConfig;
class RunPlanVec;

class SimRunner {
    friend class CUDAEnsemble;
    SimRunner(const std::shared_ptr<const ModelData> _model,
        std::atomic<unsigned int> &_err_ct,
        std::atomic<unsigned int> &_next_run,
        const RunPlanVec &_plans,
        std::shared_ptr<const StepLoggingConfig> _step_log_config,
        std::shared_ptr<const LoggingConfig> _exit_log_config,
        int _device_id,
        unsigned int _runner_id,
        bool _verbose,
        std::vector<RunLog> &run_logs,
        std::queue<unsigned int> &log_export_queue,
        std::mutex &log_export_queue_mutex,
        std::condition_variable &log_export_queue_cdn);
    // Each sim runner takes it's own clone of model description hierarchy, so it can manipulate environment without conflict
    const std::shared_ptr<const ModelData> model;
    // Index of current/last run in RunPlanVec
    unsigned int run_id;
    // CUDA Device index of runner
    const int device_id;
    // Per device unique runner id
    const unsigned int runner_id;
    // Flag for whether to print progress
    const bool verbose;
    std::thread thread;
    void start();
    // External references
    std::atomic<unsigned int> &err_ct;
    std::atomic<unsigned int> &next_run;
    const RunPlanVec &plans;
    const std::shared_ptr<const StepLoggingConfig> step_log_config;
    const std::shared_ptr<const LoggingConfig> exit_log_config;
    std::vector<RunLog> &run_logs;
    std::queue<unsigned int> &log_export_queue;
    std::mutex &log_export_queue_mutex;
    std::condition_variable &log_export_queue_cdn;
};

#endif  // INCLUDE_FLAMEGPU_SIM_SIMRUNNER_H_
