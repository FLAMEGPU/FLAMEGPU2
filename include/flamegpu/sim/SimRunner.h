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

namespace flamegpu {

struct ModelData;
class LoggingConfig;
class StepLoggingConfig;
class RunPlanVector;

/**
 * A thread class which executes RunPlans on a single GPU
 *
 * This class is used by CUDAEnsemble, it creates one SimRunner instance per GPU, each executes in a separate thread.
 * There may be multiple instances per GPU, if running small models on large GPUs.
 */
class SimRunner {
    friend class CUDAEnsemble;
    /**
     * Constructor, creates and initialise a new SimRunner
     * @param _model A copy of the ModelDescription hierarchy for the RunPlanVector, this is used to create the CUDASimulation instances.
     * @param _err_ct Reference to an atomic integer for tracking how many errors have occurred
     * @param _next_run Atomic counter for safely selecting the next run plan to execute across multiple threads
     * @param _plans The vector of run plans to be executed by the ensemble
     * @param _step_log_config The config of which data should be logged each step
     * @param _exit_log_config The config of which data should be logged at run exit
     * @param _device_id The GPU that all runs should execute on
     * @param _runner_id A unique index assigned to the runner
     * @param _verbose If true more information will be written to stdout
     * @param run_logs Reference to the vector to store generate run logs
     * @param log_export_queue The queue of logs to exported to disk
     * @param log_export_queue_mutex This mutex must be locked to access log_export_queue
     * @param log_export_queue_cdn The condition is notified every time a log has been added to the queue
     */
    SimRunner(const std::shared_ptr<const ModelData> _model,
        std::atomic<unsigned int> &_err_ct,
        std::atomic<unsigned int> &_next_run,
        const RunPlanVector &_plans,
        std::shared_ptr<const StepLoggingConfig> _step_log_config,
        std::shared_ptr<const LoggingConfig> _exit_log_config,
        int _device_id,
        unsigned int _runner_id,
        bool _verbose,
        std::vector<RunLog> &run_logs,
        std::queue<unsigned int> &log_export_queue,
        std::mutex &log_export_queue_mutex,
        std::condition_variable &log_export_queue_cdn);
    /**
     * Each sim runner takes it's own clone of model description hierarchy, so it can manipulate environment without conflict
     */
    const std::shared_ptr<const ModelData> model;
    /**
     * Index of current/last run in RunPlanVector executed by this SimRunner
     */
    unsigned int run_id;
    /**
     * CUDA Device index of runner
     */
    const int device_id;
    /**
     * Per instance unique runner id
     */
    const unsigned int runner_id;
    /**
     * Flag for whether to print progress
     */
    const bool verbose;
    /**
     * The thread which the SimRunner executes on
     */
    std::thread thread;
    /**
     * Start executing the SimRunner in it's separate thread
     */
    void start();
    // External references
    /**
     * Reference to an atomic integer for tracking how many errors have occurred
     */
    std::atomic<unsigned int> &err_ct;
    /**
     * Atomic counter for safely selecting the next run plan to execute across multiple threads
     */
    std::atomic<unsigned int> &next_run;
    /**
     * Reference to the vector of run configurations to be executed
     */
    const RunPlanVector &plans;
    /**
     * Config specifying which data to log per step
     */
    const std::shared_ptr<const StepLoggingConfig> step_log_config;
    /**
     * Config specifying which data to log at run exit
     */
    const std::shared_ptr<const LoggingConfig> exit_log_config;
    /**
     * Reference to the vector to store generate run logs
     */
    std::vector<RunLog> &run_logs;
    /**
     * The queue of logs to exported to disk
     */
    std::queue<unsigned int> &log_export_queue;
    /**
     * This mutex must be locked to access log_export_queue
     */
    std::mutex &log_export_queue_mutex;
    /**
     * The condition is notified every time a log has been added to the queue
     */
    std::condition_variable &log_export_queue_cdn;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_SIMRUNNER_H_
