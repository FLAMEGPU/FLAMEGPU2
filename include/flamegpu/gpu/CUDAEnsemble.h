#ifndef INCLUDE_FLAMEGPU_GPU_CUDAENSEMBLE_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAENSEMBLE_H_

#include <string>
#include <memory>
#include <set>
#include <vector>

#include "flamegpu/defines.h"

namespace flamegpu {

struct ModelData;
class ModelDescription;
class RunPlanVector;
class LoggingConfig;
class StepLoggingConfig;
struct RunLog;
/**
 * Manager for automatically executing multiple copies of a model simultaneously
 * This can be used to conveniently execute parameter sweeps and batch validation runs
 */
class CUDAEnsemble {
 public:
    /**
     * Execution config for running a CUDAEnsemble
     */
    struct EnsembleConfig {
        /**
         * Directory to store output data (primarily logs)
         * Defaults to "" (the working directory, no subdirectory)
         */
        std::string out_directory = "";
        /**
         * Output format
         * This must be a supported format e.g.: "json" or "xml"
         * Defaults to "json"
         */
        std::string out_format = "json";
        /**
         * The maximum number of concurrent runs
         * Defaults to 4
         */
        unsigned int concurrent_runs = 4;
        /**
         * The CUDA device ids of devices to be used
         * If this is left empty, all available devices will be used
         * Defaults to empty set (all available devices)
         */
        std::set<int> devices;
        /**
         * If true progress logging to stdout will be suppressed
         * Defaults to false
         */
        flamegpu::Verbosity verbosity = Verbosity::Default;
        /**
         * Suppresses warning for unknown arguments passed to the CUDAEnsemble during initialisation. Useful for when arguments are passed to user defined models but should
         * not be considered by the FLAME GPU API.
         */
        bool silence_unknown_args = false;
        /**
         * If true, the total runtime for the ensemble will be printed to stdout at completion
         * This is independent of the EnsembleConfig::quiet
         * Defaults to false
         */
        bool timing = false;
        enum ErrorLevel { Off = 0, Slow = 1, Fast = 2 };
        /**
         * Off: Runs which fail do not cause an exception to be raised. Failed runs must be probed manually via checking the return value of calls to CUDAEnsemble::simulate()
         * Slow: If any runs fail, an EnsembleException will be raised after all runs have been attempted, before CUDAEnsemble::simulate() returns.
         * Fast: An EnsembleException will be raised as soon as a failed run is detected, cancelling remaining runs.
         * Defaults to Slow
         */
        ErrorLevel error_level = Slow;
        /**
         * If true, all log files created will truncate any existing files with the same name
         * If false, an exception will be raised when a log file already exists
         */
        bool truncate_log_files = false;
        /**
         * Prevents the computer from entering standby whilst the ensemble is running
         * @note This feature is currently only supported by Windows builds.
         */
#ifdef _MSC_VER
        bool block_standby = true;
#else
        const bool block_standby = false;
#endif
    };
    /**
     * Initialise CUDA Ensemble
     * If provided, you can pass runtime arguments to this constructor, to automatically call initialise()
     * This is not required, you can call initialise() manually later, or not at all.
     * @param model The model description to initialise the runner to execute
     * @param argc Runtime argument count
     * @param argv Runtime argument list ptr
     */
    explicit CUDAEnsemble(const ModelDescription& model, int argc = 0, const char** argv = nullptr);
    /**
     * Inverse operation of constructor
     */
    ~CUDAEnsemble();

    /**
     * Execute the ensemble of simulations.
     * This call will normally block until all simulations have completed, however may exit early if an error occurs with the error_level configuration set to Fast 
     * @param plan The plan of individual runs to execute during the ensemble
     * @return 0 on success, otherwise the number of runs which reported errors and failed
     * @see CUDAEnsemble::EnsembleConfig::error_level
     */
    unsigned int simulate(const RunPlanVector &plan);

    /**
     * @return A mutable reference to the ensemble configuration struct
     */
    EnsembleConfig &Config() { return config; }
    /**
     * @return An immutable reference to the ensemble configuration struct
     */
    const EnsembleConfig &getConfig() const { return config; }
    /*
     * Override current config with args passed via CLI
     * @note Config values not passed via CLI will remain as their current values (and not be reset to default)
     */
    void initialise(int argc, const char** argv);
    /**
     * Configure which step data should be logged
     * @param stepConfig The step logging config for the CUDAEnsemble
     * @note This must be for the same model description hierarchy as the CUDAEnsemble
     */
    void setStepLog(const StepLoggingConfig &stepConfig);
    /**
     * Configure which exit data should be logged
     * @param exitConfig The logging config for the CUDAEnsemble
     * @note This must be for the same model description hierarchy as the CUDAEnsemble
     */
    void setExitLog(const LoggingConfig &exitConfig);
    /**
     * Get the duration of the last call to simulate() in milliseconds. 
     */
    double getEnsembleElapsedTime() const { return ensemble_elapsed_time; }
    /**
     * Return the list of logs collected from the last call to simulate()
     */
    const std::vector<RunLog> &getLogs();

 private:
    /**
     * Print command line interface help
     */
    void printHelp(const char *executable);
    /**
     * Parse CLI into config
     */
    int checkArgs(int argc, const char** argv);
    /**
     * Config options for the ensemble
     */
    EnsembleConfig config;
    /**
     * Step logging config
     */
    std::shared_ptr<const StepLoggingConfig> step_log_config;
    /**
     * Exit logging config
     */
    std::shared_ptr<const LoggingConfig> exit_log_config;
    /**
     * Logs collected by simulate()
     */
    std::vector<RunLog> run_logs;
    /**
     * Model description hierarchy for the ensemble, a copy of this will be passed to every CUDASimulation
     */
    const std::shared_ptr<const ModelData> model;
    /**
     * Runtime of previous call to simulate() in seconds, initially 0.
     */
    double ensemble_elapsed_time = 0.;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAENSEMBLE_H_
