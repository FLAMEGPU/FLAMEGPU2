#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_ABSTRACTSIMRUNNER_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_ABSTRACTSIMRUNNER_H_

#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <vector>
#include <memory>

#ifdef FLAMEGPU_ENABLE_MPI
#include <mpi.h>
#endif

#include "flamegpu/defines.h"
#include "flamegpu/simulation/LogFrame.h"

namespace flamegpu {
struct ModelData;
class LoggingConfig;
class StepLoggingConfig;
class RunPlanVector;
class CUDAEnsemble;
namespace detail {
/**
* Common interface and implementation shared between SimRunner and MPISimRunner
*/
class AbstractSimRunner {
    friend class flamegpu::CUDAEnsemble;
    /**
     * Create a new thread and trigger main() to execute the SimRunner
     */
    void start();

 public:
    struct ErrorDetail {
        unsigned int run_id;
        unsigned int device_id;
        unsigned int runner_id;
        char exception_string[1024];
    };
#ifdef FLAMEGPU_ENABLE_MPI
    static MPI_Datatype createErrorDetailMPIDatatype() {
        static MPI_Datatype rtn = MPI_DATATYPE_NULL;
        if (rtn == MPI_DATATYPE_NULL) {
            ErrorDetail t;
            constexpr int count = 4;
            constexpr int array_of_blocklengths[count] = {1, 1, 1, sizeof(ErrorDetail::exception_string)};
            const char* t_ptr = reinterpret_cast<char*>(&t);
            const MPI_Aint array_of_displacements[count] = { reinterpret_cast<char*>(&t.run_id) - t_ptr,
                                                             reinterpret_cast<char*>(&t.device_id) - t_ptr,
                                                             reinterpret_cast<char*>(&t.runner_id) - t_ptr,
                                                             reinterpret_cast<char*>(&t.exception_string) - t_ptr };
            constexpr MPI_Datatype array_of_types[count] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_CHAR};
            MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements, array_of_types, &rtn);
            MPI_Type_commit(&rtn);
        }
        return rtn;
    }
#endif

    /**
     * Constructor, creates and initialises the underlying thread
     * @param _model A copy of the ModelDescription hierarchy for the RunPlanVector, this is used to create the CUDASimulation instances.
     * @param _err_ct Reference to an atomic integer for tracking how many errors have occurred
     * @param _next_run Atomic counter for safely selecting the next run plan to execute across multiple threads
     * @param _plans The vector of run plans to be executed by the ensemble
     * @param _step_log_config The config of which data should be logged each step
     * @param _exit_log_config The config of which data should be logged at run exit
     * @param _device_id The GPU that all runs should execute on
     * @param _runner_id A unique index assigned to the runner
     * @param _verbosity Verbosity level (Verbosity::Quiet, Verbosity::Default, Verbosity::Verbose)
     * @param run_logs Reference to the vector to store generate run logs
     * @param log_export_queue The queue of logs to exported to disk
     * @param log_export_queue_mutex This mutex must be locked to access log_export_queue
     * @param log_export_queue_cdn The condition is notified every time a log has been added to the queue
     * @param err_detail Structure to store error details on fast failure for main thread rethrow
     * @param _total_runners Total number of runners executing
     * @param _isSWIG Flag denoting whether it's a Python build of FLAMEGPU
     */
    AbstractSimRunner(const std::shared_ptr<const ModelData> _model,
        std::atomic<unsigned int> &_err_ct,
        std::atomic<unsigned int> &_next_run,
        const RunPlanVector &_plans,
        std::shared_ptr<const StepLoggingConfig> _step_log_config,
        std::shared_ptr<const LoggingConfig> _exit_log_config,
        int _device_id,
        unsigned int _runner_id,
        flamegpu::Verbosity _verbosity,
        std::vector<RunLog> &run_logs,
        std::queue<unsigned int> &log_export_queue,
        std::mutex &log_export_queue_mutex,
        std::condition_variable &log_export_queue_cdn,
        std::vector<ErrorDetail> &err_detail,
        unsigned int _total_runners,
        bool _isSWIG);
    /**
     * Virtual class requires polymorphic destructor
     */
    virtual ~AbstractSimRunner() {}
    /**
     * Subclass implementation of SimRunner
     */
    virtual void main() = 0;
    /**
     * Blocking call which if thread->joinable() triggers thread->join() 
     */
    void join();

 protected:
    /**
     * Create and execute the simulation for the RunPlan within plans of given index
     * @throws Exceptions during sim execution may be raised, these should be caught and handled by the caller
     */
    void runSimulation(int plan_id);
    /**
    * The thread which the SimRunner executes on
    */
    std::thread thread;
    /**
    * Each sim runner takes it's own clone of model description hierarchy, so it can manipulate environment without conflict
    */
    const std::shared_ptr<const ModelData> model;
    /**
    * CUDA Device index of runner
    */
    const int device_id;
    /**
    * Per instance unique runner id
    */
    const unsigned int runner_id;
    /**
    * Total number of runners executing
    * This is used to calculate the progress on job completion
    */
    const unsigned int total_runners;
    /**
    * Flag for whether to print progress
    */
    const flamegpu::Verbosity verbosity;
    // External references
    /**
     * Reference to an atomic integer for tracking how many errors have occurred
     */
    std::atomic<unsigned int> &err_ct;
    /**
     * Atomic counter for safely selecting the next run plan to execute across multiple threads
     * This is used differently by each class of runner
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
     * Reference to the vector to store generated run logs
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
    /**
     * Error details will be stored here
     */
    std::vector<ErrorDetail>& err_detail;
    /**
     * If true, the model is using SWIG Python interface
     **/
    const bool isSWIG;
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_ABSTRACTSIMRUNNER_H_
