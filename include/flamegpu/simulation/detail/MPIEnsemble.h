#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_MPIENSEMBLE_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_MPIENSEMBLE_H_

#include <mpi.h>

#include <map>
#include <string>
#include <mutex>
#include <vector>

#include "flamegpu/simulation/CUDAEnsemble.h"
#include "flamegpu/simulation/detail/MPISimRunner.h"

namespace flamegpu {
namespace detail {

class MPIEnsemble {
    const CUDAEnsemble::EnsembleConfig &config;
    // Tags to different the MPI messages used in protocol
    enum EnvelopeTag : int {
        // Sent from worker to manager to request a job index to process
        RequestJob = 0,
        // Sent from manager to worker to assign a job index to process in response to AssignJob
        AssignJob = 1,
        // Sent from worker to manager to report an error during job execution
        // If fail fast is enabled, following RequestJob will receive an exit job id (>=plans.size())
        ReportError = 2,
        // Sent from worker to manager to report GPUs for telemetry
        TelemetryDevices = 3,
    };

 public:
    const int world_rank;
    const int world_size;
    const unsigned int total_runs;
    /**
     * Construct the object for managing MPI comms during an ensemble
     *
     * Initialises the MPI_Datatype MPI_ERROR_DETAIL, detects world rank and size.
     *
     * @param _config The parent ensemble's config (mostly used to check error/verbosity levels)
     * @param _total_runs The total number of runs to be executed (only used for printing error warnings)
     */
    explicit MPIEnsemble(const CUDAEnsemble::EnsembleConfig &_config, unsigned int _total_runs);
    /**
     * If world_rank==0, receive any waiting errors and add their details to err_detail
     * @param err_detail The map to store new error details within
     * @return The number of errors that occurred.
     */
    int receiveErrors(std::multimap<int, AbstractSimRunner::ErrorDetail> &err_detail);
    /**
     * If world_rank==0, receive and process any waiting job requests
     * @param next_run A reference to the int which tracks the progress through the run plan vector
     * @return The number of runners that have been told to exit (if next_run>total_runs)
     */
    int receiveJobRequests(unsigned int &next_run);
    /**
     * If world_rank!=0, send the provided error detail to world_rank==0
     * @param e_detail The error detail to be sent
     */
    void sendErrorDetail(AbstractSimRunner::ErrorDetail &e_detail);
    /**
     * If world_rank!=0, request a job from world_rank==0 and return the response
     * @return The index of the assigned job
     */
    int requestJob();
    /**
     * Wait for all MPI ranks to reach a barrier
     */
    void worldBarrier();
    /**
     * If world_rank!=0, send the local GPU string to world_rank==0 and return empty string
     * If world_rank==0, receive GPU strings and assemble the full remote GPU string to be returned
     */
    std::string assembleGPUsString();
    /**
     * Common function for handling local errors during MPI execution
     */
    void retrieveLocalErrorDetail(std::mutex &log_export_queue_mutex,
        std::multimap<int, AbstractSimRunner::ErrorDetail> &err_detail,
        std::vector<AbstractSimRunner::ErrorDetail> &err_detail_local, int i);

 private:
    /**
     * @return Retrieve the local world rank from MPI
     */
    int getWorldRank();
    /**
    * @return Retrieve the world size from MPI
    */
    int getWorldSize();
    /**
     * If necessary initialise MPI, else do nothing
     */
    void initMPI();
    /**
     * Iterate config.devices to find the item at index j
     */
    unsigned int getDeviceIndex(const int j);
    /**
     * MPI representation of AbstractSimRunner::ErrorDetail type
     */
    const MPI_Datatype MPI_ERROR_DETAIL;
};
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_MPIENSEMBLE_H_
