#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_MPIENSEMBLE_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_MPIENSEMBLE_H_

#include <mpi.h>

#include <map>
#include <set>
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
    /**
     * The rank within the world MPI communicator
     */
    const int world_rank;
    /**
     * The size of the world MPI communicator
     */
    const int world_size;
    /**
     * The rank within the MPI shared memory communicator (i.e. the within the node)
     */
    const int local_rank;
    /**
     * The size of the MPI shared memory communicator (i.e. the within the node)
     */
    const int local_size;
    /**
     * The total number of runs to be executed (only used for printing error warnings)
     */
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
     * If world_rank!=0 and local_rank == 0, send the local GPU string to world_rank==0 and return empty string
     * If world_rank==0, receive GPU strings and assemble the full remote GPU string to be returned
     */
    std::string assembleGPUsString();
    /**
     * Common function for handling local errors during MPI execution
     * Needs the set of in-use devices, not the config specified list of devices
     */
    void retrieveLocalErrorDetail(std::mutex &log_export_queue_mutex,
        std::multimap<int, AbstractSimRunner::ErrorDetail> &err_detail,
        std::vector<AbstractSimRunner::ErrorDetail> &err_detail_local, int i, std::set<int> devices);
    /**
     * Create the split MPI Communicator based on if the thread is participating in ensemble execution or not, based on the group rank and number of local GPUs.
     * @param isParticipating If this rank is participating (i.e. it has a local device assigned)
     * @return success of this method
     */
    bool createParticipatingCommunicator(bool isParticipating);
    /**
     * Accessor method for if the rank is participating or not (i.e. the colour of the communicator split)
     */
    int getRankIsParticipating() { return this->rank_is_participating; }
    /**
     * Accessor method for the size of the MPI communicator containing "participating" (or non-participating) ranks
     */
    int getParticipatingCommSize() { return this->participating_size; }
    /**
     * Accessor method for the rank within the MPI communicator containing "participating" (or non-participating) ranks
     */
    int getParticipatingCommRank() { return this->participating_rank; }

    /**
     * Static method to select devices for the current mpi rank, based on the provided list of devices.
     * This static version exists so that it is testable. 
     * A non static version which queries the curernt mpi environment is also provided as a simpler interface
     * @param devicesToSelectFrom set of device indices to use, provided from the config or initialised to contain all visible devices. 
     * @param local_size the number of mpi processes on the current shared memory system
     * @param local_rank the current process' mpi rank within the shared memory system
     * @return the gpus to be used by the current mpi rank, which may be empty.
     */
    static std::set<int> devicesForThisRank(std::set<int> devicesToSelectFrom, int local_size, int local_rank);

    /**
     * Method to select devices for the current mpi rank, based on the provided list of devices.
     * This non-static version calls the other overload with the current mpi size/ranks, i.e. this is the version that should be used.
     * @param devicesToSelectFrom set of device indices to use, provided from the config or initialised to contain all visible devices. 
     * @return the gpus to be used by the current mpi rank, which may be empty.
     */
    std::set<int> devicesForThisRank(std::set<int> devicesToSelectFrom);

 private:
    /**
     * @return Retrieve the local world rank from MPI
     */
    static int queryMPIWorldRank();
    /**
    * @return Retrieve the world size from MPI
    */
    static int queryMPIWorldSize();
    /**
     * @return Retrieve the local rank within the current shared memory region
     */
    static int queryMPISharedGroupRank();
    /**
     * @return retrieve the number of mpi processes on the current shared memory region
     */
    static int queryMPISharedGroupSize();
    /**
     * If necessary initialise MPI, else do nothing
     */
    static void initMPI();
    /**
     * Iterate the provided set of devices to find the item at index j
     * This doesn't use the config.devices as it may have been mutated based on the number of mpi ranks used.
     */
    unsigned int getDeviceIndex(const int j, std::set<int> devices);
    /**
     * MPI representation of AbstractSimRunner::ErrorDetail type
     */
    const MPI_Datatype MPI_ERROR_DETAIL;
    /**
     * flag indicating if the current MPI rank is a participating rank (i.e. it has atleast one GPU it can use).
     * This is not a const public member, as it can only be computed after world and local rank / sizes and local gpu count are known.
     * This is used to form the colour in the participating communicator 
     */
    bool rank_is_participating;
    /**
     * An MPI communicator, split by whether the mpi rank is participating in simulation or not.
     * non-participating ranks will have a commincator which only contains non-participating ranks, but these will never use the communicator
     */
    MPI_Comm comm_participating;
    /**
     * The size of the MPI communicator containing "participating" (or non-participating) ranks
     */
    int participating_size;
    /**
     * The rank within the MPI communicator containing "participating" (or non-participating) ranks
     */
    int participating_rank;
};
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_MPIENSEMBLE_H_
