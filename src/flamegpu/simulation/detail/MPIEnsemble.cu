#ifdef FLAMEGPU_ENABLE_MPI
#include "flamegpu/simulation/detail/MPIEnsemble.h"

#include "flamegpu/detail/compute_capability.cuh"

namespace flamegpu {
namespace detail {

MPIEnsemble::MPIEnsemble(const CUDAEnsemble::EnsembleConfig &_config, const unsigned int _total_runs)
    : config(_config)
    , world_rank(queryMPIWorldRank())
    , world_size(queryMPIWorldSize())
    , local_rank(queryMPISharedGroupRank())
    , local_size(queryMPISharedGroupSize())
    , total_runs(_total_runs)
    , MPI_ERROR_DETAIL(AbstractSimRunner::createErrorDetailMPIDatatype())
    , rank_is_participating(false)
    , comm_participating(MPI_COMM_NULL)
    , participating_size(0)
    , participating_rank(-1) { }

int MPIEnsemble::receiveErrors(std::multimap<int, AbstractSimRunner::ErrorDetail> &err_detail) {
    int errCount = 0;
    if (world_rank == 0) {
        MPI_Status status;
        int flag;
        // Check whether MPI runners have reported an error
        MPI_Iprobe(
            MPI_ANY_SOURCE,            // int source
            EnvelopeTag::ReportError,  // int tag
            MPI_COMM_WORLD,            // MPI_Comm communicator
            &flag,                     // int flag
            &status);
        while (flag) {
            // Receive the message
            memset(&status, 0, sizeof(MPI_Status));
            AbstractSimRunner::ErrorDetail e_detail;
            memset(&e_detail, 0, sizeof(AbstractSimRunner::ErrorDetail));
            MPI_Recv(
                &e_detail,                 // void* data
                1,                         // int count
                MPI_ERROR_DETAIL,          // MPI_Datatype datatype (can't use MPI_DATATYPE_NULL)
                MPI_ANY_SOURCE,            // int source
                EnvelopeTag::ReportError,  // int tag
                MPI_COMM_WORLD,            // MPI_Comm communicator
                &status);                  // MPI_Status*
            err_detail.emplace(status.MPI_SOURCE, e_detail);
            ++errCount;
            // Progress flush
            if (config.verbosity >= Verbosity::Default && config.error_level != CUDAEnsemble::EnsembleConfig::Fast) {
                fprintf(stderr, "Warning: Run %u/%u failed on rank %d, device %d, thread %u with exception: \n%s\n",
                    e_detail.run_id + 1, total_runs, status.MPI_SOURCE, e_detail.device_id, e_detail.runner_id, e_detail.exception_string);
                fflush(stderr);
            }
            // Check again
            MPI_Iprobe(MPI_ANY_SOURCE, EnvelopeTag::ReportError, MPI_COMM_WORLD, &flag, &status);
        }
    }
    return errCount;
}
int MPIEnsemble::receiveJobRequests(unsigned int &next_run) {
    int mpi_runners_fin = 0;
    if (world_rank == 0) {
        MPI_Status status;
        int flag;
        MPI_Iprobe(
            MPI_ANY_SOURCE,           // int source
            EnvelopeTag::RequestJob,  // int tag
            MPI_COMM_WORLD,           // MPI_Comm communicator
            &flag,                    // int flag
            &status);                 // MPI_Status*
        while (flag) {
            // Receive the message (kind of redundant as we already have the status and it carrys no data)
            memset(&status, 0, sizeof(MPI_Status));
            MPI_Recv(
                nullptr,                  // void* data
                0,                        // int count
                MPI_CHAR,                 // MPI_Datatype datatype (can't use MPI_DATATYPE_NULL)
                MPI_ANY_SOURCE,           // int source
                EnvelopeTag::RequestJob,  // int tag
                MPI_COMM_WORLD,           // MPI_Comm communicator
                &status);                 // MPI_Status*
            // Respond to the sender with a job assignment
            MPI_Send(
                &next_run,               // void* data
                1,                       // int count
                MPI_UNSIGNED,            // MPI_Datatype datatype
                status.MPI_SOURCE,       // int destination
                EnvelopeTag::AssignJob,  // int tag
                MPI_COMM_WORLD);         // MPI_Comm communicator
            if (next_run >= total_runs) ++mpi_runners_fin;
            ++next_run;
            // Print progress to console
            if (config.verbosity >= Verbosity::Default && next_run <= total_runs) {
                fprintf(stdout, "MPI ensemble assigned run %d/%u to rank %d\n", next_run, total_runs, status.MPI_SOURCE);
                fflush(stdout);
            }
            // Check again
            MPI_Iprobe(MPI_ANY_SOURCE, EnvelopeTag::RequestJob, MPI_COMM_WORLD, &flag, &status);
        }
    }
    return mpi_runners_fin;
}
void MPIEnsemble::sendErrorDetail(AbstractSimRunner::ErrorDetail &e_detail) {
    if (world_rank != 0) {
      MPI_Send(
          &e_detail,                  // void* data
          1,                         // int count
          MPI_ERROR_DETAIL,          // MPI_Datatype datatype (can't use MPI_DATATYPE_NULL)
          0,                         // int destination
          EnvelopeTag::ReportError,  // int tag
          MPI_COMM_WORLD);           // MPI_Comm communicator
    }
}
int MPIEnsemble::requestJob() {
    unsigned int next_run = UINT_MAX;
    if (world_rank != 0) {
        // Send a job request to 0, these have no data
        MPI_Send(
            nullptr,                  // void* data
            0,                        // int count
            MPI_CHAR,                 // MPI_Datatype datatype (can't use MPI_DATATYPE_NULL)
            0,                        // int destination
            EnvelopeTag::RequestJob,  // int tag
            MPI_COMM_WORLD);          // MPI_Comm communicator
        // Wait for a job assignment from 0
        MPI_Status status;
        memset(&status, 0, sizeof(MPI_Status));
        MPI_Recv(
            &next_run,               // void* data
            1,                       // int count
            MPI_UNSIGNED,            // MPI_Datatype datatype
            0,                       // int source
            EnvelopeTag::AssignJob,  // int tag
            MPI_COMM_WORLD,          // MPI_Comm communicator
            &status);                // MPI_Status* status
    }
    return next_run;
}
void MPIEnsemble::worldBarrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}
std::string MPIEnsemble::assembleGPUsString() {
    std::string remote_device_names;
    // One rank per node should notify rank 0 of their GPU devices. other ranks will send an empty message.
    if (world_rank == 0) {
        int bufflen = 256;  // Length of name string in cudaDeviceProp
        char *buff = static_cast<char*>(malloc(bufflen));
        for (int i = 1; i < world_size; ++i) {
            // Receive a message from each rank
            MPI_Status status;
            memset(&status, 0, sizeof(MPI_Status));
            MPI_Probe(
                MPI_ANY_SOURCE,            // int source
                EnvelopeTag::TelemetryDevices,  // int tag
                MPI_COMM_WORLD,            // MPI_Comm communicator
                &status);
            int strlen = 0;
            // Ensure our receive buffer is long enough
            MPI_Get_count(&status, MPI_CHAR, &strlen);
            if (strlen > bufflen) {
                free(buff);
                buff = static_cast<char*>(malloc(strlen));
            }
            MPI_Recv(
                buff,                           // void* data
                strlen,                         // int count
                MPI_CHAR,                       // MPI_Datatype datatype (can't use MPI_DATATYPE_NULL)
                MPI_ANY_SOURCE,                 // int source
                EnvelopeTag::TelemetryDevices,  // int tag
                MPI_COMM_WORLD,                 // MPI_Comm communicator
                &status);                       // MPI_Status*
            if (strlen > 1) {
                remote_device_names.append(", ");
                remote_device_names.append(buff);
            }
        }
        free(buff);
    } else {
        const std::string d_string = local_rank == 0 ? compute_capability::getDeviceNames(config.devices) : "";
        // Send GPU count
        MPI_Send(
            d_string.c_str(),               // void* data
            d_string.length() + 1,          // int count
            MPI_CHAR,                       // MPI_Datatype datatype
            0,                              // int destination
            EnvelopeTag::TelemetryDevices,  // int tag
            MPI_COMM_WORLD);                // MPI_Comm communicator
    }
    worldBarrier();
    return remote_device_names;
}

int MPIEnsemble::queryMPIWorldRank() {
    initMPI();
    int world_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return world_rank;
}
int MPIEnsemble::queryMPIWorldSize() {
    initMPI();
    int world_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
}

int MPIEnsemble::queryMPISharedGroupRank() {
    initMPI();
    int local_rank = -1;
    MPI_Comm group;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &group);
    MPI_Comm_rank(group, &local_rank);
    return local_rank;
}

int MPIEnsemble::queryMPISharedGroupSize() {
    initMPI();
    int local_size = -1;
    MPI_Comm group;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &group);
    MPI_Comm_size(group, &local_size);
    return local_size;
}

void MPIEnsemble::initMPI() {
    int flag = 0;
    // MPI can only be init once, for certain test cases we do some initial MPI comms for setup
    MPI_Initialized(&flag);
    if (!flag) {
        // Init MPI, fetch rank and size
        int thread_provided = 0;
        // MPI single means that only the main thread will perform MPI actions
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &thread_provided);
        if (thread_provided != MPI_THREAD_SINGLE) {
            THROW exception::UnknownInternalError("MPI unable to provide MPI_THREAD_SINGLE support");
        }
    }
}
unsigned int MPIEnsemble::getDeviceIndex(const int j, const std::set<int> devices) {
    int i = 0;
    for (auto& d : devices) {
        if (i++ == j)
            return d;
    }
    return j;  // If set is empty, then direct index should be used
}
void MPIEnsemble::retrieveLocalErrorDetail(std::mutex &log_export_queue_mutex, std::multimap<int, AbstractSimRunner::ErrorDetail> &err_detail,
std::vector<detail::AbstractSimRunner::ErrorDetail> &err_detail_local, const int i, std::set<int> devices) {
    // Fetch error detail
    detail::AbstractSimRunner::ErrorDetail e_detail;
    {
        // log_export_mutex is treated as our protection for race conditions on err_detail
        std::lock_guard<std::mutex> lck(log_export_queue_mutex);
        // Fetch corresponding error detail
        bool success = false;
        const unsigned int t_device_id = getDeviceIndex(i / config.concurrent_runs, devices);
        const unsigned int t_runner_id = i % config.concurrent_runs;
        for (auto it = err_detail_local.begin(); it != err_detail_local.end(); ++it) {
            if (it->runner_id == t_runner_id && it->device_id == t_device_id) {
                e_detail = *it;
                if (world_rank == 0) {
                    // Only rank 0 collects error details
                    err_detail.emplace(world_rank, e_detail);
                } else {
                //   fprintf(stderr, "[%d] Purged error  from device %u runner %u\n", world_rank, t_device_id, t_runner_id);  // useful debug, breaks tests
                }
                err_detail_local.erase(it);
                success = true;
                break;
            }
        }
        if (!success) {
            THROW exception::UnknownInternalError("[%d] Management thread failed to locate reported error from device %u runner %u from %u errors, in CUDAEnsemble::simulate()", world_rank, t_device_id, t_runner_id, static_cast<unsigned int>(err_detail_local.size()));
        }
    }
    if (world_rank == 0) {
        // Progress flush
        if (config.verbosity >= Verbosity::Default && config.error_level != CUDAEnsemble::EnsembleConfig::Fast) {
            fprintf(stderr, "Warning: Run %u/%u failed on rank %d, device %d, thread %u with exception: \n%s\n",
                e_detail.run_id + 1, total_runs, world_rank, e_detail.device_id, e_detail.runner_id, e_detail.exception_string);
            fflush(stderr);
        }
    } else {
        // Notify 0 that an error occurred, with the error detail
        sendErrorDetail(e_detail);
    }
}

bool MPIEnsemble::createParticipatingCommunicator(const bool isParticipating) {
    // If the communicator has not yet been created, create it and get the rank and size.
    if (this->comm_participating == MPI_COMM_NULL) {
        // determine if this thread is participating or not, i..e. the colour of the rank
        this->rank_is_participating = isParticipating;
        // Split the world  communicator, if the split fails, abort (this makes the return type not useful tbh.)
        if (MPI_Comm_split(MPI_COMM_WORLD, this->rank_is_participating, this->world_rank, &this->comm_participating) != MPI_SUCCESS) {
            fprintf(stderr, "Error creating communicator\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return false;
        }
        // Get the size of the split pariticpating communicator
        MPI_Comm_size(this->comm_participating, &this->participating_size);
        // Get the local rank within the split communicator
        MPI_Comm_rank(this->comm_participating, &this->participating_rank);
    }
    return true;
}

std::set<int> MPIEnsemble::devicesForThisRank(const std::set<int> devicesToSelectFrom, int local_size, int local_rank) {
    // create a vector from teh set to enable direct access.
    std::vector<int> devicesToSelectFromVector = std::vector<int>(devicesToSelectFrom.begin(), devicesToSelectFrom.end());
    int device_count = static_cast<int>(devicesToSelectFrom.size());
    // if there is only a single mpi rank on this shared memory system, assign all devices, or if there are no devices to select from
    if (local_size == 1 || device_count == 0) {
        return devicesToSelectFrom;
    } else if (local_size > 1 && local_size <= device_count) {
        // Otherwise, if there are more than one rank per node, but fewer ranks than gpus, attempt to load balance
        std::set<int> devices;
        // find the balanced number of gpus per rank, and how many will need +1
        int gpusPerRank = device_count / local_size;
        int unallocated = device_count - (gpusPerRank * local_size);
        // Compute the indices of the first and last gpu to be assigned to the current rank, based on how many lower ranks will have +1
        int lowerRanksWithPlus1 = local_rank < unallocated ? local_rank : unallocated;
        int lowerRanksWithPlus0 = std::max(0, local_rank - unallocated);
        int first = (lowerRanksWithPlus1 * (gpusPerRank + 1)) + (lowerRanksWithPlus0 * gpusPerRank);
        int last = local_rank < unallocated ? first + gpusPerRank + 1 : first + gpusPerRank;
        // Assign the devices for this rank
        for (int i = first; i < last; i++) {
            devices.emplace(devicesToSelectFromVector[i]);
        }
        return devices;
    } else {
        // Otherwise, there are more ranks than gpus, so use upto one gpu per rank.
        std::set<int> devices;
        for (const auto & d : devicesToSelectFromVector) {}
        if (local_rank < device_count) {
            devices.emplace(local_rank);
        }
        return devices;
    }
}

std::set<int> MPIEnsemble::devicesForThisRank(const std::set<int> devicesToSelectFrom) {
    return MPIEnsemble::devicesForThisRank(devicesToSelectFrom, this->local_size, this->local_rank);
}

}  // namespace detail
}  // namespace flamegpu
#endif
