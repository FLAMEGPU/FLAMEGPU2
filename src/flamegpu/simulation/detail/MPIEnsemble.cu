#ifdef FLAMEGPU_ENABLE_MPI
#include "flamegpu/simulation/detail/MPIEnsemble.h"

#include "flamegpu/detail/compute_capability.cuh"

namespace flamegpu {
namespace detail {

MPIEnsemble::MPIEnsemble(const CUDAEnsemble::EnsembleConfig &_config)
    : MPI_ERROR_DETAIL(AbstractSimRunner::createErrorDetailMPIDatatype())
    , config(_config)
    , world_rank(getWorldRank())
    , world_size(getWorldSize()) { }

int MPIEnsemble::receiveErrors(std::multimap<int, AbstractSimRunner::ErrorDetail> &err_detail, const unsigned int total_runs) {
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
int MPIEnsemble::receiveJobRequests(unsigned int &next_run, const unsigned int total_runs) {
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
    // All ranks should notify rank 0 of their GPU devices
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
            remote_device_names.append(", ");
            remote_device_names.append(buff);
        }
        free(buff);
    } else {
        const std::string d_string = compute_capability::getDeviceNames(config.devices);
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

int MPIEnsemble::getWorldRank() {
    initMPI();
    int world_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return world_rank;
}
int MPIEnsemble::getWorldSize() {
    initMPI();
    int world_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
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

}  // namespace detail
}  // namespace flamegpu
#endif
