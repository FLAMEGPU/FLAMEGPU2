#include "flamegpu/gpu/CUDAEnsemble.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <thread>
#include <set>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <map>

#include "flamegpu/version.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/sim/RunPlanVector.h"
#include "flamegpu/util/detail/compute_capability.cuh"
#include "flamegpu/util/detail/SteadyClockTimer.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/io/StateWriterFactory.h"
#include "flamegpu/sim/LoggingConfig.h"
#include "flamegpu/sim/SimRunner.h"
#include "flamegpu/sim/LogFrame.h"
#include "flamegpu/sim/SimLogger.h"
#include "flamegpu/util/detail/cuda.cuh"
#include "flamegpu/io/Telemetry.h"

namespace flamegpu {

CUDAEnsemble::EnsembleConfig::EnsembleConfig(): telemetry(flamegpu::io::Telemetry::globalTelemetryEnabled()) {}


CUDAEnsemble::CUDAEnsemble(const ModelDescription& _model, int argc, const char** argv)
    : model(_model.model->clone()) {
    initialise(argc, argv);
}
CUDAEnsemble::~CUDAEnsemble() {
// Call this here incase simulate() exited with an exception
#ifdef _MSC_VER
    if (config.block_standby) {
        // Disable prevention of standby
        SetThreadExecutionState(ES_CONTINUOUS);
    }
#endif
}



unsigned int CUDAEnsemble::simulate(const RunPlanVector &plans) {
#ifdef _MSC_VER
    if (config.block_standby) {
        // This thread requires the system continuously until it exits
        SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
    }
#endif
    // Validate that RunPlan model matches CUDAEnsemble model
    if (*plans.environment != this->model->environment->properties) {
        THROW exception::InvalidArgument("RunPlan is for a different ModelDescription, in CUDAEnsemble::simulate()");
    }
    // Validate/init output directories
    if (!config.out_directory.empty()) {
        // Validate out format is right
        config.out_format = io::StateWriterFactory::detectSupportedFileExt(config.out_format);
        if (config.out_format.empty()) {
            THROW exception::InvalidArgument("The out_directory config option also requires the out_format options to be set to a suitable type (e.g. 'json', 'xml'), in CUDAEnsemble::simulate()");
        }
        // Check that output files don't already exist
        if (std::filesystem::exists(config.out_directory)) {
            std::set<std::filesystem::path> exit_files;
            for (unsigned int p = 0; p < plans.size(); ++p) {
                std::filesystem::path exit_path = config.out_directory;
                if (!plans[p].getOutputSubdirectory().empty())
                    exit_path /= std::filesystem::path(plans[p].getOutputSubdirectory());
                exit_path /= std::filesystem::path("exit." + config.out_format);
                exit_files.insert(exit_path);
            }
            if (!config.truncate_log_files) {
                // Step
                for (unsigned int p = 0; p < plans.size(); ++p) {
                    std::filesystem::path step_path = config.out_directory;
                    if (!plans[p].getOutputSubdirectory().empty())
                        step_path /= std::filesystem::path(plans[p].getOutputSubdirectory());
                    step_path /= std::filesystem::path(std::to_string(p) + "." + config.out_format);
                    if (std::filesystem::exists(step_path)) {
                        THROW exception::FileAlreadyExists("Step log file '%s' already exists, in CUDAEnsemble::simulate()", step_path.c_str());
                    }
                }
                // Exit
                for (const auto &exit_path : exit_files) {
                    if (std::filesystem::exists(exit_path)) {
                        THROW exception::FileAlreadyExists("Exit log file '%s' already exists, in CUDAEnsemble::simulate()", exit_path.c_str());
                    }
                }
            } else {
                // Delete pre-existing exit log files
                for (const auto& exit_path : exit_files) {
                    std::filesystem::remove(exit_path);  // Returns false if the file didn't exist
                }
            }
        }
        // Create any missing directories
        try {
            std::filesystem::create_directories(config.out_directory);
        } catch (const std::exception &e) {
            THROW exception::InvalidArgument("Unable to use output directory '%s', in CUDAEnsemble::simulate(): %s", config.out_directory.c_str(), e.what());
        }
        for (const auto &p : plans) {
            const auto subdir = p.getOutputSubdirectory();
            if (!subdir.empty()) {
                std::filesystem::path sub_path = config.out_directory;
                try {
                    sub_path.append(subdir);
                    std::filesystem::create_directories(sub_path);
                } catch (const std::exception &e) {
                    THROW exception::InvalidArgument("Unable to use output subdirectory '%s', in CUDAEnsemble::simulate(): %s", sub_path.generic_string().c_str(), e.what());
                }
            }
        }
    }
    // Purge run logs, and resize ready for new runs
    // Resize means we can setup logs during execution out of order, without risk of list being reallocated
    run_logs.clear();
    run_logs.resize(plans.size());
    // Workout how many devices and runner we will be executing
    int device_count = -1;
    cudaError_t cudaStatus = cudaGetDeviceCount(&device_count);
    if (cudaStatus != cudaSuccess) {
        THROW exception::InvalidCUDAdevice("Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?, in CUDAEnsemble::simulate()");
    }
    if (device_count == 0) {
        THROW exception::InvalidCUDAdevice("Error no CUDA devices found!, in CUDAEnsemble::simulate()");
    }
    for (const int id : config.devices) {
        if (id >= device_count) {
            THROW exception::InvalidCUDAdevice("Requested CUDA device %d is not valid, only %d CUDA devices available!, in CUDAEnsemble::simulate()", id, device_count);
        }
    }

    std::set<int> devices;
    if (config.devices.size()) {
        devices = config.devices;
    } else {
        for (int i = 0; i < device_count; ++i) {
            devices.emplace(i);
        }
    }
    // Check that each device is capable, and init cuda context
    for (auto d = devices.begin(); d != devices.end(); ++d) {
        if (!util::detail::compute_capability::checkComputeCapability(*d)) {
            fprintf(stderr, "FLAMEGPU2 has not been built with an appropriate compute capability for device %d, this device will not be used.\n", *d);
            d = devices.erase(d);
            --d;
        } else {
            gpuErrchk(cudaSetDevice(*d));
            gpuErrchk(flamegpu::util::detail::cuda::cudaFree(nullptr));
        }
    }
    // Return to device 0 (or check original device first?)
    gpuErrchk(cudaSetDevice(0));

    // Init runners, devices * concurrent runs
    std::atomic<unsigned int> err_ct = {0};
    std::atomic<unsigned int> next_run = {0};
    const unsigned int TOTAL_RUNNERS = static_cast<unsigned int>(devices.size()) * config.concurrent_runs;
    SimRunner *runners = static_cast<SimRunner *>(malloc(sizeof(SimRunner) * TOTAL_RUNNERS));

    // Log Time (We can't use CUDA events here, due to device resets)
    auto ensemble_timer = util::detail::SteadyClockTimer();
    ensemble_timer.start();
    // Reset the elapsed time.
    ensemble_elapsed_time = 0.;

    // Logging thread-safety items
    std::queue<unsigned int> log_export_queue;
    std::mutex log_export_queue_mutex;
    std::condition_variable log_export_queue_cdn;
    SimRunner::ErrorDetail fast_err_detail = {};

    // Init with placement new
    {
        unsigned int i = 0;
        for (auto &d : devices) {
            for (unsigned int j = 0; j < config.concurrent_runs; ++j) {
                new (&runners[i++]) SimRunner(model, err_ct, next_run, plans,
                    step_log_config, exit_log_config,
                    d, j,
                    config.verbosity, config.error_level == EnsembleConfig::Fast,
                    run_logs, log_export_queue, log_export_queue_mutex, log_export_queue_cdn, fast_err_detail, TOTAL_RUNNERS);
            }
        }
    }

    // Init log worker
    SimLogger *log_worker = nullptr;
    if (!config.out_directory.empty()) {
        log_worker = new SimLogger(run_logs, plans, config.out_directory, config.out_format, log_export_queue, log_export_queue_mutex, log_export_queue_cdn,
        step_log_config.get(), exit_log_config.get(), step_log_config && step_log_config->log_timing, exit_log_config && exit_log_config->log_timing);
    }

    // Wait for all runners to exit
    for (unsigned int i = 0; i < TOTAL_RUNNERS; ++i) {
        runners[i].thread.join();
        runners[i].~SimRunner();
    }
    // Notify logger to exit
    if (log_worker) {
        {
            std::lock_guard<std::mutex> lck(log_export_queue_mutex);
            log_export_queue.push(UINT_MAX);
        }
        log_export_queue_cdn.notify_one();
        log_worker->thread.join();
        delete log_worker;
        log_worker = nullptr;
    }

    // Record and store the elapsed time
    ensemble_timer.stop();
    ensemble_elapsed_time = ensemble_timer.getElapsedSeconds();

    // Ensemble has finished, print summary
    if (config.verbosity > Verbosity::Quiet) {
        printf("\rCUDAEnsemble completed %u runs successfully!\n", static_cast<unsigned int>(plans.size() - err_ct));
        if (err_ct)
            printf("There were a total of %u errors.\n", err_ct.load());
    }
    if (config.timing || config.verbosity >= Verbosity::Verbose) {
        printf("Ensemble time elapsed: %fs\n", ensemble_elapsed_time);
    }

    // Send Telemetry
    if (config.telemetry) {
        // Generate some payload items
        std::map<std::string, std::string> payload_items;
        payload_items["GPUDevices"] = flamegpu::util::detail::compute_capability::getDeviceNames(config.devices);
        payload_items["SimTime(s)"] = std::to_string(ensemble_elapsed_time);
        #if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_PATCH__)
            payload_items["NVCCVersion"] = std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
        #endif
        // generate telemetry data
        std::string telemetry_data = flamegpu::io::Telemetry::generateTelemetryData("ensemble-run", payload_items);
        // send
        if (!flamegpu::io::Telemetry::sendTelemetryData(telemetry_data)) {
            if ((config.verbosity > Verbosity::Quiet))
                fprintf(stderr, "Warning: Usage statistics for CUDAEnsemble failed to send.\n");
        }
        // print
        if ((config.verbosity >= Verbosity::Verbose)) {
            fprintf(stdout, "Telemetry packet sent to '%s' json was: %s\n", flamegpu::io::Telemetry::TELEMETRY_ENDPOINT, telemetry_data.c_str());
        }
    } else {
        // Occasional hinting of telemetry if not in use (and not Quiet and not testing mode)
        if ((config.verbosity > Verbosity::Quiet))
            flamegpu::io::Telemetry::hintTelemetryUsage();
    }

    // Free memory
    free(runners);

    if (config.error_level == EnsembleConfig::Fast && err_ct.load()) {
        THROW exception::EnsembleError("Run %u failed on device %d, thread %u with exception: \n%s\n",
            fast_err_detail.run_id, fast_err_detail.device_id, fast_err_detail.runner_id, fast_err_detail.exception_string.c_str());
    } else if (config.error_level == EnsembleConfig::Slow && err_ct.load()) {
        THROW exception::EnsembleError("%u/%u runs failed!\n.", err_ct.load(), static_cast<unsigned int>(plans.size()));
    }
#ifdef _MSC_VER
    if (config.block_standby) {
        // Disable prevention of standby
        SetThreadExecutionState(ES_CONTINUOUS);
    }
#endif

    return err_ct.load();
}

void CUDAEnsemble::initialise(int argc, const char** argv) {
    if (!checkArgs(argc, argv)) {
        exit(EXIT_FAILURE);
    }
    // If verbose, output the flamegpu version and seed.
    if (config.verbosity == Verbosity::Verbose) {
        fprintf(stdout, "FLAME GPU %s\n", flamegpu::VERSION_FULL);
        fprintf(stdout, "Ensemble configuration:\n");
        fprintf(stdout, "\tConcurrent runs: %u\n", config.concurrent_runs);
    }
}
int CUDAEnsemble::checkArgs(int argc, const char** argv) {
    // Parse optional args
    int i = 1;
    for (; i < argc; i++) {
        // Get arg as lowercase
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // -h/--help. Print the help output and exit.
        if (arg.compare("--help") == 0 || arg.compare("-h") == 0) {
            printHelp(argv[0]);
            return false;
        }
        // --concurrent <runs>, Number of concurrent simulations to run per device
        if (arg.compare("--concurrent") == 0 || arg.compare("-c") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            config.concurrent_runs = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // --devices <string>, comma separated list of uints
        if (arg.compare("--devices") == 0 || arg.compare("-d") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            // Split and parse string
            std::string device_string = argv[++i];
            device_string += ",";  // Append comma, to catch final item
            int max_id = 0;  // Catch max device so we can validate it exists
            size_t pos;
            while ((pos = device_string.find(",")) != std::string::npos) {
                const unsigned int id = static_cast<unsigned int>(strtoul(device_string.substr(0, pos).c_str(), nullptr, 0));
                if (id == 0 && (device_string.length() < 2 || (device_string[0] != '0' || device_string[1] != ','))) {
                    fprintf(stderr, "'%s' is not a valid device index.\n", device_string.substr(0, pos).c_str());
                    printHelp(argv[0]);
                    return false;
                }
                max_id = static_cast<int>(id) > max_id ? id : max_id;
                config.devices.emplace(id);
                device_string.erase(0, pos + 1);
            }
            int ct = -1;
            gpuErrchk(cudaGetDeviceCount(&ct));
            if (max_id >= ct) {
                fprintf(stderr, "Device id %u exceeds available CUDA devices %d\n", max_id, ct);
                printHelp(argv[0]);
                return false;
            }
            continue;
        }
        // -o/--out <directory> <filetype>, Quiet FLAME GPU output.
        if (arg.compare("--out") == 0 || arg.compare("-o") == 0) {
            if (i + 2 >= argc) {
                fprintf(stderr, "%s requires two trailing arguments\n", arg.c_str());
                return false;
            }
            // Validate output directory is valid (and recursively create it if necessary)
            try {
                std::filesystem::path out_directory = argv[++i];
                std::filesystem::create_directories(out_directory);
                config.out_directory = out_directory.generic_string();
            } catch (const std::exception &e) {
                // Catch any exceptions, probably std::filesystem::filesystem_error, but other implementation defined errors also possible
                fprintf(stderr, "Unable to use '%s' as output directory:\n%s\n", argv[i], e.what());
                return false;
            }
            // Validate output format is available in io module
            config.out_format = io::StateWriterFactory::detectSupportedFileExt(argv[++i]);
            if (config.out_format.empty()) {
                fprintf(stderr, "'%s' is not a supported output file type.\n", argv[i]);
                return false;
            }
            continue;
        }
        // -q/--quiet, Don't report progress to console.
        if (arg.compare("--quiet") == 0 || arg.compare("-q") == 0) {
            config.verbosity = Verbosity::Quiet;
            continue;
        }
        // -v/--verbose, Report all progress to console.
        if (arg.compare("--verbose") == 0 || arg.compare("-v") == 0) {
            config.verbosity = Verbosity::Verbose;
            continue;
        }
        // -t/--timing, Output timing information to stdout
        if (arg.compare("--timing") == 0 || arg.compare("-t") == 0) {
            config.timing = true;
            continue;
        }
        // -u/--silence-unknown-args, Silence warning for unknown arguments
        if (arg.compare("--silence-unknown-args") == 0 || arg.compare("-u") == 0) {
            config.silence_unknown_args = true;
            continue;
        }
        // -e/--error, Specify the error level
        if (arg.compare("--error") == 0 || arg.compare("-e") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            std::string error_level_string = argv[++i];
            // Shift the trailing arg to lower
            std::transform(error_level_string.begin(), error_level_string.end(), error_level_string.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
            if (error_level_string.compare("off") == 0 || error_level_string.compare(std::to_string(EnsembleConfig::Off)) == 0) {
                config.error_level = EnsembleConfig::Off;
            } else if (error_level_string.compare("slow") == 0 || error_level_string.compare(std::to_string(EnsembleConfig::Slow)) == 0) {
                config.error_level = EnsembleConfig::Slow;
            } else if (error_level_string.compare("fast") == 0 || error_level_string.compare(std::to_string(EnsembleConfig::Fast)) == 0) {
                config.error_level = EnsembleConfig::Fast;
            } else {
                fprintf(stderr, "%s is not an appropriate argument for %s\n", error_level_string.c_str(), arg.c_str());
                return false;
            }
            continue;
        }
        // --truncate, Truncate output files
        if (arg.compare("--truncate") == 0) {
            config.truncate_log_files = true;
            continue;
        }
        // --standby Disable the blocking of standby
        if (arg.compare("--standby") == 0) {
#ifdef _MSC_VER
            config.block_standby = false;
#endif
            continue;
        }
        // Warning if not in QUIET verbosity or if silnce-unknown-args is set
        if (!(config.verbosity == flamegpu::Verbosity::Quiet || config.silence_unknown_args))
            fprintf(stderr, "Warning: Unknown argument '%s' passed to Ensemble will be ignored\n", arg.c_str());
    }
    return true;
}
void CUDAEnsemble::printHelp(const char *executable) {
    printf("FLAME GPU %s\n", flamegpu::VERSION_FULL);
    printf("Usage: %s [optional arguments]\n", executable);
    printf("Optional Arguments:\n");
    const char *line_fmt = "%-18s %s\n";
    printf(line_fmt, "-h, --help", "show this help message and exit");
    printf(line_fmt, "-d, --devices <device ids>", "Comma separated list of device ids to be used");
    printf(line_fmt, "", "By default, all available devices will be used.");
    printf(line_fmt, "-c, --concurrent <runs>", "Number of concurrent simulations to run per device");
    printf(line_fmt, "", "By default, 4 will be used.");
    printf(line_fmt, "-o, --out <directory> <filetype>", "Directory and filetype for ensemble outputs");
    printf(line_fmt, "-q, --quiet", "Do not print progress information to console");
    printf(line_fmt, "-v, --verbose", "Print config, progress and timing (-t) information to console");
    printf(line_fmt, "-t, --timing", "Output timing information to stdout");
    printf(line_fmt, "-e, --error <error level>", "The error level 0, 1, 2, off, slow or fast");
    printf(line_fmt, "-u, --silence-unknown-args", "Silence warnings for unknown arguments passed after this flag.");
#ifdef _MSC_VER
    printf(line_fmt, "    --standby", "Allow the machine to enter standby during execution");
#endif
    printf(line_fmt, "", "By default, \"slow\" will be used.");
}
void CUDAEnsemble::setStepLog(const StepLoggingConfig &stepConfig) {
    // Validate ModelDescription matches
    if (*stepConfig.model != *model) {
      THROW exception::InvalidArgument("Model descriptions attached to LoggingConfig and CUDAEnsemble do not match, in CUDAEnsemble::setStepLog()\n");
    }
    // Set internal config
    step_log_config = std::make_shared<StepLoggingConfig>(stepConfig);
}
void CUDAEnsemble::setExitLog(const LoggingConfig &exitConfig) {
    // Validate ModelDescription matches
    if (*exitConfig.model != *model) {
      THROW exception::InvalidArgument("Model descriptions attached to LoggingConfig and CUDAEnsemble do not match, in CUDAEnsemble::setExitLog()\n");
    }
    // Set internal config
    exit_log_config = std::make_shared<LoggingConfig>(exitConfig);
}
const std::vector<RunLog> &CUDAEnsemble::getLogs() {
    return run_logs;
}

void CUDAEnsemble::shareUsageStatistics(bool telemetry_enabled) {
    config.telemetry = telemetry_enabled;
}

}  // namespace flamegpu
