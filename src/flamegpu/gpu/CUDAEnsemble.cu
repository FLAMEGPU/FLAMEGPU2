#include "flamegpu/gpu/CUDAEnsemble.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <thread>
#include <set>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "flamegpu/version.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/sim/RunPlanVector.h"
#include "flamegpu/util/detail/compute_capability.cuh"
#include "flamegpu/util/detail/SteadyClockTimer.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/io/StateWriterFactory.h"
#include "flamegpu/util/detail/filesystem.h"
#include "flamegpu/sim/LoggingConfig.h"
#include "flamegpu/sim/SimRunner.h"
#include "flamegpu/sim/LogFrame.h"
#include "flamegpu/sim/SimLogger.h"

namespace flamegpu {


CUDAEnsemble::CUDAEnsemble(const ModelDescription& _model, int argc, const char** argv)
    : model(_model.model->clone()) {
    initialise(argc, argv);
}
CUDAEnsemble::~CUDAEnsemble() {
    // Nothing to do
}



void CUDAEnsemble::simulate(const RunPlanVector &plans) {
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
        // Create any missing directories
        try {
            util::detail::filesystem::recursive_create_dir(config.out_directory);
        } catch (const std::exception &e) {
            THROW exception::InvalidArgument("Unable to use output directory '%s', in CUDAEnsemble::simulate(): %s", config.out_directory.c_str(), e.what());
        }
        for (const auto &p : plans) {
            const auto subdir = p.getOutputSubdirectory();
            if (!subdir.empty()) {
                path sub_path = config.out_directory;
                try {
                    sub_path.append(subdir);
                    util::detail::filesystem::recursive_create_dir(sub_path);
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
    int ct = -1;
    gpuErrchk(cudaGetDeviceCount(&ct));
    std::set<int> devices;
    if (config.devices.size()) {
        devices = config.devices;
    } else {
        for (int i = 0; i < ct; ++i) {
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
            gpuErrchk(cudaFree(nullptr));
        }
    }
    // Return to device 0 (or check original device first?)
    gpuErrchk(cudaSetDevice(0));

    // Init runners, devices * concurrent runs
    std::atomic<unsigned int> err_ct = {0};
    std::atomic<unsigned int> next_run = {0};
    const size_t TOTAL_RUNNERS = devices.size() * config.concurrent_runs;
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

    // Init with placement new
    {
        if (!config.quiet) {
            printf("\rCUDAEnsemble progress: %u/%u", 0, static_cast<unsigned int>(plans.size()));
            fflush(stdout);
        }
        unsigned int i = 0;
        for (auto &d : devices) {
            for (unsigned int j = 0; j < config.concurrent_runs; ++j) {
                new (&runners[i++]) SimRunner(model, err_ct, next_run, plans, step_log_config, exit_log_config, d, j, !config.quiet, run_logs, log_export_queue, log_export_queue_mutex, log_export_queue_cdn);
            }
        }
    }

    // Init log worker
    SimLogger *log_worker = nullptr;
    if (!config.out_directory.empty() && !config.out_format.empty()) {
        log_worker = new SimLogger(run_logs, plans, config.out_directory, config.out_format, log_export_queue, log_export_queue_mutex, log_export_queue_cdn,
        step_log_config.get(), exit_log_config.get(), step_log_config && step_log_config->log_timing, exit_log_config && exit_log_config->log_timing);
    } else if (!config.out_directory.empty() ^ !config.out_format.empty())  {
        fprintf(stderr, "Warning: Only 1 of out_directory and out_format is set, both must be set for logging to commence to file.\n");
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

    // Purge singletons on every used CUDA device
    for (auto d = devices.begin(); d != devices.end(); ++d) {
        gpuErrchk(cudaSetDevice(*d));
        CUDASimulation::purgeSingletons();
    }

    // Ensemble has finished, print summary
    if (!config.quiet) {
        printf("\rCUDAEnsemble completed %u runs successfully!\n", static_cast<unsigned int>(plans.size() - err_ct));
        if (err_ct)
            printf("There were a total of %u errors.\n", err_ct.load());
    }
    if (config.timing) {
        printf("Ensemble time elapsed: %fs\n", ensemble_elapsed_time);
    }

    // Free memory
    free(runners);
}

void CUDAEnsemble::initialise(int argc, const char** argv) {
    if (!checkArgs(argc, argv)) {
        exit(EXIT_FAILURE);
    }
    /* Disabled as this is printed prior to quiet being accessible 
    // If verbsoe, output the flamegpu version.
    if (!config.quiet) {
        fprintf(stdout, "FLAME GPU %s\n", flamegpu::VERSION_FULL);
    }
    */
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
                path out_directory = argv[++i];
                util::detail::filesystem::recursive_create_dir(out_directory);
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
            config.quiet = true;
            continue;
        }
        // -t/--timing, Output timing information to stdout
        if (arg.compare("--timing") == 0 || arg.compare("-t") == 0) {
            config.timing = true;
            continue;
        }
        fprintf(stderr, "Unexpected argument: %s\n", arg.c_str());
        printHelp(argv[0]);
        return false;
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
    printf(line_fmt, "-q, --quiet", "Don't print progress information to console");
    printf(line_fmt, "-t, --timing", "Output timing information to stdout");
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

}  // namespace flamegpu
