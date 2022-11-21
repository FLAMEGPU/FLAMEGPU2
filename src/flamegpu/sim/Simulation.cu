#include "flamegpu/sim/Simulation.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>  // For PRIu64

#include "flamegpu/version.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/io/XMLStateWriter.h"
#include "flamegpu/io/StateReaderFactory.h"
#include "flamegpu/io/StateWriterFactory.h"
#include "flamegpu/io/LoggerFactory.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"
#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/model/EnvironmentData.h"
#include "flamegpu/io/Telemetry.h"

namespace flamegpu {

Simulation::Config::Config()
    : random_seed(static_cast<uint64_t>(time(nullptr)))
    , telemetry(flamegpu::io::Telemetry::isEnabled()) { }

Simulation::Simulation(const std::shared_ptr<const ModelData> &_model)
    : model(_model->clone())
    , submodel(nullptr)
    , mastermodel(nullptr)
    , config({})
    , instance_id(get_instance_id())
    , maxLayerWidth((*model).getMaxLayerWidth()) { }

Simulation::Simulation(const std::shared_ptr<SubModelData> &submodel_desc, CUDASimulation *master_model)
    : model(submodel_desc->submodel)
    , submodel(submodel_desc)
    , mastermodel(master_model)
    , config({})
    , instance_id(get_instance_id())
    , maxLayerWidth(submodel_desc->submodel->getMaxLayerWidth()) { }

void Simulation::initialise(int argc, const char** argv) {
    flamegpu::util::nvtx::Range range{"Simulation::initialise"};
    // check input args
    if (argc)
        if (!checkArgs(argc, argv))
            exit(EXIT_FAILURE);
    applyConfig();
}

void Simulation::applyConfig() {
    if (!config.input_file.empty() && config.input_file != loaded_input_file) {
        const std::string current_input_file = config.input_file;
        // Build population vector
        util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> pops;
        for (auto &agent : model->agents) {
            for (const auto &state : agent.second->states) {
                pops.emplace(util::StringPair{ agent.first, state }, std::make_shared<AgentVector>(*agent.second));
            }
        }

        env_init.clear();
        io::StateReader *read__ = io::StateReaderFactory::createReader(model->name, model->environment->properties, env_init, pops, config.input_file.c_str(), this);
        if (read__) {
            read__->parse();
            for (auto &agent : pops) {
                setPopulationData(*agent.second, agent.first.second);
            }
        }
        // Reset input file (we don't support input file recursion)
        config.input_file = current_input_file;
        // Set flag so we don't reload this in future
        loaded_input_file = current_input_file;
    }
    // Create directory for log files
    if (!config.step_log_file.empty()) {
        std::filesystem::path t_path = config.step_log_file;
        try {
            t_path = t_path.parent_path();
            if (!t_path.empty()) {
                std::filesystem::create_directories(t_path);
            }
        } catch(std::exception &e) {
            THROW exception::InvalidArgument("Failed to create step log file directory '%s': %s\n", t_path.c_str(), e.what());
        }
    }
    if (!config.exit_log_file.empty()) {
        std::filesystem::path t_path = config.exit_log_file;
        try {
            t_path = t_path.parent_path();
            if (!t_path.empty()) {
                std::filesystem::create_directories(t_path);
            }
        } catch(std::exception &e) {
            THROW exception::InvalidArgument("Failed to create exit log file directory: '%s': %s\n", t_path.c_str(), e.what());
        }
    }
    if (!config.common_log_file.empty()) {
        std::filesystem::path t_path = config.common_log_file;
        try {
            t_path = t_path.parent_path();
            if (!t_path.empty()) {
                std::filesystem::create_directories(t_path);
            }
        } catch (std::exception& e) {
            THROW exception::InvalidArgument("Failed to create common log file directory: '%s': %s\n", t_path.c_str(), e.what());
        }
    }
    // If verbose, output the flamegpu version and seed.
    if (config.verbosity == Verbosity::Verbose) {
        fprintf(stdout, "FLAME GPU %s\n", flamegpu::VERSION_FULL);
        fprintf(stdout, "Simulation configuration:\n");
        fprintf(stdout, "\tRandom Seed: %" PRIu64 "\n", config.random_seed);
        fprintf(stdout, "\tSteps: %u\n", config.steps);
    }
    // Call derived class config stuff first
    applyConfig_derived();
    // Random is handled by derived class, as it relies on singletons being init
}

const ModelData& Simulation::getModelDescription() const {
    return *model;
}

/*
 * issues: only saves the last output, hardcoded, will be changed
 */
void Simulation::exportData(const std::string &path, bool prettyPrint) {
    if (!config.truncate_log_files && std::filesystem::exists(path)) {
        THROW exception::FileAlreadyExists("File '%s' already exists, in Simulation::exportData()", path.c_str());
    }
    // Build population vector
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> pops;
    for (auto &agent : model->agents) {
        for (auto &state : agent.second->states) {
            auto a = std::make_shared<AgentVector>(*agent.second);
            getPopulationData(*a, state);
            pops.emplace(util::StringPair{agent.first, state}, a);
        }
    }

    io::StateWriter *write__ = io::StateWriterFactory::createWriter(model->name, getEnvironment(), pops, getStepCounter(), path, this);
    write__->writeStates(prettyPrint);
}
void Simulation::exportLog(const std::string &path, bool steps, bool exit, bool stepTime, bool exitTime, bool prettyPrint) {
    if (!config.truncate_log_files && std::filesystem::exists(path)) {
        THROW exception::FileAlreadyExists("Log file '%s' already exists, in Simulation::exportLog()", path.c_str());
    }
    // Create the correct type of logger
    auto logger = io::LoggerFactory::createLogger(path, prettyPrint, config.truncate_log_files);
    // Perform logging
    logger->log(getRunLog(), true, steps, exit, stepTime, exitTime);
}

int Simulation::checkArgs(int argc, const char** argv) {
    // Required args
    if (argc < 1) {
        printHelp(argv[0]);
        return false;
    }

    // First pass only looks for and handles input files
    // Remaining arguments can override args passed via input file
    // Any errors to stderr have return false and are expected to raise an exception
    int i = 1;
    for (; i < argc; i++) {
        // Get arg as lowercase
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // -in <string>, Specifies the input state file
        if (arg.compare("--in") == 0 || arg.compare("-i") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            const std::string new_input_file = std::string(argv[++i]);
            config.input_file = new_input_file;
            // Load the input file
            {
                // Build population vector
                util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> pops;
                for (auto &agent : model->agents) {
                    for (auto& state : agent.second->states) {
                        pops.emplace(util::StringPair{ agent.first, state }, std::make_shared<AgentVector>(*agent.second));
                    }
                }
                env_init.clear();
                const auto &env_desc = model->environment->properties;  // For some reason this method returns a copy, not a reference
                io::StateReader *read__ = io::StateReaderFactory::createReader(model->name, env_desc, env_init, pops, config.input_file.c_str(), this);
                if (read__) {
                    try {
                        read__->parse();
                    } catch (const std::exception &e) {
                        fprintf(stderr, "Loading input file '%s' failed!\nDetail: %s", config.input_file.c_str(), e.what());
                        return false;
                    }
                    for (auto &agent : pops) {
                        setPopulationData(*agent.second);
                    }
                }
            }
            // Reset input file (we don't support input file recursion)
            config.input_file = new_input_file;
            // Set flag so input file isn't reloaded via apply_config
            loaded_input_file = new_input_file;
            // Break, we have loaded an input file
            break;
        }
    }

    // Parse optional args
    i = 1;
    for (; i < argc; i++) {
        // Get arg as lowercase
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // -h/--help. Print the help output and exit.
        if (arg.compare("--help") == 0 || arg.compare("-h") == 0) {
            printHelp(argv[0]);
            return false;
        }
        // -in <string>, Specifies the input state file
        if (arg.compare("--in") == 0 || arg.compare("-i") == 0) {
            // We already processed input file above, skip here
            ++i;
            continue;
        }
        // -steps <uint>, The number of steps to be executed
        if (arg.compare("--steps") == 0 || arg.compare("-s") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            config.steps = static_cast<int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -random <uint>, Uses the specified random seed, defaults to clock
        if (arg.compare("--random") == 0 || arg.compare("-r") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            // Reinitialise RandomManager state
            config.random_seed = static_cast<uint64_t>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -v/--verbose, Verbose FLAME GPU output.
        if (arg.compare("--verbose") == 0 || arg.compare("-v") == 0) {
            config.verbosity = Verbosity::Verbose;
            continue;
        }
        // -q/--quiet, Verbose level quiet FLAME GPU output.
        if (arg.compare("--quiet") == 0 || arg.compare("-q") == 0) {
            config.verbosity = Verbosity::Quiet;
            continue;
        }
        // -t/--timing, Output timing information to stdout
        if (arg.compare("--timing") == 0 || arg.compare("-t") == 0) {
            config.timing = true;
            continue;
        }
        // -u/--silence-unknown-args, Silence unknown args
        if (arg.compare("--silence-unknown-args") == 0 || arg.compare("-u") == 0) {
            config.silence_unknown_args = true;
            continue;
        }
        // --out-step <file.xml/file.json>, Step log file path
        if (arg.compare("--out-step") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            config.step_log_file = argv[++i];
            continue;
        }
        // --out-exit <file.xml/file.json>, Exit log file path
        if (arg.compare("--out-exit") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            config.exit_log_file = argv[++i];
            continue;
        }
        // --out-log <file.xml/file.json>, Common log file path
        if (arg.compare("--out-log") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            config.common_log_file = argv[++i];
            continue;
        }
        // --truncate, Output log files will truncate (rather than throwing an exception)
        if (arg.compare("--truncate") == 0) {
            config.truncate_log_files = true;
            continue;
        }
#ifdef FLAMEGPU_VISUALISATION
        // -c/--console, Renders the visualisation inert
        if (arg.compare("--console") == 0 || arg.compare("-c") == 0) {
            config.console_mode = true;
            continue;
        }
#endif
        // Test this arg with the derived class
        if (checkArgs_derived(argc, argv, i)) {
            continue;
        }
        // Warning if not in QUIET verbosity or if silnce-unknown-args is set
        if (!(config.verbosity == flamegpu::Verbosity::Quiet || config.silence_unknown_args))
            fprintf(stderr, "Warning: Unknown argument '%s' passed to Simulation will be ignored\n", arg.c_str());
    }
    return true;
}

void Simulation::printHelp(const char* executable) {
    printf("FLAME GPU %s\n", flamegpu::VERSION_FULL);
    printf("Usage: %s [-s steps] [-d device_id] [-r random_seed]\n", executable);
    printf("Optional Arguments:\n");
    const char *line_fmt = "%-18s %s\n";
    printf(line_fmt, "-h, --help", "show this help message and exit");
    printf(line_fmt, "-i, --in <file.xml/file.json>", "Initial state file (XML or JSON)");
    printf(line_fmt, "    --out-step <file.xml/file.json>", "Step log file (XML or JSON)");
    printf(line_fmt, "    --out-exit <file.xml/file.json>", "Exit log file (XML or JSON)");
    printf(line_fmt, "    --out-log <file.xml/file.json>", "Common log file (XML or JSON)");
    printf(line_fmt, "-s, --steps <steps>", "Number of simulation iterations");
    printf(line_fmt, "-r, --random <seed>", "RandomManager seed");
    printf(line_fmt, "-q, --quiet", "Do not print progress information to console");
    printf(line_fmt, "-v, --verbose", "Print config, progress and timing (-t) information to console.");
    printf(line_fmt, "-t, --timing", "Output timing information to stdout");
    printf(line_fmt, "-u, --silence-unknown-args", "Silence warnings for unknown arguments passed after this flag.");
#ifdef FLAMEGPU_VISUALISATION
    printf(line_fmt, "-c, --console", "Console mode, disable the visualisation");
#endif
    printHelp_derived();
}

Simulation::Config &Simulation::SimulationConfig() {
    return config;
}
const Simulation::Config &Simulation::getSimulationConfig() const {
    return config;
}

void Simulation::reset() {
    loaded_input_file = "";
    reset(false);
}

unsigned int Simulation::get_instance_id() {
    static std::atomic<unsigned int> i = {0};;
    return 641 * (i++);
}

}  // namespace flamegpu
