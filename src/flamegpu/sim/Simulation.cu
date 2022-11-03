#include "flamegpu/sim/Simulation.h"

#include <algorithm>
#include <atomic>

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


namespace flamegpu {

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
    NVTX_RANGE("Simulation::initialise");
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
                pops.emplace(util::StringPair{ agent.first, state }, std::make_shared<AgentVector>(*agent.second->description));
            }
        }

        env_init.clear();
        const auto env_desc = model->environment->getPropertiesMap();  // For some reason this method returns a copy, not a reference
        io::StateReader *read__ = io::StateReaderFactory::createReader(model->name, env_desc, env_init, pops, config.input_file.c_str(), this);
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
    // If verbose, output the flamegpu version.
    if (config.verbosity == VERBOSE) {
        fprintf(stdout, "FLAME GPU %s\n", flamegpu::VERSION_FULL);
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
    // Build population vector
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> pops;
    for (auto &agent : model->agents) {
        for (auto &state : agent.second->states) {
            auto a = std::make_shared<AgentVector>(*agent.second->description);
            getPopulationData(*a, state);
            pops.emplace(util::StringPair{agent.first, state}, a);
        }
    }

    io::StateWriter *write__ = io::StateWriterFactory::createWriter(model->name, getEnvironment(), pops, getStepCounter(), path, this);
    write__->writeStates(prettyPrint);
}
void Simulation::exportLog(const std::string &path, bool steps, bool exit, bool stepTime, bool exitTime, bool prettyPrint) {
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
                        pops.emplace(util::StringPair{ agent.first, state }, std::make_shared<AgentVector>(*agent.second->description));
                    }
                }
                env_init.clear();
                const auto env_desc = model->environment->getPropertiesMap();  // For some reason this method returns a copy, not a reference
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
            config.verbosity = VERBOSE;
            continue;
        }
        // -q/--quiet, Verbose level quiet FLAME GPU output.
        if (arg.compare("--quiet") == 0 || arg.compare("-q") == 0) {
            config.verbosity = QUIET;
            continue;
        }
        // -t/--timing, Output timing information to stdout
        if (arg.compare("--timing") == 0 || arg.compare("-t") == 0) {
            config.timing = true;
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
#ifdef VISUALISATION
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
        fprintf(stderr, "Unexpected argument: %s\n", arg.c_str());
        printHelp(argv[0]);
        return false;
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
    printf(line_fmt, "-v, --verbose", "Verbose FLAME GPU output");
    printf(line_fmt, "-t, --timing", "Output timing information to stdout");
#ifdef VISUALISATION
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
