#include "flamegpu/sim/Simulation.h"

#include <algorithm>
#include <atomic>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/io/xmlWriter.h"
#include "flamegpu/io/factory.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/model/AgentDescription.h"  // Only forward declared by AgentPopulation.h
#include "flamegpu/util/nvtx.h"
#include "flamegpu/util/filesystem.h"


Simulation::Simulation(const std::shared_ptr<const ModelData> &_model)
    : model(_model->clone())
    , submodel(nullptr)
    , mastermodel(nullptr)
    , instance_id(get_instance_id())
    , maxLayerWidth((*model).getMaxLayerWidth()) { }

Simulation::Simulation(const std::shared_ptr<SubModelData> &submodel_desc, CUDASimulation *master_model)
    : model(submodel_desc->submodel)
    , submodel(submodel_desc)
    , mastermodel(master_model)
    , instance_id(get_instance_id())
    , maxLayerWidth(submodel_desc->submodel->getMaxLayerWidth()) { }

void Simulation::initialise(int argc, const char** argv) {
    NVTX_RANGE("Simulation::initialise");
    config = Config();  // Reset to defaults
    resetDerivedConfig();
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
        std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
        for (auto &agent : model->agents) {
            auto a = std::make_shared<AgentPopulation>(*agent.second->description);
            pops.emplace(agent.first, a);
        }

        env_init.clear();
        const auto env_desc = model->environment->getPropertiesMap();  // For some reason this method returns a copy, not a reference
        StateReader *read__ = ReaderFactory::createReader(model->name, env_desc, env_init, pops, config.input_file.c_str(), this);
        if (read__) {
            read__->parse();
            for (auto &agent : pops) {
                setPopulationData(*agent.second);
            }
        }
        // Reset input file (we don't support input file recursion)
        config.input_file = current_input_file;
        // Set flag so we don't reload this in future
        loaded_input_file = current_input_file;
    }
    // Create directory for log files
    if (!config.step_log_file.empty()) {
        path t_path = config.step_log_file;
        try {
            t_path = t_path.parent_path();
            if (!t_path.empty()) {
                util::filesystem::recursive_create_dir(t_path);
            }
        } catch(std::exception &e) {
            THROW InvalidArgument("Failed to init step log file directory '%s': %s\n", t_path.c_str(), e.what());
        }
    }
    if (!config.exit_log_file.empty()) {
        path t_path = config.exit_log_file;
        try {
            t_path = t_path.parent_path();
            if (!t_path.empty()) {
                util::filesystem::recursive_create_dir(t_path);
            }
        } catch(std::exception &e) {
            THROW InvalidArgument("Failed to init exit log file directory: '%s': %s\n", t_path.c_str(), e.what());
        }
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
    std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    for (auto &agent : model->agents) {
        auto a = std::make_shared<AgentPopulation>(*agent.second->description);
        getPopulationData(*a);
        pops.emplace(agent.first, a);
    }

    StateWriter *write__ = WriterFactory::createWriter(model->name, getInstanceID(), pops, getStepCounter(), path, this);
    write__->writeStates(prettyPrint);
}
void Simulation::exportLog(const std::string &path, bool steps, bool exit, bool prettyPrint) {
    // Create the correct type of logger
    auto logger = WriterFactory::createLogger(path, prettyPrint, config.truncate_log_files);
    // Perform logging
    logger->log(getRunLog(), true, steps, exit);
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
                std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
                for (auto &agent : model->agents) {
                    auto a = std::make_shared<AgentPopulation>(*agent.second->description);
                    pops.emplace(agent.first, a);
                }
                env_init.clear();
                const auto env_desc = model->environment->getPropertiesMap();  // For some reason this method returns a copy, not a reference
                StateReader *read__ = ReaderFactory::createReader(model->name, env_desc, env_init, pops, config.input_file.c_str(), this);
                if (read__) {
                    read__->parse();
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
            config.random_seed = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -v/--verbose, Verbose FLAME GPU output.
        if (arg.compare("--verbose") == 0 || arg.compare("-v") == 0) {
            config.verbose = true;
            continue;
        }
        // -t/--timing, Output timing information to stdout
        if (arg.compare("--timing") == 0 || arg.compare("-t") == 0) {
            config.timing = true;
            continue;
        }
        // -os/--out_step <file.xml/file.json>, Step log file path
        if (arg.compare("--out_step") == 0 || arg.compare("-os") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            config.step_log_file = argv[++i];
            continue;
        }
        // -oe/--out_exit <file.xml/file.json>, Exit log file path
        if (arg.compare("--out_exit") == 0 || arg.compare("-oe") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            config.exit_log_file = argv[++i];
            continue;
        }
        // -ol/--out_log <file.xml/file.json>, Common log file path
        if (arg.compare("--out_log") == 0 || arg.compare("-ol") == 0) {
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
    printf("Usage: %s [-s steps] [-d device_id] [-r random_seed]\n", executable);
    printf("Optional Arguments:\n");
    const char *line_fmt = "%-18s %s\n";
    printf(line_fmt, "-i, --in <file.xml/file.json>", "Initial state file (XML or JSON)");
    printf(line_fmt, "-os, --out_step <file.xml/file.json>", "Step log file (XML or JSON)");
    printf(line_fmt, "-oe, --out_exit <file.xml/file.json>", "Exit log file (XML or JSON)");
    printf(line_fmt, "-ol, --out_log <file.xml/file.json>", "Common log file (XML or JSON)");
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
