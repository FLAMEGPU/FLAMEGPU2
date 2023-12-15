#include "flamegpu/runtime/environment/HostEnvironment.cuh"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <vector>

#include "flamegpu/io/StateWriter.h"
#include "flamegpu/io/StateWriterFactory.h"
#include "flamegpu/io/StateReader.h"
#include "flamegpu/io/StateReaderFactory.h"
#include "flamegpu/simulation/CUDASimulation.h"

namespace flamegpu {
HostEnvironment::HostEnvironment(CUDASimulation &_simulation, std::shared_ptr<detail::EnvironmentManager> env, std::shared_ptr<detail::CUDAMacroEnvironment> _macro_env,
    CUDADirectedGraphMap& _directed_graph_map, detail::CUDAScatter& _scatter, const unsigned int _streamID, const cudaStream_t _stream)
    : env_mgr(std::move(env))
    , macro_env(std::move(_macro_env))
    , directed_graph_map(_directed_graph_map)
    , instance_id(_simulation.getInstanceID())
    , simulation(_simulation)
    , scatter(_scatter)
    , streamID(_streamID)
    , stream(_stream) { }

void HostEnvironment::importMacroProperty(const std::string& property_name, const std::string& file_path) const {
    // Validate the property exists
    const auto &m_props = macro_env->getPropertiesMap();
    const auto &m_prop = m_props.find(property_name);
    if (m_prop == m_props.end()) {
        THROW exception::InvalidEnvProperty("The environment macro property '%s' was not found within the model description, in HostEnvironment::importMacroProperty().", property_name.c_str());
    }
    const unsigned int m_prop_elements = std::accumulate(m_prop->second.elements.begin(), m_prop->second.elements.end(), 1, std::multiplies<unsigned int>());
    try {
        io::StateReader *read__ = io::StateReaderFactory::createReader(file_path);
        read__->parse(file_path, simulation.getModelDescription().shared_from_this(), Verbosity::Quiet);
        std::unordered_map<std::string, std::vector<char>> macro_init;
        read__->getMacroEnvironment(macro_init);
        // Validate the property exists within macro_init
        const auto &l_prop = macro_init.find(property_name);
        if (l_prop == macro_init.end()) {
            THROW exception::InvalidEnvProperty("The environment macro property '%s' was not found within the input file '%s'.", property_name.c_str(), file_path.c_str());
        }
        // Check the length validates
        if (l_prop->second.size() != m_prop_elements * m_prop->second.type_size) {
            THROW exception::InvalidInputFile("Length of input file '%s's environment macro property '%s'  does not match, (%u != %u), in HostEnvironment::importMacroProperty()",
                file_path.c_str(), property_name.c_str(), static_cast<unsigned int>(l_prop->second.size()), static_cast<unsigned int>(m_prop_elements * m_prop->second.type_size));
        }
        gpuErrchk(cudaMemcpyAsync(m_prop->second.d_ptr, l_prop->second.data(), l_prop->second.size(), cudaMemcpyHostToDevice, stream));
    } catch (const exception::UnsupportedFileType&) {
        const std::string extension = std::filesystem::path(file_path).extension().string();
        if (extension == ".bin") {
            // Additionally support raw binary dump
            // Read the file
            std::ifstream input(file_path, std::ios::binary);
            std::vector buffer(std::istreambuf_iterator<char>(input), {});
            // Check the length validates
            if (buffer.size() != m_prop_elements * m_prop->second.type_size) {
                THROW exception::InvalidInputFile("Length of binary input file '%s' does not match the environment macro property '%s', (%u != %u), in HostEnvironment::importMacroProperty()",
                    file_path.c_str(), property_name.c_str(), static_cast<unsigned int>(buffer.size()), static_cast<unsigned int>(m_prop_elements * m_prop->second.type_size));
            }
            // Update the property
            gpuErrchk(cudaMemcpyAsync(m_prop->second.d_ptr, buffer.data(), buffer.size(), cudaMemcpyHostToDevice, stream));
        } else {
            throw;
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    // If macro property exists in cache sync cache
    if (const auto cache = macro_env->getHostPropertyMetadata(property_name)) {
        cache->force_download();
    }
}
void HostEnvironment::exportMacroProperty(const std::string& property_name, const std::string& file_path, bool pretty_print) const {
    // If macro property exists in cache sync cache
    if (const auto cache = macro_env->getHostPropertyMetadata(property_name)) {
        cache->upload();
    }
    try {
        io::StateWriter* write__ = io::StateWriterFactory::createWriter(file_path);
        write__->beginWrite(file_path, pretty_print);
        write__->writeMacroEnvironment(macro_env, { property_name });
        write__->endWrite();
    }
    catch (const exception::UnsupportedFileType&) {
        const std::string extension = std::filesystem::path(file_path).extension().string();
        if (extension == ".bin") {
            // Additionally support raw binary dump
            // Validate the property exists
            const auto& m_props = macro_env->getPropertiesMap();
            const auto& m_prop = m_props.find(property_name);
            if (m_prop == m_props.end()) {
                THROW exception::InvalidEnvProperty("The environment macro property '%s' was not found within the model description, in HostEnvironment::exportMacroProperty().", property_name.c_str());
            }
            // Check the file doesn't already exist
            if (std::filesystem::exists(file_path)) {
                THROW exception::FileAlreadyExists("The binary output file '%s' already exists, in HostEnvironment::exportMacroProperty().", file_path.c_str());
            }
            // Copy the data to a temporary buffer on host
            const unsigned int m_prop_elements = std::accumulate(m_prop->second.elements.begin(), m_prop->second.elements.end(), 1, std::multiplies<unsigned int>());
            std::vector<char> buffer;
            buffer.resize(m_prop_elements * m_prop->second.type_size);
            gpuErrchk(cudaMemcpyAsync(buffer.data(), m_prop->second.d_ptr, m_prop_elements * m_prop->second.type_size, cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaStreamSynchronize(stream));
            // Output to file
            std::ofstream output(file_path, std::ios::binary);
            output.write(buffer.data(), buffer.size());
        } else {
            throw;
        }
    }
}

HostEnvironmentDirectedGraph HostEnvironment::getDirectedGraph(const std::string& name) const {
    const auto rt = directed_graph_map.find(name);
    if (rt != directed_graph_map.end())
        return HostEnvironmentDirectedGraph(rt->second, stream, scatter, streamID);
    THROW exception::InvalidGraphName("Directed Graph with name '%s' was not found, "
        "in HostEnvironment::getDirectedGraph()",
        name.c_str());
}

}  // namespace flamegpu
