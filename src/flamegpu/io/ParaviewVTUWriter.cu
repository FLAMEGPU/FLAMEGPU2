#include "flamegpu/io/ParaviewVTUWriter.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

#include "flamegpu/simulation/AgentVector.h"

namespace flamegpu {
namespace io {

void ParaviewVTUWriter::writeAgentState(const std::string& agent, const std::string& state, const std::shared_ptr<const AgentVector>& agents_map, const unsigned int step) {

    std::stringstream filename_ss;
    filename_ss << agent << "_" << state << "_" << std::setfill('0') << std::setw(5) << step << ".vtu";
    const std::filesystem::path filepath = std::filesystem::path(output_dir) / filename_ss.str();

    std::ofstream paraviewFile;
    paraviewFile.open(filepath.c_str(), std::ios::binary | std::ios::trunc);

    // Endianness (the order of bits within a byte) depends on the processor hardware
    // but it's probably LittleEndian, IBM processors are the only default BigEndian you're likely to come across
    static const uint16_t m_endianCheck(0x00ff);
    const bool is_big_endian(*((const uint8_t*)&m_endianCheck) == 0x0);
    // Header
    paraviewFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"" << (is_big_endian ? "BigEndian" : "LittleEndian") << "\"  header_type=\"UInt64\">\n";
    paraviewFile << " <UnstructuredGrid GhostLevel=\"0\">\n";
    paraviewFile << "  <Piece NumberOfPoints=\"" << agents_map->size() << "\" NumberOfCells=\"" << agents_map->size() << "\">\n";
    paraviewFile << "   <PointData>\n";
    size_t offset = 0;
    for (;;) {  // @todo pull name/type/components from config
        paraviewFile << "    <DataArray type=\"Float64\" Name=\"v\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += static_cast<size_t>(agents_map->size() * 3) * sizeof(double) + sizeof(size_t);
    }
    paraviewFile << "   </PointData>\n";
    paraviewFile << "   <CellData>\n";
    paraviewFile << "   </CellData>\n";
    paraviewFile << "   <Points>\n";
    // Points (this is the location of the agent) 
    paraviewFile << "    <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << offset << "\" />\n";
    offset += static_cast<size_t>(agents_map->size() * 3) * sizeof(float) + sizeof(size_t);
    paraviewFile << "   </Points>\n";
    paraviewFile << "   <Cells>\n";
    // Not sure of the importance of these values
    paraviewFile << "    <DataArray type=\"UInt32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << offset << "\" />\n";
    offset += static_cast<size_t>(agents_map->size()) * sizeof(unsigned int) + sizeof(size_t);
    paraviewFile << "    <DataArray type=\"UInt32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << offset << "\" />\n";
    offset += static_cast<size_t>(agents_map->size()) * sizeof(unsigned int) + sizeof(size_t);
    paraviewFile << "    <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << offset << "\" />\n";
    offset += static_cast<size_t>(agents_map->size()) * sizeof(unsigned char) + sizeof(size_t);
    paraviewFile << "   </Cells>\n";
    paraviewFile << "  </Piece>\n";
    paraviewFile << " </UnstructuredGrid>\n";
    // Data
    paraviewFile << " <AppendedData encoding=\"raw\">\n  _";
    /**
     * Based on the sparse documentation at https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html
     * and alot of testing.
     * Inside <AppendedData> the binary dump must be preceded by an underscore (_)
     * Each DataArray's binary dump must be preceded by it's length.
     * The length should be exported as the integer type specified as header_type in the opening <VTKFile> tag
     * The offset specified in the above <DataArray> tag refers to the offset from the start of the whole binary dump to the start of the length
     */
    for (;;) {  // @todo pull name/type/components from config, probably just write raw buffer
        offset = static_cast<size_t>(agents_map->size() * 3) * sizeof(double);
        paraviewFile.write(reinterpret_cast<const char*>(&offset), sizeof(size_t));
        for (unsigned int i = 0; i < lb.activeNodes.size(); ++i) {
            v_buffer[i] = lb.activeNodes[i]->u * lb.unit.Speed;
        }
        paraviewFile.write(t_buffer, offset);
    }
    // Points::Points

    // Cells::connectivity
    offset = static_cast<size_t>(agents_map->size()) * sizeof(unsigned int);
    paraviewFile.write(reinterpret_cast<const char*>(&offset), sizeof(size_t));
    for (unsigned int i = 0; i < agents_map->size(); ++i) {
        ui_buffer[i] = i;
    }
    paraviewFile.write(t_buffer, offset);
    // Cells::offsets
    offset = static_cast<size_t>(agents_map->size()) * sizeof(unsigned int);
    paraviewFile.write(reinterpret_cast<const char*>(&offset), sizeof(size_t));
    for (unsigned int i = 0; i < agents_map->size(); ++i) {
        ui_buffer[i] = i + 1;
    }
    paraviewFile.write(t_buffer, offset);
    // Cells::types
    offset = static_cast<size_t>(agents_map->size()) * sizeof(unsigned char);
    paraviewFile.write(reinterpret_cast<const char*>(&offset), sizeof(size_t));
    std::fill(uc_buffer, uc_buffer + agents_map->size(), static_cast<unsigned char>(1));
    paraviewFile.write(t_buffer, offset);
    // Footer
    paraviewFile << "</AppendedData>";
    paraviewFile << "</VTKFile>\n";

    paraviewFile.close();
}

}  // namespace io
}  // namespace flamegpu
