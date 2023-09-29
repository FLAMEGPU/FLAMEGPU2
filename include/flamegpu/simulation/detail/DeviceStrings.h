#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_DEVICESTRINGS_H_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_DEVICESTRINGS_H_

#include <string>
#include <map>
#include <sstream>

namespace flamegpu {
namespace detail {
/**
 * Utility for copying strings to device
 */
class DeviceStrings {
 public:
    /**
     * Deallocates held device pointers
     */
    ~DeviceStrings();
    /**
     * Register a device string
     */
    void registerDeviceString(const std::string &host_string);
    /**
     * Returns a device pointer to the provided string
     * @note If reallocation is required, earlier pointers may be invalidated
     */
    const char* getDeviceString(const std::string &host_string);

 private:
    std::stringstream host_stream;
    // Cache stream in a string to reduce stream->string repeat conversion during sim execution
    std::string host_buffer;
    // Hold the offset into buffer for all registered strings
    std::map<std::string, ptrdiff_t> offsets;
    char* device_buffer = nullptr;
    size_t device_buffer_occupied = 0;
    size_t device_buffer_len = 0;
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_DEVICESTRINGS_H_
