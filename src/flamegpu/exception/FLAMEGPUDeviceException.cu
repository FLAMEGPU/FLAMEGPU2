#include "flamegpu/exception/FLAMEGPUDeviceException.cuh"

#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#if !defined(SEATBELTS) || SEATBELTS

namespace flamegpu {
namespace exception {

DeviceExceptionManager::DeviceExceptionManager()
    : d_buffer()
    , hd_buffer() {
    memset (&d_buffer, 0, sizeof(d_buffer));
    memset (&hd_buffer, 0, sizeof(hd_buffer));
}
DeviceExceptionManager::~DeviceExceptionManager() {
    for (auto &i : d_buffer) {
        gpuErrchk(cudaFree(i));
    }
}
DeviceExceptionBuffer *DeviceExceptionManager::getDevicePtr(const unsigned int streamId, const cudaStream_t &stream) {
    if (streamId >= CUDAScanCompaction::MAX_STREAMS) {
        THROW exception::OutOfBoundsException("Stream id %u is out of bounds, %u >= %u, "
        "in FLAMEGPUDeviceException::getDevicePtr()\n", streamId, streamId, CUDAScanCompaction::MAX_STREAMS);
    }
    // It may be better to move this (and the memsets) out to a separate up-front reset call in the future.
    if (!d_buffer[streamId]) {
        gpuErrchk(cudaMalloc(&d_buffer[streamId], sizeof(DeviceExceptionBuffer)));
    }
    // @todo - We might need a sync here in some cases? Tests all pass without it.
    // gpuErrchk(cudaDeviceSynchronize());

    // Memset and return buffer
    gpuErrchk(cudaMemsetAsync(d_buffer[streamId], 0, sizeof(DeviceExceptionBuffer), stream));
    memset(&hd_buffer[streamId], 0, sizeof(DeviceExceptionBuffer));
    return d_buffer[streamId];
}
void DeviceExceptionManager::checkError(const std::string &function, const unsigned int streamId, const cudaStream_t &stream) {
    if (streamId >= CUDAScanCompaction::MAX_STREAMS) {
        THROW exception::OutOfBoundsException("Stream id %u is out of bounds, %u >= %u, "
        "in FLAMEGPUDeviceException::checkError()\n", streamId, streamId, CUDAScanCompaction::MAX_STREAMS);
    }
    if (d_buffer[streamId]) {
        // Grab buffer from device
        gpuErrchk(cudaMemcpyAsync(&hd_buffer[streamId], d_buffer[streamId], sizeof(DeviceExceptionBuffer), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        // If there is a reported error count
        if (hd_buffer[streamId].error_count) {
            std::string location_string = getLocationString(hd_buffer[streamId]);
            std::string error_string = getErrorString(hd_buffer[streamId]);
            throw exception::DeviceError(
            "Device function '%s' reported %u errors.\nFirst error:\n%s:\n%s",
            function.c_str(), hd_buffer[streamId].error_count, location_string.c_str(), error_string.c_str());
        }
    } else {
        THROW exception::OutOfBoundsException("FLAMEGPUDeviceExceptionBuffer for stream %u has not been allocated, "
        "in FLAMEGPUDeviceException::checkError()\n", streamId, streamId, CUDAScanCompaction::MAX_STREAMS);
    }
}
std::string DeviceExceptionManager::getLocationString(const DeviceExceptionBuffer &b) {
    char buff[DeviceExceptionBuffer::OUT_STRING_LEN];
    snprintf(buff, DeviceExceptionBuffer::OUT_STRING_LEN, "%s(%u)[%u,%u,%u][%u,%u,%u]",
        b.file_path, b.line_no,
        b.block_id[0], b.block_id[1], b.block_id[2],
        b.thread_id[0], b.thread_id[1], b.thread_id[2]);
    return buff;
}
std::string DeviceExceptionManager::getErrorString(const DeviceExceptionBuffer &b) {
    /**
     * This buffer is used to copy sub-format strings into before we send them to snprintf
     * This saves us needing to set the final+1 char to '\0'
     */
    char temp_buffer[DeviceExceptionBuffer::FORMAT_BUFF_LEN];
    /**
     * This is the buffer into which we construct the string to be returned
     */
    char out_buffer[DeviceExceptionBuffer::OUT_STRING_LEN];
    memset(out_buffer, 0, DeviceExceptionBuffer::FORMAT_BUFF_LEN);
    // Progress through b.format_string
    unsigned int format_buffer_index = 0;
    // Progress through out_buffer
    unsigned int out_index = 0;
    // Progress through b.format_args_sizes
    unsigned int arg_no = 0;
    // Progress through b.format_args
    unsigned int arg_offset = 0;
    // Whilst there is still work to be done, we are still in range of format string and all other structures used
    while (b.format_string[format_buffer_index] != '\0' &&
           format_buffer_index < DeviceExceptionBuffer::FORMAT_BUFF_LEN &&
           out_index < DeviceExceptionBuffer::FORMAT_BUFF_LEN &&
           arg_no < DeviceExceptionBuffer::MAX_ARGS) {
        // If we find the start of a sub format string
        if (b.format_string[format_buffer_index] == '%') {
            // Find the next sub format start, or end of entire format string
            unsigned int format_end = format_buffer_index + 1;
            char format_type = '\0';
            while (b.format_string[format_end] != '%' &&
                  b.format_string[format_end] != '\0' &&
                  format_end < DeviceExceptionBuffer::FORMAT_BUFF_LEN) {
                // Detect the format type, we will use this later
                if (format_type == '\0') {
                    switch (b.format_string[format_end]) {
                        // This is every format specifier supported by the printf family of functions
                        case 'd':
                        case 'i':
                        case 'u':
                        case 'o':
                        case 'x':
                        case 'X':
                        case 'f':
                        case 'e':
                        case 'g':
                        case 'G':
                        case 'a':
                        case 'A':
                        case 'c':
                        case 's':
                        case 'p':
                        case 'n':
                            format_type = b.format_string[format_end];
                            break;
                    }
                }
                ++format_end;
            }
            // Sub format string bounds have been found
            // Copy the sub format string into a temporary buffer
            memset(temp_buffer, 0, DeviceExceptionBuffer::FORMAT_BUFF_LEN);
            memcpy(temp_buffer, b.format_string + format_buffer_index, format_end - format_buffer_index);
            // Now send this substring to the formatter to process
            // Cast it to the correct type first
            // (This assumes snprintf never returns negative)
            switch (format_type) {
                case 'd':
                case 'i': {
                    // Signed integer
                    if (b.format_args_sizes[arg_no] == 4) {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, *reinterpret_cast<const int32_t*>(b.format_args+arg_offset));
                    } else {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, *reinterpret_cast<const int64_t*>(b.format_args+arg_offset));
                    }
                    break;
                }
                case 'u':
                case 'o':
                case 'x':
                case 'X': {
                    // Unsigned integer
                    if (b.format_args_sizes[arg_no] == 4) {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, *reinterpret_cast<const uint32_t*>(b.format_args+arg_offset));
                    } else {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, *reinterpret_cast<const uint64_t*>(b.format_args+arg_offset));
                    }
                    break;
                }
                case 'f':
                case 'e':
                case 'g':
                case 'G':
                case 'a':
                case 'A': {
                    // Floating point
                    if (b.format_args_sizes[arg_no] == 4) {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, *reinterpret_cast<const float*>(b.format_args+arg_offset));
                    } else {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, *reinterpret_cast<const double*>(b.format_args+arg_offset));
                    }
                    break;
                }
                case 'c': {
                    // Char
                    out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, *reinterpret_cast<const char*>(b.format_args+arg_offset));
                    break;
                }
                case 's': {
                    // Char string
                    out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, reinterpret_cast<const char*>(b.format_args+arg_offset));
                    break;
                }
                case 'p': {
                    // Pointer
                    out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, reinterpret_cast<const void*>(b.format_args+arg_offset));
                    break;
                }
                case 'n': {
                    // No of chars written (signed pointer to have value written back to)
                    if (b.format_args_sizes[arg_no] == 4) {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, reinterpret_cast<const int32_t*>(b.format_args+arg_offset));
                    } else {
                        out_index += snprintf(out_buffer + out_index, DeviceExceptionBuffer::OUT_STRING_LEN - out_index, temp_buffer, reinterpret_cast<const int64_t*>(b.format_args+arg_offset));
                    }
                    break;
                }
            }
            // Update arg counters
            arg_offset += b.format_args_sizes[arg_no];
            ++arg_no;
            // Update pointer into main format string and continue loop
            format_buffer_index = format_end;
        } else {
            // Copy the single char
            // This will only happen until we hit first sub format string
            out_buffer[out_index] = b.format_string[format_buffer_index];
            ++out_index;
            ++format_buffer_index;
        }
    }
    return out_buffer;
}

}  // namespace exception
}  // namespace flamegpu

#endif  // SEATBELTS are off
