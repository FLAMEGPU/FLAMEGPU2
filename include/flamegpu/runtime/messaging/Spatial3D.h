#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3D_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3D_H_

#include <algorithm>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/runtime/messaging.h"
#include "flamegpu/gpu/CUDAMessage.h"

#include "flamegpu/runtime/cuRVE/curve.h"

namespace {
struct GridPos3D {
    int x, y, z;
};
}  // namespace

class MsgSpatial3D {
    typedef unsigned int size_type;

 public:
    class Message;   // Forward declare inner classes
    class iterator;  // Forward declare inner classes
    struct MetaData {
        float min[3];
        float max[3];
        float radius;
        unsigned int *PBM;
        unsigned int gridDim[3];
        float environmentWidth[3];
    };
    /**
     * This class is returned to user by Device API
     * It gives access to message iterators
     */
    class In {
     public:
        class Filter {
            friend class MsgSpatial3D::Message;

         public:
            // Inner class representing an individual message
            class Message {
             private:
                const Filter &_parent;
                int relative_cell[2] = { -2, 1 };
                int cell_index_max = 0;
                int cell_index = 0;

             public:
                __device__ Message(const Filter &parent, const int &relative_cell_y, const int &relative_cell_z, const int &_cell_index_max, const int &_cell_index)
                    : _parent(parent)
                    , cell_index_max(_cell_index_max)
                    , cell_index(_cell_index) {
                    relative_cell[0] = relative_cell_y;
                    relative_cell[1] = relative_cell_z;
                }
                __device__ bool operator==(const Message& rhs) const {
                    return this->relative_cell[0] == rhs.relative_cell[0]
                        && this->relative_cell[1] == rhs.relative_cell[1]
                        && this->cell_index_max == rhs.cell_index_max
                        && this->cell_index == rhs.cell_index;
                }
                __device__ bool operator!=(const Message& rhs) const { return !(*this == rhs); }
                __device__ Message& operator++();
                __device__ void nextStrip() {
                    if (relative_cell[1] >= 1) {
                        relative_cell[1] = -1;
                        relative_cell[0]++;
                    } else {
                        relative_cell[1]++;
                    }
                }
                template<typename T, size_type N>
                __device__ T getVariable(const char(&variable_name)[N]) const;
            };
            // message list iterator inner class.
            class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
             private:
                Message _message;

             public:
                __device__ iterator(const Filter &parent, const int &relative_cell_y, const int &relative_cell_z, const int &_cell_index_max, const int &_cell_index)
                    : _message(parent, relative_cell_y, relative_cell_z, _cell_index_max, _cell_index) {
                    // Increment to find first message
                    ++_message;
                }
                __device__ iterator& operator++() { ++_message;  return *this; }
                __device__ iterator operator++(int) { iterator tmp(*this); operator++(); return tmp; }
                __device__ bool operator==(const iterator& rhs) { return  _message == rhs._message; }
                __device__ bool operator!=(const iterator& rhs) { return  _message != rhs._message; }
                __device__ Message& operator*() { return _message; }
            };
            __device__ Filter(const MetaData *_metadata, const Curve::NamespaceHash &combined_hash, const float &x, const float &y, const float &z);
            inline __device__ iterator begin(void) const {
                return iterator(*this, -2, 1, 1, 0);
            }
            inline __device__ iterator end(void) const {
                return iterator(*this, 1, 0, 1, 0);
            }

         private:
            float loc[3];
            GridPos3D cell;
            const MetaData *metadata;
            Curve::NamespaceHash combined_hash;
        };
        __device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *_metadata)
            : combined_hash(agentfn_hash + msg_hash)
            , metadata(reinterpret_cast<const MetaData*>(_metadata))
        { }
        /**
         * Returns a Filter object which provides access to message iterator
         */
        inline __device__ Filter operator() (const float &x, const float &y, const float &z) const {
            return Filter(metadata, combined_hash, x, y, z);
        }

     private:
        // agent_function + msg_hash
        Curve::NamespaceHash combined_hash;
        const MetaData *metadata;
    };

    class Out {
     public:
        __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, unsigned int _streamId)
            : combined_hash(agentfn_hash + msg_hash)
            , streamId(_streamId)
        { }
        template<typename T, unsigned int N>
        __device__ void setVariable(const char(&variable_name)[N], T value) const;
        __device__ void setLocation(const float &x, const float &y, const float &z) const;

     private:
        // agent_function + msg_hash
        Curve::NamespaceHash combined_hash;
        unsigned int streamId;
    };
    template<typename SimSpecialisationMsg>
    class CUDAModelHandler : public MsgSpecialisationHandler<SimSpecialisationMsg> {
     public:
        explicit CUDAModelHandler(CUDAMessage &a)
            : MsgSpecialisationHandler(a) {
            const Spatial3DMessageData &d = (const Spatial3DMessageData&)a.getMessageDescription();
            hd_data.radius = d.radius;
            hd_data.min[0] = d.minX; hd_data.min[1] = d.minY; hd_data.min[2] = d.minZ;
            hd_data.max[0] = d.maxX; hd_data.max[1] = d.maxY; hd_data.max[2] = d.maxZ;
            binCount = 1;
            for (unsigned int axis = 0; axis < 3; ++axis) {
                hd_data.environmentWidth[axis] = hd_data.max[axis] - hd_data.min[axis];
                hd_data.gridDim[axis] = static_cast<unsigned int>(ceil(hd_data.environmentWidth[axis] / static_cast<float>(hd_data.radius)));
                binCount *= hd_data.gridDim[axis];
            }
            gpuErrchk(cudaMalloc(&d_histogram, (binCount + 1) * sizeof(unsigned int)));
            gpuErrchk(cudaMalloc(&hd_data.PBM, (binCount + 1) * sizeof(unsigned int)));
            gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
            gpuErrchk(cudaMemcpy(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice));
            resizeCubTemp();
        }
        ~CUDAModelHandler() override {
            d_CUB_temp_storage_bytes = 0;
            gpuErrchk(cudaFree(d_CUB_temp_storage));
            gpuErrchk(cudaFree(d_histogram));
            gpuErrchk(cudaFree(hd_data.PBM));
            gpuErrchk(cudaFree(d_data));
            if (d_keys) {
                d_keys_vals_storage_bytes = 0;
                gpuErrchk(cudaFree(d_keys));
                gpuErrchk(cudaFree(d_vals));
            }
        }
        void buildIndex() override;
        const void *getMetaDataDevicePtr() const override { return d_data; }

     private:
        // Currently assumed that bounds of environment/rad never change
        void resizeCubTemp();

        void resizeKeysVals(const unsigned int &newSize);
        // Number of bins, arrays are +1 this length
        unsigned int binCount = 0;
        size_t d_CUB_temp_storage_bytes = 0;
        unsigned int *d_CUB_temp_storage = nullptr;
        unsigned int *d_histogram = nullptr;
        unsigned int *d_keys = nullptr, *d_vals = nullptr;
        size_t d_keys_vals_storage_bytes = 0;
        MetaData hd_data;
        MetaData *d_data = nullptr;
    };
};
#ifdef __CUDACC__
#include "flamegpu/gpu/CUDAScatter.h"
#ifdef _MSC_VER
#pragma warning(push, 3)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

template<typename T, unsigned int N>
__device__ T MsgSpatial3D::In::Filter::Message::getVariable(const char(&variable_name)[N]) const {
    //// Ensure that the message is within bounds.
    if (relative_cell[0] < 2) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, cell_index);
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}

/**
 * \brief adds a message
 * \param variable_name Name of message variable to set
 * \param value Value to set it to
 */
template<typename T, unsigned int N>
__device__ void MsgSpatial3D::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variable using curve
    Curve::setVariable<T>(variable_name, combined_hash, value, index);

    // Don't bother, handled by setLocation
    // Set scan flag incase the message is optional
    // flamegpu_internal::CUDAScanCompaction::ds_message_configs[streamId].scan_flag[index] = 1;
}

namespace Spatial3D {
    __device__ __forceinline__ GridPos3D getGridPosition(const MsgSpatial3D::MetaData *md, float x, float y, float z) {
        // Clamp each grid coord to 0<=x<dim
        int gridPos[3] = {
            static_cast<int>(floor((x / md->environmentWidth[0])*md->gridDim[0])),
            static_cast<int>(floor((y / md->environmentWidth[1])*md->gridDim[1])),
            static_cast<int>(floor((z / md->environmentWidth[2])*md->gridDim[2]))
        };
        GridPos3D rtn = {
            gridPos[0] < 0 ? 0 : (gridPos[0] > md->gridDim[0] - 1 ? static_cast<int>(md->gridDim[0] - 1) : gridPos[0]),
            gridPos[1] < 0 ? 0 : (gridPos[1] > md->gridDim[1] - 1 ? static_cast<int>(md->gridDim[1] - 1) : gridPos[1]),
            gridPos[2] < 0 ? 0 : (gridPos[2] > md->gridDim[2] - 1 ? static_cast<int>(md->gridDim[2] - 1) : gridPos[2])
        };
        return rtn;
    }
    __device__ __forceinline__ unsigned int getHash(const MsgSpatial3D::MetaData *md, const GridPos3D &xyz) {
        // Bound gridPos to gridDimensions
        unsigned int gridPos[3] = {
            xyz.x < 0 ? 0 : (xyz.x > md->gridDim[0] - 1 ? md->gridDim[0] - 1 : xyz.x),
            xyz.y < 0 ? 0 : (xyz.y > md->gridDim[1] - 1 ? md->gridDim[1] - 1 : xyz.y),
            xyz.z < 0 ? 0 : (xyz.z > md->gridDim[2] - 1 ? md->gridDim[2] - 1 : xyz.z)
        };
        // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
        return (unsigned int)(
            (gridPos[2] * md->gridDim[0] * md->gridDim[1]) +   // z
            (gridPos[1] * md->gridDim[0]) +                    // y
             gridPos[0]);                                      // x
    }
}  // namespace Spatial3D
namespace {
    __global__ void atomicHistogram3D(
        const MsgSpatial3D::MetaData *md,
        unsigned int* bin_index,
        unsigned int* bin_sub_index,
        unsigned int *pbm_counts,
        unsigned int message_count,
        const float * __restrict__ x,
        const float * __restrict__ y,
        const float * __restrict__ z) {
        unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        // Kill excess threads
        if (index >= message_count) return;

        GridPos3D gridPos = Spatial3D::getGridPosition(md, x[index], y[index], z[index]);
        unsigned int hash = Spatial3D::getHash(md, gridPos);
        bin_index[index] = hash;
        unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
        bin_sub_index[index] = bin_idx;
    }
}  // namespace

template <typename SimSpecialisationMsg>
void MsgSpatial3D::CUDAModelHandler<SimSpecialisationMsg>::buildIndex() {
    const unsigned int MESSAGE_COUNT = sim_message.getMessageCount();
    resizeKeysVals(sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    {  // Build atomic histogram
        gpuErrchk(cudaMemset(d_histogram, 0x00000000, (binCount + 1) * sizeof(unsigned int)));
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram3D, 32, 0));  // Randomly 32
        // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        atomicHistogram3D << <gridSize, blockSize >> >(d_data, d_keys, d_vals, d_histogram, MESSAGE_COUNT,
            reinterpret_cast<float*>(sim_message.getReadPtr("x")),
            reinterpret_cast<float*>(sim_message.getReadPtr("y")),
            reinterpret_cast<float*>(sim_message.getReadPtr("z"));
        gpuErrchk(cudaDeviceSynchronize());
    }
    {  // Scan (sum), to finalise PBM
        cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_histogram, hd_data.PBM, binCount + 1);
    }
    {  // Reorder messages
        // Copy messages from d_messages to d_messages_swap, in hash order
        auto &cs = CUDAScatter::getInstance(0);  // Choose proper stream_id in future!
        cs.pbm_reorder(sim_message.getMessageDescription().variables, sim_message.getReadList(), sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        sim_message.swap(false, 0);  // Stream id is unused here
    }
    {  // Fill PBM and Message Texture Buffers
        // gpuErrchk(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * MESSAGE_COUNT));
        // gpuErrchk(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (binCount + 1)));
    }
}
template <typename SimSpecialisationMsg>
void MsgSpatial3D::CUDAModelHandler<SimSpecialisationMsg>::resizeCubTemp() {
    size_t bytesCheck;
    cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, hd_data.PBM, d_histogram, binCount + 1);
    if (bytesCheck > d_CUB_temp_storage_bytes) {
        if (d_CUB_temp_storage) {
            gpuErrchk(cudaFree(d_CUB_temp_storage));
        }
        d_CUB_temp_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
    }
}
template <typename SimSpecialisationMsg>
void MsgSpatial3D::CUDAModelHandler<SimSpecialisationMsg>::resizeKeysVals(const unsigned int &newSize) {
    size_t bytesCheck = newSize * sizeof(unsigned int);
    if (bytesCheck > d_keys_vals_storage_bytes) {
        if (d_keys) {
            gpuErrchk(cudaFree(d_keys));
            gpuErrchk(cudaFree(d_vals));
        }
        d_keys_vals_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_keys, d_keys_vals_storage_bytes));
        gpuErrchk(cudaMalloc(&d_vals, d_keys_vals_storage_bytes));
    }
}
#endif  // __CUDACC__
#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3D_H_
