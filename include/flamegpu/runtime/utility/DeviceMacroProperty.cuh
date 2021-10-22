#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEMACROPROPERTY_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEMACROPROPERTY_CUH_

#include <cstdint>
#include <limits>
#include <algorithm>

namespace flamegpu {

/**
 * This template class is used for conveniently converting a multi-dimensional pointer to an array
 * Theoretically the compiler should be able to optimise away most of it at compile time
 */
template<typename T, unsigned int I = 1, unsigned int J = 1, unsigned int K = 1, unsigned int W = 1>
class ReadOnlyDeviceMacroProperty {
 protected:
    T* ptr;
#if !defined(SEATBELTS) || SEATBELTS
    /**
     * This flag is used by seatbelts to check for a read/atomic-write conflict
     * Reading sets 1<<0
     * Writing sets 1<<1
     */
    unsigned int* read_write_flag;
    /**
     * Utility function for setting/checking read flag
     */
    __device__ void setCheckReadFlag() const;
    /**
     * Utility function for setting/checking write flag
     */
    __device__ void setCheckWriteFlag() const;
#endif

 public:
#if !defined(SEATBELTS) || SEATBELTS
     /**
      * Constructor
      * @param _ptr Pointer to buffer
      * @param _rwf Pointer to read_write_flag
      */
     __device__ explicit ReadOnlyDeviceMacroProperty(T* _ptr, unsigned int* _rwf);
#else
     /**
      * Constructor
      * @param _ptr Pointer to buffer
      */
     __device__ explicit ReadOnlyDeviceMacroProperty(T* _ptr);
#endif
     /**
      * Access the next dimension of the array
      * @throws exception::DeviceError If i >= I.
      * @throws exception::DeviceError If template arguments I, J, K , W are all 1. Which denotes the macro property has no dimensions remaining to be indexed.
      */
     __device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, J, K, W, 1> operator[](unsigned int i) const;
     /**
      * Read-only access to the current element
      * @throws exception::DeviceError If template arguments I, J, K , W are not all 1. Which denotes the macro property has dimensions remaining to be indexed.
      */
     __device__ __forceinline__ operator T() const;
};
template<typename T, unsigned int I = 1, unsigned int J = 1, unsigned int K = 1, unsigned int W = 1>
class DeviceMacroProperty : public ReadOnlyDeviceMacroProperty<T, I, J, K, W> {
 public:
#if !defined(SEATBELTS) || SEATBELTS
    /**
     * Constructor
     * @param _ptr Pointer to buffer
     * @param _rwf Pointer to read_write_flag
     */
    __device__ explicit DeviceMacroProperty(T* _ptr, unsigned int *_rwf);
#else
    /**
     * Constructor
     * @param _ptr Pointer to buffer
     */
     __device__ explicit DeviceMacroProperty(T* _ptr);
#endif
    /**
     * Access the next dimension of the array
     * @throws exception::DeviceError If i >= I. 
     * @throws exception::DeviceError If template arguments I, J, K , W are all 1. Which denotes the macro property has no dimensions remaining to be indexed. 
     */
    __device__ __forceinline__ DeviceMacroProperty<T, J, K, W, 1> operator[](unsigned int i) const;
    /**
     * atomic add
     * @param val The 2nd operand
     * @return a reference to this
     * Note, taking value of the returned object will fail, due to the risk of atomic conflicts
     * @note Only suitable where T is type int32_t, uint32_t, uint64_t, float, double
     */
    __device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W>& operator +=(const T& val);
    /**
     * atomic subtraction
     * @param val The 2nd operand
     * @return a reference to this
     * Note, taking value of the returned object will fail, due to the risk of atomic conflicts
     * @note Only suitable where T is type int32_t or uint32_t
     */
    __device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W>& operator -=(const T& val);
    /**
     * atomic add
     * @param val The 2nd operand
     * @return (this + val)
     * @note Only suitable where T is type int32_t, uint32_t, uint64_t, float, double
     */
    __device__ __forceinline__ T operator+(const T& val) const;
    /**
     * atomic subtraction
     * @param val The 2nd operand
     * @return (this - val)
     * @note Only suitable where T is type int32_t or uint32_t
     */
    __device__ __forceinline__ T operator-(const T& val) const;
    /**
     * atomic pre-increment
     * @return the value after the increment operation is performed
     * @note A reference is not returned, as this makes it easy to fall into the trap (race condition) of reading it's value.
     * @note Only suitable where T is type uint32_t
     */
    __device__ __forceinline__ T operator++();
    /**
     * atomic pre-decrement
     * @return the value after the decrement operation is performed
     * @note A reference is not returned, as this makes it easy to fall into the trap (race condition) of reading it's value.
     * @note Only suitable where T is type uint32_t
     */
    __device__ __forceinline__ T operator--();
    /**
     * atomic post-increment
     * @return the value before increment operation is performed
     * @note Only suitable where T is type uint32_t
     */
    __device__ __forceinline__ T operator++(int);
    /**
     * atomic post-decrement
     * @return the value before increment operation is performed
     * @note Only suitable where T is type uint32_t
     */
    __device__ __forceinline__ T operator--(int);
    /**
     * atomic min
     * @return min(this, val)
     * @note Only suitable where T is type int32_t, uint32_t, uint64_t
     */
    __device__ __forceinline__ T min(T val);
    /**
     * atomic max
     * @return max(this, val)
     * @note Only suitable where T is type int32_t, uint32_t, uint64_t
     */
    __device__ __forceinline__ T max(T val);
    /**
     * atomic compare and swap
     * Computes (old == compare ? val : old)
     * @return old
     * @note Only suitable where T is type int32_t, uint32_t, uint64_t, uint16_t
     */
    __device__ __forceinline__ T CAS(T compare, T val);
    /**
     * atomic exchange
     * Returns the current value stored in the element, and replaces it with val
     * @return the value before the exchange
     * @note Only suitable for 32/64 bit types
     */
    __device__ __forceinline__ T exchange(T val);
};

#if !defined(SEATBELTS) || SEATBELTS
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, I, J, K, W>::ReadOnlyDeviceMacroProperty(T* _ptr, unsigned int* _rwf)
    : ptr(_ptr)
    , read_write_flag(_rwf)
{ }
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W>::DeviceMacroProperty(T* _ptr, unsigned int* _rwf)
    : ReadOnlyDeviceMacroProperty<T, I, J, K, W>(_ptr, _rwf)
{ }
#ifdef __CUDACC__
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ void ReadOnlyDeviceMacroProperty<T, I, J, K, W>::setCheckReadFlag() const {
    const unsigned int old = atomicOr(read_write_flag, 1u << 0);
    if (old & 1u << 1) {
        DTHROW("DeviceMacroProperty read and atomic write operations cannot be mixed in the same layer, as this may cause race conditions.\n");
        return;
    }
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ void ReadOnlyDeviceMacroProperty<T, I, J, K, W>::setCheckWriteFlag() const {
    const unsigned int old = atomicOr(read_write_flag, 1u << 1);
    if (old & 1u << 0) {
        DTHROW("DeviceMacroProperty read and atomic write operations cannot be mixed in the same layer as this may cause race conditions.\n");
        return;
    }
}
#endif
#else
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, I, J, K, W>::ReadOnlyDeviceMacroProperty(T* _ptr)
    :ptr(_ptr)
{ }
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ DeviceMacroProperty<T, I , J, K, W>::DeviceMacroProperty(T* _ptr)
    : ReadOnlyDeviceMacroProperty<T, I, J, K, W>(_ptr)
{ }
#endif
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, J, K, W, 1> ReadOnlyDeviceMacroProperty<T, I, J, K, W>::operator[](unsigned int i) const {
#if !defined(SEATBELTS) || SEATBELTS
    if (I == 1 && J == 1 && K == 1 && W == 1) {
        DTHROW("Indexing error, property has less dimensions.\n");
        return ReadOnlyDeviceMacroProperty<T, J, K, W, 1>(nullptr, nullptr);
    } else if (i >= I) {
        DTHROW("Indexing error, out of bounds %u >= %u.\n", i, I);
        return ReadOnlyDeviceMacroProperty<T, J, K, W, 1>(nullptr, nullptr);
    } else if (this->ptr == nullptr) {
        return ReadOnlyDeviceMacroProperty<T, J, K, W, 1>(nullptr, nullptr);
    }
#endif
    // (i * J * K * W) + (j * K * W) + (k * W) + w
#if !defined(SEATBELTS) || SEATBELTS
    return ReadOnlyDeviceMacroProperty<T, J, K, W, 1>(this->ptr + (i * J * K * W), this->read_write_flag);
#else
    return DeviceMacroProperty<T, J, K, W, 1>(this->ptr + (i * J * K * W));
#endif
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ DeviceMacroProperty<T, J, K, W, 1> DeviceMacroProperty<T, I, J, K, W>::operator[](unsigned int i) const {
#if !defined(SEATBELTS) || SEATBELTS
    if (I == 1 && J == 1 && K == 1 && W == 1) {
        DTHROW("Indexing error, property has less dimensions.\n");
        return DeviceMacroProperty<T, J, K, W, 1>(nullptr, nullptr);
    } else if (i >= I) {
        DTHROW("Indexing error, out of bounds %u >= %u.\n", i, I);
        return DeviceMacroProperty<T, J, K, W, 1>(nullptr, nullptr);
    } else if (this->ptr == nullptr) {
        return DeviceMacroProperty<T, J, K, W, 1>(nullptr, nullptr);
    }
#endif
    // (i * J * K * W) + (j * K * W) + (k * W) + w
#if !defined(SEATBELTS) || SEATBELTS
    return DeviceMacroProperty<T, J, K, W, 1>(this->ptr + (i * J * K * W), this->read_write_flag);
#else
    return DeviceMacroProperty<T, J, K, W, 1>(this->ptr + (i * J * K * W));
#endif
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, I, J, K, W>::operator T() const {
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckReadFlag();
#endif
    return *this->ptr;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W>& DeviceMacroProperty<T, I, J, K, W>::operator+=(const T& val) {
    static_assert(std::is_same<T, int32_t>::value ||
        std::is_same<T, uint32_t>::value ||
        std::is_same<T, uint64_t>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value, "atomic add only supports the types int32_t/uint32_t/uint64_t/float/double.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return *this;
    } else if (this->ptr == nullptr) {
        return *this;
    }
    this->setCheckWriteFlag();
#endif
    atomicAdd(this->ptr, val);
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W>& DeviceMacroProperty<T, I, J, K, W>::operator-=(const T& val) {
    static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value, "atomic subtract only supports the types int32_t/uint32_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return *this;
    } else if (this->ptr == nullptr) {
        return *this;
    }
    this->setCheckWriteFlag();
#endif
    atomicSub(this->ptr, val);
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::operator+(const T& val) const {
    static_assert(std::is_same<T, int32_t>::value ||
        std::is_same<T, uint32_t>::value ||
        std::is_same<T, uint64_t>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value, "atomic add only supports the types int32_t/uint32_t/uint64_t/float/double.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    return atomicAdd(this->ptr, val) + val;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::operator-(const T& val) const {
    static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value, "atomic subtract only supports the types int32_t/uint32_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    return atomicSub(this->ptr, val) - val;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::operator++() {
    static_assert(std::is_same<T, uint32_t>::value, "atomic increment only supports the type uint32_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return *this;
    } else if (this->ptr == nullptr) {
        return *this;
    }
    this->setCheckWriteFlag();
#endif
    const T old = atomicInc(this->ptr, std::numeric_limits<T>::max());
    return ((old >= std::numeric_limits<T>::max()) ? 0 : (old + 1));
}

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::operator--() {
    static_assert(std::is_same<T, uint32_t>::value, "atomic decrement only supports the type uint32_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return *this;
    } else if (this->ptr == nullptr) {
        return *this;
    }
    this->setCheckWriteFlag();
#endif
    const T old = atomicDec(this->ptr, std::numeric_limits<T>::max());
    return  (((old == 0) || (old > std::numeric_limits<T>::max())) ? std::numeric_limits<T>::max() : (old - 1));
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::operator++(int) {
    static_assert(std::is_same<T, uint32_t>::value, "atomic increment only supports the type uint32_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    return atomicInc(this->ptr, std::numeric_limits<T>::max());
}

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::operator--(int) {
    static_assert(std::is_same<T, uint32_t>::value, "atomic decrement only supports the type uint32_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    return atomicDec(this->ptr, std::numeric_limits<T>::max());
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::min(T val) {
    static_assert(std::is_same<T, int32_t>::value ||
        std::is_same<T, uint32_t>::value ||
        std::is_same<T, uint64_t>::value, "atomic min only supports the types int32_t/uint32_t/uint64_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    return std::min(atomicMin(this->ptr, val), val);
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::max(T val) {
    static_assert(std::is_same<T, int32_t>::value ||
        std::is_same<T, uint32_t>::value ||
        std::is_same<T, uint64_t>::value, "atomic max only supports the types int32_t/uint32_t/uint64_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    return std::max(atomicMax(this->ptr, val), val);
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::CAS(T compare, T val) {
    static_assert(std::is_same<T, int32_t>::value ||
        std::is_same<T, uint32_t>::value ||
        std::is_same<T, uint64_t>::value ||
        std::is_same<T, uint16_t>::value, "atomic compare and swap only supports the types int32_t/uint32_t/uint64_t/uint16_t.");
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    return atomicCAS(this->ptr, compare, val);
}

// GCC doesn't like seeing atomicExch with host compiler
#ifdef __CUDACC__
#pragma diag_suppress = initialization_not_reachable
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
__device__ __forceinline__ T DeviceMacroProperty<T, I, J, K, W>::exchange(T val) {
    static_assert(std::is_same<T, int32_t>::value ||
        std::is_same<T, int64_t>::value ||
        std::is_same<T, uint32_t>::value ||
        std::is_same<T, uint64_t>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value, "atomic exchange only supports the types int32_t/int64_t/uint32_t/uint64_t/float/double.");
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long int), "uint64_t != unsigned long long int.");  // NOLINT(runtime/int)
#if !defined(SEATBELTS) || SEATBELTS
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        DTHROW("Indexing error, property has more dimensions.\n");
        return { };
    } else if (this->ptr == nullptr) {
        return { };
    }
    this->setCheckWriteFlag();
#endif
    if (sizeof(T) == sizeof(uint64_t)) {  // Convert all 64 bit types to unsigned long long int (can't build as uint64_t on gcc)
        const unsigned long long int rval = atomicExch(reinterpret_cast<unsigned long long int*>(this->ptr), *reinterpret_cast<unsigned long long int*>(&val));  // NOLINT(runtime/int)
        return *reinterpret_cast<const T*>(&rval);
    }
    // else 32-bit
    const uint32_t rval = atomicExch(reinterpret_cast<uint32_t*>(this->ptr), *reinterpret_cast<uint32_t*>(&val));
    return *reinterpret_cast<const T*>(&rval);
    // return atomicExch(this->ptr, val);
}
#pragma diag_default = initialization_not_reachable
#endif

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEMACROPROPERTY_CUH_
