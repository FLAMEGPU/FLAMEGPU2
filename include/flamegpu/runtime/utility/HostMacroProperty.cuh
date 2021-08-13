#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTMACROPROPERTY_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTMACROPROPERTY_CUH_

#include <cstdint>
#include <algorithm>
#include <array>
#include <memory>
#include <cmath>
#include <string>

namespace flamegpu {

struct HostMacroProperty_MetaData {
    /**
     * Constructor
     */
    HostMacroProperty_MetaData(void* _d_base_ptr, const std::array<unsigned int, 4>& _dims, size_t _type_size,
        bool _device_read_flag, const std::string &name)
        : h_base_ptr(nullptr)
        , d_base_ptr(static_cast<char*>(_d_base_ptr))
        , dims(_dims)
        , elements(dims[0] * dims[1] * dims[2] * dims[3])
        , type_size(_type_size)
        , has_changed(false)
        , device_read_flag(_device_read_flag)
        , property_name(name)
    { }
    ~HostMacroProperty_MetaData() {
        upload();
        if (h_base_ptr)
            std::free(h_base_ptr);
    }
    /**
     * Download data
     */
    void download() {
        if (!h_base_ptr) {
            h_base_ptr = static_cast<char*>(malloc(elements * type_size));
            gpuErrchk(cudaMemcpy(h_base_ptr, d_base_ptr, elements * type_size, cudaMemcpyDeviceToHost));
            has_changed = false;
        }
    }
    /**
     * Upload data
     */
    void upload() {
        if (h_base_ptr && has_changed) {
#if !defined(SEATBELTS) || SEATBELTS
            if (device_read_flag) {
                THROW flamegpu::exception::InvalidEnvProperty("The environment macro property '%s' was not found, "
                    "in HostMacroProperty_MetaData::upload()\n",
                    property_name.c_str());
            }
#endif
            gpuErrchk(cudaMemcpy(d_base_ptr, h_base_ptr, elements * type_size, cudaMemcpyHostToDevice));
            has_changed = false;
        }
    }
    char* h_base_ptr;
    char* d_base_ptr;
    std::array<unsigned int, 4> dims;
    unsigned int elements;
    size_t type_size;
    bool has_changed;
    bool device_read_flag;
    std::string property_name;
};

/**
 * This template class is used for conveniently converting a multi-dimensional pointer to an array
 * Theoretically the compiler should be able to optimise away most of it at compile time
 */
template<typename T, unsigned int I = 1, unsigned int J = 1, unsigned int K = 1, unsigned int W = 1>
class HostMacroProperty {
    /**
     * Data ptr to current section
     */
    ptrdiff_t offset;
    std::shared_ptr<HostMacroProperty_MetaData> metadata;

 public:
    /**
     * Root Constructor
     * @param _metadata Metadata struct
     */
    explicit HostMacroProperty(const std::shared_ptr<HostMacroProperty_MetaData>& _metadata);
    /**
     * Sub Constructor
     * @param _metadata Preallocated metadata struct, shared with other current instances of same host macro property
     * @param _offset Pointer to buffer in device memory
     */
    HostMacroProperty(const std::shared_ptr<HostMacroProperty_MetaData> &_metadata, const ptrdiff_t& _offset);

    /**
     * Access the next dimension of the array
     * @throws exception::OutOfBoundsException If i >= I. 
     * @throws exception::InvalidOperation If template arguments I, J, K , W are all 1. Which denotes the macro property has no dimensions remaining to be indexed. 
     */
    HostMacroProperty<T, J, K, W, 1> operator[](unsigned int i) const;
    /**
     * Read/Write access to the current element
     * @throws exception::InvalidOperation If template arguments I, J, K , W are not all 1. Which denotes the macro property has more dimensions remaining to be indexed. 
     */
    operator T();
    /**
     * Read-only access to the current element
     * @throws exception::InvalidOperation If template arguments I, J, K , W are not all 1. Which denotes the macro property has more dimensions remaining to be indexed. 
     */
    operator T() const;
    /**
     * Assign value
     * @param val New value of the element
     * @throws exception::InvalidOperation If template arguments I, J, K , W are not all 1. Which denotes the macro property has more dimensions remaining to be indexed. 
     */
    HostMacroProperty<T, I, J, K, W>& operator=(const T &val);
    /**
     * Zero's the selected area of the array
     */
    void zero();

    /**
     * Increment operators
     */
    HostMacroProperty<T, I, J, K, W>& operator++();
    T operator++(int);
    HostMacroProperty<T, I, J, K, W>& operator--();
    T operator--(int);

// Swig breaks if it see's auto return type
#ifndef SWIG
    /**
     * Arithmetic operators
     */
    template<typename T2>
    auto operator+(const T2& b) const;
    template<typename T2>
    auto operator-(const T2& b) const;
    template<typename T2>
    auto operator*(const T2& b) const;
    template<typename T2>
    auto operator/(const T2& b) const;
    template<typename T2>
    auto operator%(const T2& b) const;
#endif
    /**
     * Assignment operators
     */
    // HostMacroProperty<T, I, J, K, W>& operator=(const T& b);  // Defined above
    template<typename T2>
    HostMacroProperty<T, I, J, K, W>& operator+=(const T2& b);
    template<typename T2>
    HostMacroProperty<T, I, J, K, W>& operator-=(const T2& b);
    template<typename T2>
    HostMacroProperty<T, I, J, K, W>& operator*=(const T2& b);
    template<typename T2>
    HostMacroProperty<T, I, J, K, W>& operator/=(const T2& b);
    template<typename T2>
    HostMacroProperty<T, I, J, K, W>& operator%=(const T2& b);

 private:
    /**
     * Validate and download data, set changed flag
     */
    T& _get();
    /**
     * Validate and download data
     */
    T& _get() const;
};

#ifdef SWIG
/**
 * This template class is used for conveniently converting a multi-dimensional pointer to an array
 * Theoretically the compiler should be able to optimise away most of it at compile time
 */
template<typename T>
class HostMacroProperty_swig {
    /**
     * This exists in place of the template args found in HostMacroProperty
     */
    const std::array<unsigned int, 4> dimensions;
    /**
     * Data ptr to current section
     */
    ptrdiff_t offset;
    std::shared_ptr<HostMacroProperty_MetaData> metadata;

 public:
    /**
     * Root Constructor
     * @param _metadata Metadata struct
     */
    explicit HostMacroProperty_swig(const std::shared_ptr<HostMacroProperty_MetaData> &_metadata);
    /**
     * Sub Constructor
     * @param _metadata Preallocated metadata struct, shared with other current instances of same host macro property
     * @param _offset Pointer to buffer in device memory
     * @param _dimensions In place of HostMacroProperty's template args
     */
    HostMacroProperty_swig(const std::shared_ptr<HostMacroProperty_MetaData>& _metadata, const ptrdiff_t& _offset, const std::array<unsigned int, 4>& _dimensions);
    /**
     * Access the next dimension of the array
     * @throws exception::OutOfBoundsException If i >= I. 
     * @throws exception::InvalidOperation If the macro property has no dimensions remaining to be indexed. 
     */
    HostMacroProperty_swig<T> __getitem__(unsigned int i) const;
    /**
     * Set the item at position in the array
     * @note This is due to the fact Python doesn't less us override operator= directly.
     * @throws exception::OutOfBoundsException If i >= I.
     * @throws exception::InvalidOperation If the macro property has no dimensions remaining to be indexed.
     */
    void __setitem__(unsigned int i,  const T &val);
    /**
     * Explicit set method, as we lack operator= in python
     */
    void set(T val);

    int __int__();
    int64_t __long__();
    double __float__();
    bool __bool__();

    bool __eq__(const T& other) const;
    bool __ne__(const T &other) const;
    bool __lt__(const T& other) const;
    bool __le__(const T& other) const;
    bool __gt__(const T& other) const;
    bool __ge__(const T& other) const;

    /**
     * Required for iterable
     */
    unsigned int __len__() const { return dimensions[0]; }

    /**
     * Required for python extension
     */
    T get() const;
    /**
     * Zero's the selected area of the array
     */
    void zero();
};
#endif

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W>::HostMacroProperty(const std::shared_ptr<HostMacroProperty_MetaData> &_metadata)
: offset(0u)
, metadata(_metadata)
{ }
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W>::HostMacroProperty(const std::shared_ptr<HostMacroProperty_MetaData> &_metadata, const ptrdiff_t & _offset)
    : offset(_offset)
    , metadata(_metadata)
{ }
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, J, K, W, 1> HostMacroProperty<T, I, J, K, W>::operator[](unsigned int i) const {
    if (I == 1 && J == 1 && K == 1 && W == 1) {
        THROW exception::InvalidOperation("Indexing error, property has less dimensions.\n");
    } else if (i >= I) {
        THROW exception::OutOfBoundsException("Indexing error, out of bounds %u >= %u.\n", i, I);
    }
    // (i * J * K * W) + (j * K * W) + (k * W) + w
    return HostMacroProperty<T, J, K, W, 1>(metadata, offset + (i * J * K * W));
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W>::operator T() {
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    // Must assume changed
    metadata->has_changed = true;
    return *(reinterpret_cast<T*>(metadata->h_base_ptr) + offset);
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W>::operator T() const {
    metadata->download();
    return *(reinterpret_cast<T*>(metadata->h_base_ptr) + offset);
}

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
void HostMacroProperty<T, I, J, K, W>::zero() {
    if (metadata->h_base_ptr) {
        // Memset on host
        memset(reinterpret_cast<T*>(metadata->h_base_ptr) + offset, 0, I * J * K * W * metadata->type_size);
        metadata->has_changed = true;
    } else {
#if !defined(SEATBELTS) || SEATBELTS
        if (metadata->device_read_flag) {
            THROW flamegpu::exception::InvalidEnvProperty("The environment macro property '%s' was not found, "
                "in HostMacroProperty::zero()\n",
                metadata->property_name.c_str());
        }
#endif
        // Memset on device
        gpuErrchk(cudaMemset(reinterpret_cast<T*>(metadata->d_base_ptr) + offset, 0, I * J * K * W * metadata->type_size));
    }
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
T& HostMacroProperty<T, I, J, K, W>::_get() const {
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset];
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
T& HostMacroProperty<T, I, J, K, W>::_get() {
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    metadata->has_changed = true;
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset];
}


template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator++() {
    ++_get();
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
T HostMacroProperty<T, I, J, K, W>::operator++(int) {
    T &t = _get();
    T ret = t;
    ++t;
    return ret;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator--() {
    --_get();
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
T HostMacroProperty<T, I, J, K, W>::operator--(int) {
    T& t = _get();
    T ret = t;
    --t;
    return ret;
}

#ifndef SWIG
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
auto HostMacroProperty<T, I, J, K, W>::operator+(const T2& b) const {
    return _get() + b;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
auto HostMacroProperty<T, I, J, K, W>::operator-(const T2& b) const {
    return _get() - b;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
auto HostMacroProperty<T, I, J, K, W>::operator*(const T2& b) const {
    return _get() * b;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
auto HostMacroProperty<T, I, J, K, W>::operator/(const T2& b) const {
    return _get() / b;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
auto HostMacroProperty<T, I, J, K, W>::operator%(const T2& b) const {
    return _get() % b;
}
#endif

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator+=(const T2& b) {
    _get() += b;
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator-=(const T2& b) {
    _get() -= b;
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator*=(const T2& b) {
    _get() *= b;
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator/=(const T2& b) {
    _get() /= b;
    return *this;
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
template<typename T2>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator%=(const T2& b) {
    _get() %= b;
    return *this;
}

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W>& HostMacroProperty<T, I, J, K, W>::operator=(const T& val) {
    if (I != 1 || J != 1 || K != 1 || W != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    reinterpret_cast<T*>(metadata->h_base_ptr)[offset] = val;
    metadata->has_changed = true;
    return *this;
}

// SWIG versions
#ifdef SWIG
template<typename T>
HostMacroProperty_swig<T>::HostMacroProperty_swig(const std::shared_ptr<HostMacroProperty_MetaData>& _metadata)
    : dimensions(_metadata->dims)
    , offset(0u)
    , metadata(_metadata)
{ }
template<typename T>
HostMacroProperty_swig<T>::HostMacroProperty_swig(const std::shared_ptr<HostMacroProperty_MetaData>& _metadata, const ptrdiff_t& _offset, const std::array<unsigned int, 4> &_dimensions)
    : dimensions(_dimensions)
    , offset(_offset)
    , metadata(_metadata)
{ }
template<typename T>
HostMacroProperty_swig<T> HostMacroProperty_swig<T>::__getitem__(unsigned int i) const {
    if (dimensions[0] == 1 && dimensions[1] == 1 && dimensions[2] == 1 && dimensions[3] == 1) {
        THROW exception::InvalidOperation("Indexing error, property has less dimensions.\n");
    } else if (i >= dimensions[0]) {
        THROW exception::OutOfBoundsException("Indexing error, out of bounds %u >= %u.\n", i, dimensions[0]);
    }
    // (i * J * K * W) + (j * K * W) + (k * W) + w
    return HostMacroProperty_swig<T>(metadata, offset + (i * dimensions[1] * dimensions[2] * dimensions[3]), { dimensions[1], dimensions[2], dimensions[3], 1 });
}

template<typename T>
void HostMacroProperty_swig<T>::zero() {
    if (metadata->h_base_ptr) {
        // Memset on host
        memset(reinterpret_cast<T*>(metadata->h_base_ptr) + offset, 0, dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3] * metadata->type_size);
        metadata->has_changed = true;
    } else {
        // Memset on device
        gpuErrchk(cudaMemset(reinterpret_cast<T*>(metadata->d_base_ptr) + offset, 0, dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3] * metadata->type_size));
    }
}
template<typename T>
void HostMacroProperty_swig<T>::set(T val) {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    reinterpret_cast<T*>(metadata->h_base_ptr)[offset] = val;
    metadata->has_changed = true;
}
template<typename T>
void HostMacroProperty_swig<T>::__setitem__(unsigned int i, const T& val) {
    if (dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    } else if (i >= dimensions[0]) {
        THROW exception::InvalidOperation("Indexing out of bounds %u >= %u.\n", i, dimensions[0]);
    }
    metadata->download();
    unsigned int t_offset = offset + (i * dimensions[1] * dimensions[2] * dimensions[3]);
    reinterpret_cast<T*>(metadata->h_base_ptr)[t_offset] = val;
    metadata->has_changed = true;
}
template<typename T>
int HostMacroProperty_swig<T>::__int__() {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return static_cast<int>(reinterpret_cast<T*>(metadata->h_base_ptr)[offset]);
}
template<typename T>
int64_t HostMacroProperty_swig<T>::__long__() {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return static_cast<int64_t>(reinterpret_cast<T*>(metadata->h_base_ptr)[offset]);
}
template<typename T>
double HostMacroProperty_swig<T>::__float__() {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return static_cast<double>(reinterpret_cast<T*>(metadata->h_base_ptr)[offset]);
}
template<typename T>
bool HostMacroProperty_swig<T>::__bool__() {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return static_cast<bool>(reinterpret_cast<T*>(metadata->h_base_ptr)[offset]);
}
template<typename T>
bool HostMacroProperty_swig<T>::__eq__(const T& other) const {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset] == other;
}
template<typename T>
bool HostMacroProperty_swig<T>::__ne__(const T& other) const {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset] != other;
}
template<typename T>
bool HostMacroProperty_swig<T>::__lt__(const T& other) const {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset] < other;
}
template<typename T>
bool HostMacroProperty_swig<T>::__le__(const T& other) const {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset] <= other;
}
template<typename T>
bool HostMacroProperty_swig<T>::__gt__(const T& other) const {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset] > other;
}
template<typename T>
bool HostMacroProperty_swig<T>::__ge__(const T& other) const {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset] >= other;
}
// template<typename T>
// T HostMacroProperty_swig<T>::__mod__(const T& other) {
//     if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
//         THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
//     }
//     metadata->download();
//     return reinterpret_cast<T*>(metadata->h_base_ptr)[offset] % other;
// }
template<typename T>
T HostMacroProperty_swig<T>::get() const {
    if (dimensions[0] != 1 || dimensions[1] != 1 || dimensions[2] != 1 || dimensions[3] != 1) {
        THROW exception::InvalidOperation("Indexing error, property has more dimensions.\n");
    }
    metadata->download();
    return reinterpret_cast<T*>(metadata->h_base_ptr)[offset];
}
#endif

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTMACROPROPERTY_CUH_
