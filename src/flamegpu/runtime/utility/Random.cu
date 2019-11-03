#include "Random.h"

#include<ctime>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>
#include <cstdio>

/**
 * Anonymous namespace to hide __device__ declarations
 */
namespace {
	__device__ curandState *d_random_state;
	__device__ Random::size_type d_random_size;
    curandState *hd_random_state;
    Random::size_type hd_random_size;
}
/**
 * Static member vars
 */
unsigned long long Random::seed = 0;
Random::size_type Random::length = 0;
float Random::shrinkModifier = 0;
float Random::growthModifier = 0;
/**
 * Member fns
 */
unsigned long long Random::seedFromTime()
{
    return static_cast<unsigned long long>(time(nullptr));
}
void Random::init(const unsigned long long &_seed)
{
    Random::seed = _seed;
    free();
}
void Random::free()
{   
    //Clear size
    length = 0;
    hd_random_size = 0;
    if (cudaMemcpyToSymbol(d_random_size, &hd_random_size, sizeof(Random::size_type)) != cudaSuccess)
        printf("(%s:%d) CUDA Error initialising curand.", __FILE__, __LINE__);
    //Release old
    if (hd_random_state != nullptr && cudaFree(hd_random_state) != cudaSuccess)
        printf("(%s:%d) CUDA Error Random::~Random().", __FILE__, __LINE__);
    //Update pointers
    hd_random_state = nullptr;
    if (cudaMemcpyToSymbol(d_random_state, &hd_random_state, sizeof(curandState*)) != cudaSuccess)
        printf("(%s:%d) CUDA Error Random::~Random().", __FILE__, __LINE__);
}

bool Random::resize(const size_type &_length)
{
    auto t_length = length;
    while (t_length < _length) {
        t_length  = static_cast<Random::size_type>(t_length * growthModifier);
        if(shrinkModifier < 1.0f) {
            while(t_length * shrinkModifier > _length)
            {
                t_length = static_cast<Random::size_type>(t_length * shrinkModifier);
            }
        }        
    }
    resizeDeviceArray(t_length);
    return t_length != length;
}
__global__ void init_curand(unsigned long threadCount, unsigned long long seed, Random::size_type offset) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < threadCount)
        curand_init(seed, offset + id, 0, &d_random_state[offset + id]);
}
/**
 * Should probably implement shrink differently, otherwise a curand state being shrunk and regrowed seed will reset
 * Perhaps maintain longest length curand on host and only shrink device side
 */
void Random::resizeDeviceArray(const size_type &_length)
{
    if(_length > length)
    {
        //Growing array
        curandState *t_hd_random_state = nullptr;
        //Allocate new
        if (cudaMalloc(&t_hd_random_state, _length * sizeof(curandState)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        //Copy old->new[****    ]
        if (hd_random_state != nullptr)
            if (cudaMemcpy(t_hd_random_state, hd_random_state, length * sizeof(curandState), cudaMemcpyDeviceToDevice))
                printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        //Init new[    ****]
        unsigned int initThreads = 512;
        unsigned int initBlocks = (_length - length / initThreads) + 1;
        init_curand<<<initBlocks, initThreads>>>(_length - length, seed, length);
        //Update pointers
        hd_random_state = t_hd_random_state;
        if (cudaMemcpyToSymbol(d_random_state, &hd_random_state, sizeof(curandState*)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
    }
    else
    {
        //Shrinking array
        curandState *t_hd_random_state = nullptr;
        //Allocate new
        if (cudaMalloc(&t_hd_random_state, _length * sizeof(curandState)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        //Copy old->new
        if (hd_random_state != nullptr) 
            if(cudaMemcpy(t_hd_random_state, hd_random_state, _length * sizeof(curandState), cudaMemcpyDeviceToDevice))
                printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        //Update pointers
        hd_random_state = t_hd_random_state;
        if (cudaMemcpyToSymbol(d_random_state, &hd_random_state, sizeof(curandState*)) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
        //Release old
        if (hd_random_state!=nullptr && cudaFree(hd_random_state) != cudaSuccess)
            printf("(%s:%d) CUDA Error Random::resizeDeviceArray().", __FILE__, __LINE__);
    }
    //Update length
    length = _length;
    hd_random_size = _length;
    if (cudaMemcpyToSymbol(d_random_size, &hd_random_size, sizeof(Random::size_type)) != cudaSuccess)
        printf("(%s:%d) CUDA Error initialising curand.", __FILE__, __LINE__);
}
void Random::setGrowthModifier(float _growthModifier) {
    assert(growthModifier > 1.0);
    Random::growthModifier = _growthModifier;
}
float Random::getGrowthModifier() {
    return Random::growthModifier;
}
void Random::setShrinkModifier(float _shrinkModifier) {
    assert(shrinkModifier > 0.0);
    assert(shrinkModifier <= 1.0);
    Random::shrinkModifier = _shrinkModifier;
}
float Random::getShrinkModifier() {
    return Random::shrinkModifier;
}