
#include <stdio.h>
#include "curve.h"
#include "assert.h"

#define CURVE_MAX_VARIABLES 			32 							//!< Default maximum number of cuRVE variables (must be a power of 2)
#define VARIABLE_DISABLED 				0
#define VARIABLE_ENABLED 				1
#define NAMESPACE_NONE 					0

#ifdef DEBUG
#define CUDA_SAFE_CALL(x)                                                                               		\
{                                                                                                         		\
	cudaError_t error = (x);                                                                                	\
	if (error != cudaSuccess && error != cudaErrorNotReady)                                                 	\
	{                                                                                                       	\
		printf("%s.%s.%d: 0x%x (%s)\n", __FILE__, __FUNCTION__, __LINE__, error, cudaGetErrorString(error));  	\
		cudaGetLastError();                                                                                   	\
		exit(1);                                                                                              	\
	}                                                                                                       	\
}
#else
#define CUDA_SAFE_CALL(x) (x)
#endif




inline void cudaCheckError(cudaError_t error, char* file, char* function, int line){
	if (error != cudaSuccess && error != cudaErrorNotReady)
	{
		printf("%s.%s.%d: 0x%x (%s)\n", file, function, line, error, cudaGetErrorString(error));
		cudaGetLastError();
		exit(1);
	}
}


unsigned int h_namespace;

CurveVariableHash h_hashes[CURVE_MAX_VARIABLES];				//Host array of the hash values of registered variables
void* h_d_variables[CURVE_MAX_VARIABLES];						//Host array of pointer to device memory addresses for variable storage
int	h_states[CURVE_MAX_VARIABLES];								//Host array of the states of registered variables

__constant__ CurveNamespaceHash d_namespace;
__constant__ CurveVariableHash d_hashes[CURVE_MAX_VARIABLES];	//Device array of the hash values of registered variables
__device__ float* d_variables[CURVE_MAX_VARIABLES];				//Device array of pointer to device memory addresses for variable storage
__constant__ int* d_states[CURVE_MAX_VARIABLES];				//Device array of the states of registered variables

__device__ curveDeviceError d_curve_error;
curveHostError h_curve_error;


/*private functions*/

__device__ __inline__ CurveVariable getVariable(const CurveVariableHash variable_hash); /* loop unrolling of hash collision detection */


__device__ __inline__ CurveVariable getVariable(const CurveVariableHash variable_hash)
{
	const CurveVariableHash hash = variable_hash + d_namespace;
	for (unsigned int x=0; x< CURVE_MAX_VARIABLES; x++)
	{
		const CurveVariable i = ((hash + x) & (CURVE_MAX_VARIABLES-1));
		const CurveVariableHash h = d_hashes[i];
		if ( h == hash)
			return i;
	}
	return UNKNOWN_CURVE_VARIABLE;
}


/* header implementations */

__host__ CurveVariable curveGetVariableHandle(CurveVariableHash variable_hash)
{
    unsigned int i, n;

    variable_hash += h_namespace;
    n = 0;
	i = (variable_hash) % CURVE_MAX_VARIABLES;

	while (h_hashes[i] != 0)
    {
        if (h_hashes[i] == variable_hash)
		{
			return i;
		}
        n += 1;
        if (n >= CURVE_MAX_VARIABLES)
		{
			break;
		}
        i += 1;
        if (i >= CURVE_MAX_VARIABLES)
        {
            i = 0;
        }
    }
	return UNKNOWN_CURVE_VARIABLE;
}





__host__ void curveInit()
{
	unsigned int *_d_hashes;
	float** _d_variables;
    int** _d_states;

	//namespace
	h_namespace = NAMESPACE_NONE;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_namespace, &h_namespace, sizeof(unsigned int)));

	//get a host pointer to d_hashes and d_variables
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_hashes, d_hashes));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_variables, d_variables));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));

	//set values of hash table to 0 on host and device
	memset(h_hashes, 0,  sizeof(unsigned int)*CURVE_MAX_VARIABLES);
	memset(h_states, 0,  sizeof(int)*CURVE_MAX_VARIABLES);

	//initialise data to 0 on device
	CUDA_SAFE_CALL(cudaMemset(_d_hashes, 0, sizeof(unsigned int)*CURVE_MAX_VARIABLES));
	CUDA_SAFE_CALL(cudaMemset(_d_variables, 0, sizeof(void*)*CURVE_MAX_VARIABLES));
	CUDA_SAFE_CALL(cudaMemset(_d_states, VARIABLE_DISABLED, sizeof(int)*CURVE_MAX_VARIABLES));

	curveClearErrors();
}

__host__ CurveVariable curveRegisterVariableByHash(CurveVariableHash variable_hash, void * d_ptr)
{
	unsigned int i, n;
	unsigned int *_d_hashes;
	void** _d_variables;
	int** _d_states;

	n = 0;
	variable_hash += h_namespace;
	i = (variable_hash) % CURVE_MAX_VARIABLES;

	while (h_hashes[i] != 0)
	{
		n += 1;
		if (n >= CURVE_MAX_VARIABLES)
		{
			h_curve_error = CURVE_ERROR_TOO_MANY_VARIABLES;
			return UNKNOWN_CURVE_VARIABLE;
		}
		i += 1;
		if (i >= CURVE_MAX_VARIABLES)
		{
			i = 0;
		}
	}
	h_hashes[i] = variable_hash;

	//get a host pointer to d_hashes and d_variables
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_hashes, d_hashes));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_variables, d_variables));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));

	//copy hash to device
	CUDA_SAFE_CALL(cudaMemcpy(&_d_hashes[i], &h_hashes[i], sizeof(unsigned int), cudaMemcpyHostToDevice));

	//make a host copy of the pointer and copy to the device
	h_d_variables[i] = d_ptr;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_variables[i], &h_d_variables[i], sizeof(float*), cudaMemcpyHostToDevice));

	//set the state to enabled
	h_states[i] = VARIABLE_ENABLED;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_states[i], &h_states[i], sizeof(int), cudaMemcpyHostToDevice));

	printf("Var with hash is %u at index %d with %d collisions\n", variable_hash, i, n);

	return i;
}

/**
 * TODO: Does un-registering imply that other variable with collisions will no longer be found. I.e. do you need to re-register all other variable when one is removed.
 */
__host__ void curveUnregisterVariableByHash(CurveVariableHash variable_hash)
{
	unsigned int *_d_hashes;
	void** _d_variables;
	int** _d_states;
	CurveVariable cv;

	//get the curve variable
	cv = curveGetVariableHandle(variable_hash);

	//error checking
	if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
		return;
	}

	//get a host pointer to d_hashes and d_variables
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_hashes, d_hashes));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_variables, d_variables));
	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));

	//clear hash location on host and copy hash to device
	h_hashes[cv] = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_hashes[cv], &h_hashes[cv], sizeof(unsigned int), cudaMemcpyHostToDevice));

	//set a host pointer to null and copy to the device
	h_d_variables[cv] = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_variables[cv], &h_d_variables[cv], sizeof(float*), cudaMemcpyHostToDevice));

	//return the state to disabled
	h_states[cv] = VARIABLE_DISABLED;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));

	printf("Var with hash %u has been un-registered\n", variable_hash);
}



__host__ void curveDisableVariableByHash(CurveVariableHash variable_hash)
{
	CurveVariable cv = curveGetVariableHandle(variable_hash);
	int** _d_states;

	//error checking
	if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
		return;
	}

	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));
	h_states[cv] = VARIABLE_DISABLED;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void curveEnableVariableByHash(CurveVariableHash variable_hash)
{
	CurveVariable cv = curveGetVariableHandle(variable_hash);
	int** _d_states;

	//error checking
	if (cv == UNKNOWN_CURVE_VARIABLE)
	{
		h_curve_error = CURVE_ERROR_UNKNOWN_VARIABLE;
		return;
	}

	CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&_d_states, d_states));
	h_states[cv] = VARIABLE_ENABLED;
	CUDA_SAFE_CALL(cudaMemcpy(&_d_states[cv], &h_states[cv], sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void curveSetNamespaceByHash(CurveNamespaceHash namespace_hash)
{
	h_namespace = namespace_hash;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_namespace, &h_namespace, sizeof(unsigned int)));
}

__host__ void curveSetDefaultNamespace()
{
	h_namespace = NAMESPACE_NONE;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_namespace, &h_namespace, sizeof(unsigned int)));
}

__device__ void* curveGetVariablePtrByHash(const CurveVariableHash variable_hash, size_t offset)
{
    CurveVariable cv;

	cv = getVariable(variable_hash);

    //error checking
    if (cv == UNKNOWN_CURVE_VARIABLE)
    {
    	d_curve_error = CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE;
    	return NULL;
    }
    if(!d_states[cv])
    {
    	d_curve_error = CURVE_DEVICE_ERROR_VARIABLE_DISABLED;
    	return NULL;
    }

	//return a generic pointer to variable address for given offset
	//TODO: Add vector length checking
    return &(d_variables[cv])[offset];
}


/* errors */
void __device__ curvePrintLastDeviceError(const char* file, const char* function, const int line){
	if (d_curve_error != CURVE_DEVICE_ERROR_NO_ERRORS)
	{
		printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)d_curve_error, curveGetDeviceErrorString(d_curve_error));
	}
}

void __host__ curvePrintLastHostError(const char* file, const char* function, const int line){
	if (h_curve_error != CURVE_ERROR_NO_ERRORS)
	{
		printf("%s.%s.%d: cuRVE Host Error %d (%s)\n", file, function, line, (unsigned int)h_curve_error, curveGetHostErrorString(h_curve_error));
	}
}

void __host__ curvePrintErrors(const char* file, const char* function, const int line){
	curveDeviceError d_curve_error_local;

	curvePrintLastHostError(file, function, line);

	//check device errors
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&d_curve_error_local, d_curve_error, sizeof(curveDeviceError)));
	if (d_curve_error_local != CURVE_DEVICE_ERROR_NO_ERRORS)
	{
		printf("%s.%s.%d: cuRVE Device Error %d (%s)\n", file, function, line, (unsigned int)d_curve_error_local, curveGetDeviceErrorString(d_curve_error_local));
	}
}

__device__ __host__ const char* curveGetDeviceErrorString(curveDeviceError e)
{
	switch (e){
		case(CURVE_DEVICE_ERROR_NO_ERRORS):
				return "No cuRVE errors";
		case(CURVE_DEVICE_ERROR_UNKNOWN_VARIABLE):
				return "Unknown cuRVE variable in current namespace";
		case(CURVE_DEVICE_ERROR_VARIABLE_DISABLED):
				return "cuRVE variable is disabled";
		default:
			return "Unspecified cuRVE error";
	}
}

__host__ const char* curveGetHostErrorString(curveHostError e)
{
	switch (e){
		case(CURVE_ERROR_NO_ERRORS):
				return "No cuRVE errors";
		case(CURVE_ERROR_UNKNOWN_VARIABLE):
				return "Unknown cuRVE variable";
		case(CURVE_ERROR_TOO_MANY_VARIABLES):
				return "Too many cuRVE variables";
		default:
			return "Unspecified cuRVE error";
	}
}

__device__ curveDeviceError curveGetLastDeviceError()
{
	return d_curve_error;
}

__host__ curveHostError curveGetLastHostError()
{
	return h_curve_error;
}


__host__ void curveClearErrors()
{
	curveDeviceError curve_error_none;

	curve_error_none = CURVE_DEVICE_ERROR_NO_ERRORS;
	h_curve_error  = CURVE_ERROR_NO_ERRORS;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_curve_error, &curve_error_none, sizeof(curveDeviceError)));

}

