#ifndef INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVEINSTANCE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVEINSTANCE_H_

#include "curve.h"

/** @brief    A cuRVE instance.   
 * 
 * cuRVE is a C library and this singleton class acts as a mechanism to ensure that any reference to the library is handled correctly.
 * For example multiple objects may which to request that curve is initialised. This class will ensure that this function call is only made once the first time that a cuRVEInstance is required.
 */
class cuRVEInstance {
 private:

    /** @brief    Default constructor.    
     *
     *  Private destructor to prevent this singleton being created more than once. Classes requiring a cuRVEInstance object should instead use the getInstance() method.
     *  This ensure that curveInit is only ever called once by the program.
     */
    cuRVEInstance() { 
        curveInit(); 
    };
    
    ~cuRVEInstance() {};
 public:

    /**
     * @brief    Gets the instance.
     *
     * @return    A new instance if this is the first request for an instance otherwise an existing instance.
     */
    static cuRVEInstance& getInstance(){
        static cuRVEInstance c;
        return c;
    }

};

#endif // INCLUDE_FLAMEGPU_RUNTIME_CURVE_CURVEINSTANCE_H_

