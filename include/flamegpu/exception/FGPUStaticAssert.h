#ifndef INCLUDE_FLAMEGPU_EXCEPTION_FGPUSTATICASSERT_H_
#define INCLUDE_FLAMEGPU_EXCEPTION_FGPUSTATICASSERT_H_

#include <cstdint>

/**
 * These are taken from MSVCs std to allow us to perform static assertions
 */
namespace FGPU_SA {
/**
 * Convenient template for integral constant types
 */
template<class _Ty,
    _Ty _Val>
struct integral_constant {
    static constexpr _Ty value = _Val;

    typedef _Ty value_type;
    typedef integral_constant<_Ty, _Val> type;

    constexpr operator value_type() const noexcept {
        return (value);  // return stored value
    }

    constexpr value_type operator()() const noexcept {
        return (value);  // return stored value
    }
};
typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;
/**
 * Base class for type predicates
 */
template<bool _Val>
struct _Cat_base
    : integral_constant<bool, _Val> {
};
/**
 * TEMPLATE CLASS is_same
 * determine whether _Ty1 and _Ty2 are the same type
 */
template<class _Ty1,
    class _Ty2>
struct is_same
    : false_type { };
template<class _Ty1>
struct is_same<_Ty1, _Ty1>
    : true_type { };
/**
 * Determine whether _Ty satisfies HostRandom's RealType requirements
 */
template<class _Ty>
struct _Is_RealType
    : _Cat_base<is_same<_Ty, float>::value
    || is_same<_Ty, double>::value> {
};
/**
 * Determine whether _Ty satisfies HostRandom's IntType requirements
 */
template<class _Ty>
struct _Is_IntType
    : _Cat_base<is_same<_Ty, unsigned char>::value
    || is_same<_Ty, char>::value
    || is_same<_Ty, signed char>::value
    || is_same<_Ty, uint16_t>::value
    || is_same<_Ty, int16_t>::value
    || is_same<_Ty, uint32_t>::value
    || is_same<_Ty, int32_t>::value
    || is_same<_Ty, uint64_t>::value
    || is_same<_Ty, int64_t>::value> {
};
}  // namespace FGPU_SA

#endif  // INCLUDE_FLAMEGPU_EXCEPTION_FGPUSTATICASSERT_H_
