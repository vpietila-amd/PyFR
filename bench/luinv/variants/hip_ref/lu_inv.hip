// The frozen baseline kernel as rendered by PyFR, compiled as hand written
// HIP.  It checks that the raw HIP route reproduces the baseline exactly,
// and is a starting point for incremental changes to the generated code.
//
// LUINV_HEADER names the rendering for the matrix size and precisions; see
// generate() in __init__.py.

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#define LUINV_STR_(x) #x
#define LUINV_STR(x) LUINV_STR_(x)

extern "C"
{
#include LUINV_STR(luinv/hip_ref/generated/LUINV_HEADER)
}
