// The optimisation playground kernel: the baseline rendering plus the
// staged hand-written changes applied by generate() in __init__.py.  With no
// options set it reproduces the baseline (and hip_ref) exactly; the
// LUINV_* environment variables select the experiments.
//
// LUINV_HEADER names the rendering for the matrix size, precisions and the
// active options; see generate() in __init__.py.

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#define LUINV_STR_(x) #x
#define LUINV_STR(x) LUINV_STR_(x)

extern "C"
{
#include LUINV_STR(luinv/hip_opt/generated/LUINV_HEADER)
}
