// The two kernel LU inverse: lu_factor_hip_v2 factorises each matrix into
// its workspace, then lu_solve_hip_v2 builds the inverse a column chunk per
// block, with the chunks of all the matrices in one launch.
//
// LUINV_HEADER names the generated kernels for the matrix size, precisions
// and options; see generate() in __init__.py.

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#define LUINV_STR_(x) #x
#define LUINV_STR(x) LUINV_STR_(x)

extern "C"
{
#include LUINV_STR(luinv/hip_v2/generated/LUINV_HEADER)
}
