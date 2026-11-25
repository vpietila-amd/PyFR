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

// The hand written solves with fp64 MFMA, for matrices padded to a multiple
// of 16; see solve.hpp
#if defined(LUINV_MFMA)
#include "luinv/hip_v2/solve.hpp"

extern "C" __global__ __launch_bounds__(64*LUINV_WPB, LUINV_K2MB)
void lu_solve_mfma_hip_v2(ixdtype_t _nx, double* __restrict__ inv_v,
                          const ixdtype_t* __restrict__ inv_vix,
                          const int16_t* __restrict__ perm_v,
                          const ixdtype_t* __restrict__ wk_v,
                          const double* __restrict__ ws_v)
{
    luinv::lu_solve_mfma<LUINV_N, LUINV_NP, LUINV_WSZ, LUINV_WPB, LUINV_TR,
                         LUINV_TC, LUINV_PADC>(
        ixdtype_t(PYFR_BLOCK_ID), inv_v, inv_vix, perm_v, wk_v, ws_v
    );
}

extern "C" __global__ __launch_bounds__(256)
void lu_pack_hip_v2(ixdtype_t _nx, const ixdtype_t* __restrict__ wk_v,
                    double* __restrict__ ws_v)
{
    luinv::lu_pack<LUINV_NP, LUINV_WSZ>(ixdtype_t(PYFR_BLOCK_ID), wk_v, ws_v);
}
#endif
