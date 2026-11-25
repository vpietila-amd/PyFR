// The LU inverse in separate kernels for the factorisation and the solves:
// for fp64 hand written ones with MFMA, otherwise the generated kernel of
// the frozen baseline split in two.
//
// LUINV_HEADER names the generated kernels for the matrix size, precisions
// and options, and LUINV_MFMA selects the hand written ones; see
// __init__.py.

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#define LUINV_STR_(x) #x
#define LUINV_STR(x) LUINV_STR_(x)

#if defined(LUINV_HEADER)
extern "C"
{
#include LUINV_STR(luinv/hip_v2/generated/LUINV_HEADER)
}
#endif

// The hand written MFMA path: build, factorise a matrix per block and solve
// 16 columns per wave, in LUINV_FTYPE (double or float) from Jacobians in
// LUINV_STYPE to inverses in LUINV_OTYPE; see factor.hpp and solve.hpp
#if defined(LUINV_MFMA)
#include "luinv/hip_v2/common.hpp"
#include "luinv/hip_v2/factor.hpp"

typedef LUINV_FTYPE luinv_ftype;
typedef LUINV_STYPE luinv_stype;
typedef LUINV_OTYPE luinv_otype;

extern "C" __global__ __launch_bounds__(256)
void lu_build_hip_v2(ixdtype_t nx, LUINV_SCALE_T scale,
                     const luinv_stype* __restrict__ src_v, ixdtype_t ldsrc,
                     const ixdtype_t* __restrict__ wk_v,
                     luinv_ftype* __restrict__ ws_v)
{
    luinv::lu_build<LUINV_N, LUINV_NP, LUINV_WSZ>(
        ixdtype_t(LUINV_BLOCK_ID), nx, scale, src_v, ldsrc, wk_v, ws_v
    );
}

extern "C" __global__ __launch_bounds__(64*LUINV_K1W, LUINV_K1MB)
void lu_factor_mfma_hip_v2(ixdtype_t _nx, const ixdtype_t* __restrict__ wk_v,
                           luinv_ftype* __restrict__ ws_v,
                           int16_t* __restrict__ perm_v,
                           unsigned long long* __restrict__ ticks)
{
    luinv::lu_factor<LUINV_NP, LUINV_WSZ, LUINV_K1W, LUINV_S, LUINV_IOL>(
        ixdtype_t(LUINV_BLOCK_ID), wk_v, ws_v, perm_v, ticks
    );
}

extern "C" __global__ __launch_bounds__(64*LUINV_WPB, LUINV_K2MB)
void lu_solve_mfma_hip_v2(ixdtype_t _nx, luinv_otype* __restrict__ inv_v,
                          const ixdtype_t* __restrict__ inv_vix,
                          const int16_t* __restrict__ perm_v,
                          const ixdtype_t* __restrict__ wk_v,
                          const luinv_ftype* __restrict__ ws_v)
{
    luinv::lu_solve_mfma<LUINV_N, LUINV_NP, LUINV_WSZ, LUINV_WPB, LUINV_TR,
                         LUINV_TC, LUINV_PADC>(
        ixdtype_t(LUINV_BLOCK_ID), inv_v, inv_vix, perm_v, wk_v, ws_v
    );
}
#endif
