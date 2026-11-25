// Checks the operand layouts in mfma.hpp: one wave forms the 16x16 product
// of a 16xK and a Kx16 row-major matrix, four columns of A at a time.

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#include "luinv/hip_v2/mfma.hpp"

template<typename T>
__device__ void mfma_gemm(int k, const T* a, const T* b, T* d)
{
    using namespace luinv;
    int l = threadIdx.x;
    vec4<T> acc = {0, 0, 0, 0};

    for (int k0 = 0; k0 < k; k0 += 4)
    {
        T av = a[k*mfma_lrow(l) + k0 + mfma_lk(l)];
        T bv = b[16*(k0 + mfma_lk(l)) + mfma_lrow(l)];
        acc = mfma_16x16x4(av, bv, acc);
    }

    for (int v = 0; v < 4; v++)
        d[16*mfma_crow<T>(l, v) + mfma_lrow(l)] = acc[v];
}

extern "C" __global__ void mfma_gemm_f64(int k, const double* a,
                                         const double* b, double* d)
{
    mfma_gemm(k, a, b, d);
}

extern "C" __global__ void mfma_gemm_f32(int k, const float* a,
                                         const float* b, float* d)
{
    mfma_gemm(k, a, b, d);
}
