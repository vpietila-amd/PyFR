// Checks the operand layouts in mfma.hpp: one wave forms the 16x16 product
// of a 16xK and a Kx16 row-major matrix, four columns of A at a time.

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#include "luinv/hip_v2/mfma.hpp"

extern "C" __global__ void mfma_gemm(int k, const double* a, const double* b,
                                     double* d)
{
    using namespace luinv;
    int l = threadIdx.x;
    f64x4 acc = {0, 0, 0, 0};

    for (int k0 = 0; k0 < k; k0 += 4)
    {
        double av = a[k*mfma_lrow(l) + k0 + mfma_lk(l)];
        double bv = b[16*(k0 + mfma_lk(l)) + mfma_lrow(l)];
        acc = mfma_f64_16x16x4(av, bv, acc);
    }

    for (int v = 0; v < 4; v++)
        d[16*mfma_crow(l, v) + mfma_lrow(l)] = acc[v];
}
