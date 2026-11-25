// Operand layouts of the fp64 matrix fused multiply-add of gfx90a/gfx94x,
// v_mfma_f64_16x16x4_f64, which a wave issues as D = A.B + C with A 16x4,
// B 4x16 and C, D 16x16.  Lane l of the wave holds
//   A[l % 16][l / 16],  B[l / 16][l % 16],  C[l / 16 + 4*v][l % 16]
// for v = 0..3 of the four accumulator values; test_mfma.py checks these.
#pragma once

namespace luinv
{

typedef double f64x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ f64x4 mfma_f64_16x16x4(double a, double b,
                                                  f64x4 c)
{
    return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}

// Row and column of a matrix element the lane holds in each operand
__device__ __forceinline__ int mfma_lrow(int lane) { return lane % 16; }
__device__ __forceinline__ int mfma_lk(int lane) { return lane / 16; }
__device__ __forceinline__ int mfma_crow(int lane, int v)
{
    return lane / 16 + 4*v;
}

}
