// Inverse of a matrix from its LU factorisation with fp64 MFMA.
//
// The factorisation leaves P A = L U in the workspace, as an NP x NP row
// major matrix with the lower L on and below the diagonal and the unit
// upper U above it, and the row order P in perm.  The inverse is
// A^-1 = U^-1 L^-1 P, so with Z = U^-1 L^-1 solving L U Z = I column by
// column, column k of Z is column perm[k] of A^-1.
//
// lu_pack first copies the factor into the workspace after it as NB x NB
// blocks of 16 x 16 (NB = NP/16) in the order of the A operand of the
// MFMA, with the inverses of the diagonal blocks of L in place of them and
// those of U after all the blocks.  A wave then loads the operands of a
// block as 2 KB in a row.
//
// lu_solve_mfma has every wave solve for 16 columns of Z on its own,
// holding them in registers as NB accumulator tiles of 16 x 16; nothing is
// shared between waves, so there is no LDS and no barrier.  Each block row
// i of a triangular solve is one product with the inverse of the diagonal
// block, D_ii^-1 Z_i, then the updates Z_j -= M_ji Z_i of the block rows
// still to be solved.
//
// In the accumulator layout of v_mfma_f64_16x16x4_f64 lane l holds
// Z[l/16 + 4v][l%16] for v = 0..3, which is exactly the B operand of the
// 4-deep slice v of a product with Z, so a solved tile feeds the updates
// straight from its registers.
#pragma once

#include "luinv/hip_v2/mfma.hpp"

namespace luinv
{

// Offsets in the packed factor of block (j, i) and of the inverse of U_ii
template<int NB>
__device__ __forceinline__ int pk_block(int j, int i) { return 256*(NB*j + i); }

template<int NB>
__device__ __forceinline__ int pk_uinv(int i) { return 256*(NB*NB + i); }

// Position in a packed block of element (r, c): the lane holding it and
// the slice of the A operand it is in
__device__ __forceinline__ int pk_ix(int r, int c)
{
    return 4*(r + 16*(c % 4)) + c / 4;
}

// acc +/- M . Z for the packed 16 x 16 block M at m
template<bool Neg>
__device__ __forceinline__ f64x4 block_mma(const double* m, int lane,
                                           f64x4 z, f64x4 acc)
{
    f64x4 a = *reinterpret_cast<const f64x4*>(m + 4*lane);

#pragma unroll
    for (int s = 0; s < 4; s++)
        acc = mfma_f64_16x16x4(Neg ? -a[s] : a[s], z[s], acc);

    return acc;
}

// Solve L U Z = I for the 16 columns of Z from column block cb, given the
// packed factor pk
template<int NB>
__device__ __forceinline__ void solve_columns(const double* pk, int cb,
                                              int lane, f64x4 (&z)[NB])
{
    const f64x4 zero = {0, 0, 0, 0};

    // Start from the columns of the identity
#pragma unroll
    for (int j = 0; j < NB; j++)
    {
#pragma unroll
        for (int v = 0; v < 4; v++)
            z[j][v] = (j == cb && mfma_crow(lane, v) == mfma_lrow(lane));
    }

    // Forward: L^-1 is lower triangular so the block rows above cb stay zero
#pragma unroll
    for (int i = 0; i < NB; i++)
    {
        if (i < cb)
            continue;

        z[i] = block_mma<false>(pk + pk_block<NB>(i, i), lane, z[i], zero);

#pragma unroll
        for (int j = i + 1; j < NB; j++)
            z[j] = block_mma<true>(pk + pk_block<NB>(j, i), lane, z[i], z[j]);
    }

    // Backward
#pragma unroll
    for (int i = NB - 1; i >= 0; i--)
    {
        z[i] = block_mma<false>(pk + pk_uinv<NB>(i), lane, z[i], zero);

#pragma unroll
        for (int j = 0; j < i; j++)
            z[j] = block_mma<true>(pk + pk_block<NB>(j, i), lane, z[i], z[j]);
    }
}

// Index of element (r, c) of an inverse in PyFR's tiled matrices
template<int TR, int TC, int PADC>
__device__ __forceinline__ int tiled_ix(int r, int c)
{
    return (r / TR)*(PADC / TC)*TR*TC + (c / TC)*TR*TC + (r % TR)*TC + c % TC;
}

// One block of WPB waves per WPB*16 columns of one matrix
template<int N, int NP, long long WSZ, int WPB, int TR, int TC, int PADC,
         typename IX>
__device__ __forceinline__ void lu_solve_mfma(
    IX b, double* __restrict__ inv_v, const IX* __restrict__ inv_vix,
    const short* __restrict__ perm_v, const IX* __restrict__ wk_v,
    const double* __restrict__ ws_v)
{
    constexpr int NB = NP/16, BPM = NB/WPB;
    static_assert(NB % WPB == 0, "waves must split the column blocks");

    const int lane = threadIdx.x % 64;
    const IX x = b / BPM;
    const int cb = int(b % BPM)*WPB + threadIdx.x / 64;

    const IX wk = wk_v[x];
    const double* pk = ws_v + wk*WSZ + NP*NP;
    const short* perm = perm_v + wk*NP;

    f64x4 z[NB];
    solve_columns<NB>(pk, cb, lane, z);

    // Write the columns out in pivoted order
    const int gcol = perm[16*cb + mfma_lrow(lane)];
    if (gcol < N)
    {
        double* inv = inv_v + inv_vix[x];

#pragma unroll
        for (int j = 0; j < NB; j++)
        {
#pragma unroll
            for (int v = 0; v < 4; v++)
            {
                int r = 16*j + mfma_crow(lane, v);
                if (r < N)
                    inv[tiled_ix<TR, TC, PADC>(r, gcol)] = z[j][v];
            }
        }
    }
}

// Pack block row j of the factor of one matrix per block of 256 lanes;
// lanes 0-15 also invert the diagonal block of L, a column each, and lanes
// 16-31 that of U
template<int NP, long long WSZ, typename IX>
__device__ __forceinline__ void lu_pack(IX b, const IX* __restrict__ wk_v,
                                        double* __restrict__ ws_v)
{
    constexpr int NB = NP/16;

    const IX x = b / NB;
    const int j = int(b % NB), t = threadIdx.x;

    double* f = ws_v + wk_v[x]*WSZ;
    double* pk = f + NP*NP;

    // Copy the off diagonal blocks, each lane a run of 4 contiguous entries
    for (int e = 4*t; e < 256*NB; e += 4*256)
    {
        int i = e / 256, l = e % 256 / 4;
        if (i == j)
            continue;

        const double* m = f + NP*(16*j + mfma_lrow(l)) + 16*i + mfma_lk(l);
        f64x4 a = {m[0], m[4], m[8], m[12]};
        *reinterpret_cast<f64x4*>(pk + pk_block<NB>(j, i) + 4*l) = a;
    }

    if (t >= 32)
        return;

    const double* d = f + NP*16*j + 16*j;
    const int c = t % 16;
    double y[16];

    if (t < 16)
    {
        // Lower: y_r = ([r == c] - sum_{k<r} L_rk y_k) / L_rr
#pragma unroll
        for (int r = 0; r < 16; r++)
        {
            double s = (r == c);
#pragma unroll
            for (int k = 0; k < r; k++)
                s -= d[NP*r + k]*y[k];
            y[r] = r < c ? 0.0 : s / d[NP*r + r];
        }

#pragma unroll
        for (int r = 0; r < 16; r++)
            pk[pk_block<NB>(j, j) + pk_ix(r, c)] = y[r];
    }
    else
    {
        // Unit upper: y_r = [r == c] - sum_{k>r} U_rk y_k
#pragma unroll
        for (int r = 15; r >= 0; r--)
        {
            double s = (r == c);
#pragma unroll
            for (int k = r + 1; k < 16; k++)
                s -= d[NP*r + k]*y[k];
            y[r] = r > c ? 0.0 : s;
        }

#pragma unroll
        for (int r = 0; r < 16; r++)
            pk[pk_uinv<NB>(j) + pk_ix(r, c)] = y[r];
    }
}

}
