// LU factorisation with partial pivoting with fp64 MFMA, one wave per
// matrix, for the solves of solve.hpp.
//
// The workspace of a matrix holds it as NB = NP/16 strips of 16 columns,
// each strip NP rows of 16 entries in a row (strip-major).  lu_build fills
// it with A = I - scale*src, padded with the identity.  lu_factor then
// factorises it left-looking, a strip at a time, as P A = L U with L lower
// and U unit upper (the convention of PyFR's template), overwriting each
// strip with its 16 x 16 blocks packed for the MFMA (see solve.hpp): the
// blocks of U above the diagonal, the inverse of the diagonal block of L,
// and the blocks of L below it.  The inverses of the diagonal blocks of U
// go after the strips.
//
// For strip k the wave
//   1. loads the strip in the current row order into accumulator tiles,
//   2. applies the factored strips: Z_i = L_ii^-1 Z_i for the block rows
//      i < k, each followed by Z_j -= L_ji Z_i for j > i,
//   3. stores the blocks i < k, which are now U_ik,
//   4. factorises the remaining NP - 16k rows of the strip with partial
//      pivoting, a row per lane and column, as the template does,
//   5. swaps the pivoted rows in the factored strips and the row order,
//   6. stores the factored blocks and the inverses of the diagonal blocks.
// The strips to the right are untouched until loaded in the row order of
// the time, so they take the row interchanges as they are loaded.
#pragma once

#include "luinv/hip_v2/mfma.hpp"
#include "luinv/hip_v2/solve.hpp"

namespace luinv
{

// Workspace offset of entry (r, c) of a strip-major matrix
template<int NP>
__device__ __forceinline__ int sm_ix(int r, int c)
{
    return NP*16*(c / 16) + 16*r + c % 16;
}

// Fill the strip-major workspaces with A = I - scale*src for one row of
// the 64 matrices of an AoSoA group of src per block of 256 lanes, staging
// 16 columns at a time through LDS so both the loads and stores coalesce
template<int N, int NP, long long WSZ, typename IX>
__device__ __forceinline__ void lu_build(
    IX b, IX nx, double scale, const double* __restrict__ src_v, IX ldsrc,
    const IX* __restrict__ wk_v, double* __restrict__ ws_v)
{
    constexpr int SOA = 64;
    __shared__ double st[16][SOA + 1];

    const IX g = b / NP;
    const int r = int(b % NP), t = threadIdx.x;

    for (int c0 = 0; c0 < NP; c0 += 16)
    {
        // Load entry (r, c0 + c) of the matrices of the group
        for (int i = t; i < 16*SOA; i += 256)
        {
            int c = c0 + i / SOA, xm = i % SOA;
            double v = 0;
            if (r < N && c < N && SOA*g + xm < nx)
                v = src_v[ldsrc*r + (g*N + c)*SOA + xm];
            st[i / SOA][xm] = (r == c) - scale*v;
        }
        __syncthreads();

        // Store them a run of 16 columns per matrix
        for (int i = t; i < 16*SOA; i += 256)
        {
            int xm = i / 16, c = i % 16;
            IX x = SOA*g + xm;
            if (x < nx)
                ws_v[wk_v[x]*WSZ + sm_ix<NP>(r, c0 + c)] = st[c][xm];
        }
        __syncthreads();
    }
}

// Pivot search key of entry v of row r: the larger magnitude wins, then
// the lower row; the low 12 bits of the magnitude make way for the row
__device__ __forceinline__ unsigned long long pivot_key(double v, int r)
{
    unsigned long long m = __double_as_longlong(fabs(v));
    return (m & ~0xfffull) | (0xfffu - r);
}

__device__ __forceinline__ int pivot_row(unsigned long long key)
{
    return 0xfff - int(key & 0xfff);
}

// Lane value of v moved by the DPP control Ctrl, rows outside RowMask
// keeping their own
template<int Ctrl, int RowMask = 0xf>
__device__ __forceinline__ unsigned long long dpp_u64(unsigned long long v)
{
    int lo = int(v), hi = int(v >> 32);
    lo = __builtin_amdgcn_update_dpp(lo, lo, Ctrl, RowMask, 0xf, false);
    hi = __builtin_amdgcn_update_dpp(hi, hi, Ctrl, RowMask, 0xf, false);
    return (unsigned long long)(unsigned) lo | (unsigned long long) hi << 32;
}

// Largest key of the wave by DPP moves, as in a reduction to lane 63
__device__ __forceinline__ unsigned long long wave_max(unsigned long long v)
{
    auto mx = [](unsigned long long a, unsigned long long b)
    {
        return a > b ? a : b;
    };

    v = mx(v, dpp_u64<0xb1>(v));        // quad_perm [1, 0, 3, 2]
    v = mx(v, dpp_u64<0x4e>(v));        // quad_perm [2, 3, 0, 1]
    v = mx(v, dpp_u64<0x141>(v));       // row_half_mirror
    v = mx(v, dpp_u64<0x140>(v));       // row_mirror
    v = mx(v, dpp_u64<0x142, 0xa>(v));  // row_bcast:15
    v = mx(v, dpp_u64<0x143, 0xc>(v));  // row_bcast:31

    int lo = __builtin_amdgcn_readlane(int(v), 63);
    int hi = __builtin_amdgcn_readlane(int(v >> 32), 63);
    return (unsigned long long)(unsigned) lo | (unsigned long long) hi << 32;
}

// Broadcast the value of v in lane l, uniform across the wave
__device__ __forceinline__ double read_lane(double v, int l)
{
    int lo = __builtin_amdgcn_readlane(__double2loint(v), l);
    int hi = __builtin_amdgcn_readlane(__double2hiint(v), l);
    return __hiloint2double(hi, lo);
}

// Invert a 16 x 16 diagonal block d (row major in LDS) into packed blocks,
// lanes 0-15 of the first wave given taking the columns of L^-1, with L on
// and below the diagonal of d, and those of the second the columns of
// U^-1, with U unit upper above it.  idg holds the reciprocals of the
// diagonal of d.
__device__ __forceinline__ void invert_diag(const double* d,
                                            const double* idg, int lane,
                                            bool lower, double* linv,
                                            double* uinv)
{
    const int c = lane % 16;

    // Column c of the inverse by substitution a column of d at a time, each
    // solved entry updating all the rows after it at once
    double y[16];
#pragma unroll
    for (int r = 0; r < 16; r++)
        y[r] = (r == c);

    if (lower && lane < 16)
    {
#pragma unroll
        for (int k = 0; k < 16; k++)
        {
            y[k] *= idg[k];
#pragma unroll
            for (int r = k + 1; r < 16; r++)
                y[r] -= d[16*r + k]*y[k];
        }

#pragma unroll
        for (int r = 0; r < 16; r++)
            linv[pk_ix(r, c)] = y[r];
    }
    else if (!lower && lane < 16)
    {
#pragma unroll
        for (int k = 15; k >= 0; k--)
        {
#pragma unroll
            for (int r = 0; r < k; r++)
                y[r] -= d[16*r + k]*y[k];
        }

#pragma unroll
        for (int r = 0; r < 16; r++)
            uinv[pk_ix(r, c)] = y[r];
    }
}

// Make the global and LDS stores of the block visible to all of its lanes
__device__ __forceinline__ void block_sync()
{
    __threadfence_block();
    __syncthreads();
}

// Cycles wave 0 spends in each phase of the factorisation, summed into
// ticks when LUINV_PHASES is defined
#if defined(LUINV_PHASES)
#define LUINV_TICK(k) do { if (threadIdx.x == 0) { long long _t = clock64(); \
    ph[k] += _t - t0; t0 = _t; } } while (0)
#else
#define LUINV_TICK(k) do {} while (0)
#endif

// Phases of lu_factor, in the order of LUINV_TICK
#define LUINV_FACTOR_PHASES "load", "update", "store", "", "", "", "", \
    "panel", "invert"

// Factor rows r0 = 16k to NP - 1 of the panel of strip k, which the block
// left row major in the strip, a row per lane and pass.  For each column the
// waves reduce the pivot search by DPP moves and post their best candidate
// row in LDS, so after one barrier every lane picks the pivot row and reads
// it from there.  Then apply the row interchanges to the factored strips and
// the row order rord, store the blocks of L below the diagonal packed and
// stage the diagonal block in dg.  Kept out of line, as its registers are
// separate from those of the updates.
template<int NP, int W>
__device__ __noinline__ void factor_panel(
    double* a, int k, short* rord, int* ipiv, double (*cand)[W][16],
    double (*grow)[16], unsigned long long (*ckey)[W], double* dg)
{
    constexpr int NB = NP/16, NL = 64*W, NQ = (NP + NL - 1)/NL;

    const int tid = threadIdx.x, lane = tid % 64, w = tid / 64;
    const int r0 = 16*k;
    double* s = a + 256*NB*k;

    double pv[NQ][16];
#pragma unroll
    for (int q = 0; q < NQ; q++)
    {
        int r = r0 + tid + NL*q;
        if (r < NP)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                pv[q][c] = s[16*r + c];
        }
    }

#pragma unroll
    for (int j = 0; j < 16; j++)
    {
        const int gk = r0 + j, b = j % 2;

        // Find the pivot: largest magnitude in column j on or below gk, the
        // first such row on a near tie
        unsigned long long key = 0;
        int mq = -1;
#pragma unroll
        for (int q = 0; q < NQ; q++)
        {
            int r = r0 + tid + NL*q;
            unsigned long long kq = pivot_key(pv[q][j], r);
            if (r >= gk && r < NP && kq > key)
            {
                key = kq;
                mq = q;
            }
        }
        const unsigned long long mine = key;
        key = wave_max(key);

        // Post the best row of the wave as it would be as the pivot row,
        // scaled right of the diagonal, and the diagonal row
#pragma unroll
        for (int q = 0; q < NQ; q++)
        {
            if (mq == q && mine == key)
            {
                const double ipv = 1.0 / pv[q][j];
#pragma unroll
                for (int c = 0; c < 16; c++)
                    cand[b][w][c] = c > j ? pv[q][c]*ipv : pv[q][c];
            }
        }
        if (tid == j)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                grow[b][c] = pv[0][c];
        }
        if (lane == 0)
            ckey[b][w] = key;
        __syncthreads();

        // Pick the best of the waves
        int wb = 0;
        key = ckey[b][0];
#pragma unroll
        for (int v = 1; v < W; v++)
        {
            if (ckey[b][v] > key)
            {
                key = ckey[b][v];
                wb = v;
            }
        }
        const int mi = pivot_row(key);
        if (tid == 0)
            ipiv[j] = mi;

        // Swap in the pivot row and eliminate the rows below
        const double* pr = cand[b][wb];
#pragma unroll
        for (int q = 0; q < NQ; q++)
        {
            int r = r0 + tid + NL*q;
            if (r == gk)
            {
#pragma unroll
                for (int c = 0; c < 16; c++)
                    pv[q][c] = pr[c];
            }
            else if (r > gk && r < NP)
            {
                if (r == mi)
                {
#pragma unroll
                    for (int c = 0; c < 16; c++)
                        pv[q][c] = grow[b][c];
                }
                double lv = pv[q][j];
#pragma unroll
                for (int c = j + 1; c < 16; c++)
                    pv[q][c] -= lv*pr[c];
            }
        }
    }
    __syncthreads();

    // Apply the interchanges to the factored strips and the row order
#pragma unroll 1
    for (int j = 0; j < 16; j++)
    {
        const int gk = r0 + j, mi = ipiv[j];
        if (mi == gk)
            continue;

        for (int e = tid; e < 16*k; e += NL)
        {
            double* sb = a + 256*NB*(e / 16);
            int c = e % 16;
            int ig = 256*(gk / 16) + pk_ix(gk % 16, c);
            int im = 256*(mi / 16) + pk_ix(mi % 16, c);
            double t = sb[ig];
            sb[ig] = sb[im];
            sb[im] = t;
        }
        if (tid == 0)
        {
            short t = rord[gk];
            rord[gk] = rord[mi];
            rord[mi] = t;
        }
    }

    // Store the factored blocks below the diagonal, and stage the diagonal
    // block
#pragma unroll
    for (int q = 0; q < NQ; q++)
    {
        int r = r0 + tid + NL*q;
        if (r < r0 + 16)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                dg[16*(r - r0) + c] = pv[q][c];
        }
        else if (r < NP)
        {
#pragma unroll
            for (int c = 0; c < 16; c++)
                s[256*(r / 16) + pk_ix(r % 16, c)] = pv[q][c];
        }
    }
}

// Factorise one matrix per block of W waves.
//
// The waves deal the block rows of a strip between them: wave w holds block
// rows jb + w, jb + w + W, ... as tiles 0, 1, ..., with jb advancing by W
// as the leading tiles are solved.  For each step i of the updates the
// owner of block row i has published the solved Z_i through LDS; every wave
// then subtracts L_ji Z_i from its block rows below, the owner of block row
// i + 1 first, solving and publishing it for the next step, so each step
// takes one barrier.
//
// The panel is spread over all the lanes too; see factor_panel.
template<int NP, long long WSZ, int W, typename IX>
__device__ __forceinline__ void lu_factor(IX x, const IX* __restrict__ wk_v,
                                          double* __restrict__ ws_v,
                                          short* __restrict__ perm_v,
                                          unsigned long long* ticks)
{
#if defined(LUINV_PHASES)
    long long ph[9] = {}, t0 = clock64();
#endif
    constexpr int NB = NP/16, NT = (NB + W - 1)/W, NL = 64*W;
    constexpr int NQ = (NP + NL - 1)/NL;

    __shared__ short rord[NP];
    __shared__ double dg[256], idg[16];
    __shared__ f64x4 zb[2][64];
    __shared__ double cand[2][W][16], grow[2][16];
    __shared__ unsigned long long ckey[2][W];
    __shared__ int ipiv[16];

    const int tid = threadIdx.x, lane = tid % 64, w = tid / 64;
    const IX wk = wk_v[x];
    double* a = ws_v + wk*WSZ;
    double* uinv = a + pk_uinv<NB>(0);
    const f64x4 zero = {0, 0, 0, 0};

    // Start from the rows in their given order
    for (int r = tid; r < NP; r += NL)
        rord[r] = r;
    __syncthreads();

    for (int k = 0; k < NB; k++)
    {
        double* s = a + 256*NB*k;
        const int r0 = 16*k;

        // Load the block rows of this wave in the current row order
        f64x4 z[NT];
#pragma unroll
        for (int t = 0; t < NT; t++)
        {
            int j = w + W*t;
            if (j < NB)
            {
#pragma unroll
                for (int v = 0; v < 4; v++)
                {
                    int r = rord[16*j + mfma_crow(lane, v)];
                    z[t][v] = s[16*r + mfma_lrow(lane)];
                }
            }
        }

        LUINV_TICK(0);

        // The operands of step i of the updates for the tiles, numbered
        // from block row jb, and of the solve of block row i + 1
        f64x4 pa[NT], pd;
        auto fetch = [&](int i, int jb)
        {
#pragma unroll
            for (int t = 0; t < NT; t++)
            {
                int j = jb + w + W*t;
                if (j > i && j < NB)
                    pa[t] = load_a(a + pk_block<NB>(j, i), lane);
            }
            if (i + 1 < k && w == (i + 1) % W)
                pd = load_a(a + pk_block<NB>(i + 1, i + 1), lane);
        };

        // Solve and publish block row 0, storing it as U_0k
        if (k > 0)
        {
            fetch(0, 0);
            if (w == 0)
            {
                z[0] = block_mma<false>(a + pk_block<NB>(0, 0), lane, z[0],
                                        zero);
                zb[0][lane] = z[0];
#pragma unroll
                for (int v = 0; v < 4; v++)
                    s[pk_ix(mfma_crow(lane, v), mfma_lrow(lane))] = z[0][v];
            }
        }
        __syncthreads();

        // Steps ib to ib + W - 1 of the updates, those before k, each
        // fetching the operands of the next
        auto steps = [&](int ib)
        {
#pragma unroll
            for (int u = 0; u < W; u++)
            {
                const int i = ib + u;
                if (i >= k)
                    break;

                const f64x4 zi = zb[u % 2][lane];
                const bool next = i + 1 < k;

                f64x4 ca[NT], cd = pd;
#pragma unroll
                for (int t = 0; t < NT; t++)
                    ca[t] = pa[t];
                if (next)
                    fetch(i + 1, u == W - 1 ? ib + W : ib);

                // Update, solve, publish and store the next block row
                const int on = (u + 1) % W, tn = (u + 1) / W;
                if (next && w == on)
                {
                    f64x4 zn = mma_a<true>(ca[tn], zi, z[tn]);
                    zn = mma_a<false>(cd, zn, zero);
                    z[tn] = zn;
                    zb[(u + 1) % 2][lane] = zn;
#pragma unroll
                    for (int v = 0; v < 4; v++)
                        s[256*(i + 1) + pk_ix(mfma_crow(lane, v),
                                              mfma_lrow(lane))] = zn[v];
                }

                // Update the other block rows below
#pragma unroll
                for (int t = 0; t < NT; t++)
                {
                    int j = ib + w + W*t;
                    if (j > i + next && j < NB)
                        z[t] = mma_a<true>(ca[t], zi, z[t]);
                }
                __syncthreads();
            }
        };

        // Whole groups of steps, after which the leading tiles are solved
        int jb = 0;
        for (; jb + W <= k; jb += W)
        {
            steps(jb);

#pragma unroll
            for (int t = 0; t < NT - 1; t++)
                z[t] = z[t + 1];
        }
        if (jb < k)
            steps(jb);

        LUINV_TICK(1);

        // Store the rest of the strip row major to pick it up a row a lane
#pragma unroll
        for (int t = 0; t < NT; t++)
        {
            int j = jb + w + W*t;
            if (j >= k && j < NB)
            {
#pragma unroll
                for (int v = 0; v < 4; v++)
                    s[256*j + 16*mfma_crow(lane, v) + mfma_lrow(lane)] =
                        z[t][v];
            }
        }
        block_sync();

        LUINV_TICK(2);

        factor_panel<NP, W>(a, k, rord, ipiv, cand, grow, ckey, dg);
        block_sync();

        LUINV_TICK(7);

        if (tid < 16)
            idg[tid] = 1.0 / dg[17*tid];
        __syncthreads();

        if (w < 2)
            invert_diag(dg, idg, lane, w == 0, s + 256*k, uinv + 256*k);
        block_sync();

        LUINV_TICK(8);
    }

    // Leave the row order for the solves
    short* perm = perm_v + wk*NP;
    for (int r = tid; r < NP; r += NL)
        perm[r] = rord[r];

#if defined(LUINV_PHASES)
    if (tid == 0)
        for (int i = 0; i < 9; i++)
            atomicAdd(ticks + i, (unsigned long long) ph[i]);
#endif
}

}
