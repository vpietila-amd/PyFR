// Types and block numbering of the hand written kernels, matching those of
// the PyFR kernels: LUINV_IXTYPE is the index type of the memory model and
// LUINV_NDIES the number of dies (XCDs) the blocks are dealt across.
#pragma once

#if defined(__HIPCC_RTC__)
typedef short int16_t;
typedef long long int64_t;
#endif

typedef LUINV_IXTYPE ixdtype_t;

// Give each die a contiguous range of blocks, as PYFR_BLOCK_ID does, so the
// blocks of a matrix run on one die and share its L2
#define LUINV_BLOCK_ID ((blockIdx.x % LUINV_NDIES)*(gridDim.x / LUINV_NDIES) \
    + min(blockIdx.x % LUINV_NDIES, gridDim.x % LUINV_NDIES)                  \
    + blockIdx.x / LUINV_NDIES)
