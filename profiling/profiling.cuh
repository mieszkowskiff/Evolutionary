#ifndef PROFILING_CUH
#define PROFILING_CUH

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#ifdef ENABLE_NVTX
#define NVTX_PUSH_ENABLED(enabled, name) \
    do { if (enabled) nvtxRangePushA(name); } while (0)

#define NVTX_POP_ENABLED(enabled) \
    do { if (enabled) nvtxRangePop(); } while (0)
#else
#define NVTX_PUSH_ENABLED(enabled, name) do {} while (0)
#define NVTX_POP_ENABLED(enabled) do {} while (0)
#endif

#endif