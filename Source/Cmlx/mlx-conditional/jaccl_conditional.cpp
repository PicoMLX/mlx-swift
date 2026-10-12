// Copyright © 2026 Apple Inc.

#include "jaccl_conditional.h"

// the JACCL backend or its stub -- see jaccl_conditional.h
#if MLX_SWIFT_BUILD_JACCL
#include "../mlx/mlx/distributed/jaccl/jaccl.cpp"
#else
#include "../mlx/mlx/distributed/jaccl/no_jaccl.cpp"
#endif
