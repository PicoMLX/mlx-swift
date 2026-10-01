// Copyright © 2026 Apple Inc.

#pragma once

#include <TargetConditionals.h>

// JACCL talks to Thunderbolt 5 RDMA devices through librdma, whose header ships
// with the macOS 26.2 SDK.  The jaccl_*conditional.cpp wrappers build the
// backend when that header is present and its no_jaccl.cpp stub otherwise, as
// MLX's CMake does.
//
// Unlike MLX's CMake this doesn't also require a 26.2 deployment target: JACCL
// loads librdma at runtime, so on an older macOS the backend is built but
// MLXDistributed.isAvailable(.jaccl) reports false.
#if TARGET_OS_OSX && __has_include(<infiniband/verbs.h>)
#define MLX_SWIFT_BUILD_JACCL 1
#else
#define MLX_SWIFT_BUILD_JACCL 0
#endif
