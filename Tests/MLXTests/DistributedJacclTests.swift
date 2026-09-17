// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// Multi process tests for the JACCL backend.
///
/// Python has no JACCL-specific test file: its common distributed tests run on
/// whichever backend `mlx.launch` started.  These run the same ported bodies as
/// ``DistributedRingTests`` and ``DistributedNNRingTests``, plus `sumScatter`,
/// which JACCL implements and the ring backend doesn't.
///
/// The ranks run on different Macs connected over Thunderbolt 5 with RDMA, so
/// the tests can't start them the way ``DistributedHarness`` does.  Launch one
/// process per Mac with `mlx.launch`, which sets `MLX_RANK`,
/// `MLX_JACCL_COORDINATOR` and `MLX_IBV_DEVICES`.  The test bundle has to be at
/// the same path on every Mac:
///
/// ```
/// mlx.launch --backend jaccl --hostfile hosts.json -- \
///     xcrun xctest -XCTest DistributedJacclTests /path/to/MLXTests.xctest
/// ```
///
/// Skipped unless that environment is present.  The test methods run in the
/// same order in every process and share the group MLX caches.
class DistributedJacclTests: XCTestCase {

    override func setUpWithError() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(
            environment["MLX_RANK"] != nil && environment["MLX_JACCL_COORDINATOR"] != nil
                && environment["MLX_IBV_DEVICES"] != nil,
            "Launch with mlx.launch --backend jaccl to run the JACCL tests.")
        try XCTSkipUnless(
            MLXDistributed.isAvailable(.jaccl),
            "JACCL needs macOS 26.2 or later, built with the macOS 26.2 SDK or later.")
    }

    func testCollectives() throws {
        try withJaccl { group in
            print("[rank \(group.rank)] joined a JACCL group of size \(group.size)")

            try groupsBody(world: group)
            try reductionsBody(world: group)
            try allGatherBody(world: group)
            try sendRecvBody(world: group)
            try allGatherVJPBody(world: group)
        }
    }

    func testShardedLayers() throws {
        // the ported layer tests split dimensions of 128 and 1024 among the
        // ranks, as the Python tests do
        let size = try MLXDistributed.initialize(backend: .jaccl, strict: true).size
        try XCTSkipUnless(
            size.nonzeroBitCount == 1, "The layer tests need a power of two number of ranks.")

        try withJaccl { group in
            try shardLinearBody(world: group)
            try shardPredicateBody(world: group)
            try moduleSubstitutionBody(world: group)
            try donationBody(world: group)
            try quantizedShardedConstructionBody(world: group)
            try shardingEdgeCasesBody(world: group)
            try averageGradientsBody(world: group)
            try clipGradNormShardedBody(world: group)
        }
    }

    /// Port of `test_sum_scatter` from the NCCL tests.
    ///
    /// Skipped for now.  `ReduceScatter::eval_cpu` in MLX asserts
    /// `inputs.size() == 0` (`mlx/backend/cpu/distributed.cpp`), although the
    /// primitive always has one input, and mlx-swift doesn't define `NDEBUG`,
    /// so evaluating `sumScatter` in a group of more than one aborts the
    /// process.  Remove the skip once MLX fixes the assert.  The same body runs
    /// in a group of one in ``DistributedTests``.
    func testSumScatter() throws {
        try XCTSkipIf(true, "MLX's ReduceScatter::eval_cpu asserts on its input.")

        try withJaccl { group in
            try sumScatterBody(world: group)
        }
    }

    /// Join the JACCL group and run `body` with it.
    ///
    /// MLX errors become test failures that name the rank.
    private func withJaccl(_ body: (MLXDistributed.Group) throws -> Void) throws {
        do {
            let group = try MLXDistributed.initialize(backend: .jaccl, strict: true)
            try withError {
                try body(group)
            }
        } catch {
            let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
            XCTFail("rank \(rank) failed: \(error)")
            throw error
        }
    }
}
