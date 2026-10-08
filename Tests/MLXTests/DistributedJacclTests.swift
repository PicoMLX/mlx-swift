// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// Multi process tests for the JACCL backend.
///
/// Python has no JACCL-specific test file: its common distributed tests run on
/// whichever backend `mlx.launch` started.  Each test method here runs one of
/// those ported tests, named after the Python test, plus the ring tests JACCL
/// passes too and `sumScatter`, which JACCL implements and the ring backend
/// doesn't.
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
/// Skipped unless that environment is present.  XCTest runs the methods in the
/// same order in every process, and they share the group MLX caches.
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

    // MARK: - mlx_distributed_tests.py

    func testAllGather() throws {
        try withJaccl { try allGatherBody(world: $0) }
    }

    func testAllReduce() throws {
        try withJaccl { try reductionsBody(world: $0) }
    }

    func testAverageGradients() throws {
        try withJaccl { try averageGradientsBody(world: $0) }
    }

    func testClipGradNormSharded() throws {
        try withJaccl { try clipGradNormShardedBody(world: $0) }
    }

    func testDonation() throws {
        try withJaccl { try donationBody(world: $0) }
    }

    func testShardLinear() throws {
        try withJaccl(powerOfTwo: true) { try shardLinearBody(world: $0) }
    }

    func testShardPredicate() throws {
        try withJaccl(powerOfTwo: true) { try shardPredicateBody(world: $0) }
    }

    // MARK: - ring_test_distributed.py, which hold for JACCL too

    /// JACCL can't split a group either.
    func testGroups() throws {
        try withJaccl { group in
            print("[rank \(group.rank)] joined a JACCL group of size \(group.size)")
            try groupsBody(world: group)
        }
    }

    func testSendRecv() throws {
        try withJaccl { try sendRecvBody(world: $0) }
    }

    func testAllGatherVJP() throws {
        try withJaccl { try allGatherVJPBody(world: $0) }
    }

    // MARK: - nccl_test_distributed.py

    /// Skipped for now.  `ReduceScatter::eval_cpu` in MLX asserts
    /// `inputs.size() == 0` (`mlx/backend/cpu/distributed.cpp`), although the
    /// primitive always has one input, and mlx-swift doesn't define `NDEBUG`,
    /// so evaluating `sumScatter` in a group of more than one aborts the
    /// process.  Remove the skip once MLX fixes the assert.  The same body runs
    /// in a group of one in ``DistributedTests``.
    func testSumScatter() throws {
        try XCTSkipIf(true, "MLX's ReduceScatter::eval_cpu asserts on its input.")
        try withJaccl { try sumScatterBody(world: $0) }
    }

    // MARK: - Checks the Python tests don't have

    /// `test_quantized_sharded_linear_construction` is a single process test in
    /// `test_nn.py`; this runs it across the ranks.
    func testQuantizedShardedConstruction() throws {
        try withJaccl(powerOfTwo: true) { try quantizedShardedConstructionBody(world: $0) }
    }

    func testModuleSubstitution() throws {
        try withJaccl(powerOfTwo: true) { try moduleSubstitutionBody(world: $0) }
    }

    func testShardingEdgeCases() throws {
        try withJaccl(powerOfTwo: true) { try shardingEdgeCasesBody(world: $0) }
    }

    /// Join the JACCL group and run `body` with it.
    ///
    /// MLX errors become test failures that name the rank.
    ///
    /// - Parameters:
    ///   - powerOfTwo: the layer tests split dimensions such as 128 and 1024
    ///     among the ranks, as the Python tests do, so they are skipped for
    ///     other numbers of ranks
    ///   - body: the test body
    private func withJaccl(
        powerOfTwo: Bool = false, _ body: (MLXDistributed.Group) throws -> Void
    ) throws {
        let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
        do {
            let group = try MLXDistributed.initialize(backend: .jaccl, strict: true)
            if powerOfTwo {
                try XCTSkipUnless(
                    group.size.nonzeroBitCount == 1,
                    "This test needs a power of two number of ranks.")
            }
            try withError {
                try body(group)
            }
        } catch let skip as XCTSkip {
            throw skip
        } catch {
            XCTFail("rank \(rank) failed: \(error)")
            throw error
        }
    }
}
