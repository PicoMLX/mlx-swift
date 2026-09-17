// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

/// Multi process tests for the MPI backend.
///
/// These port `mpi_test_distributed.py`: the common distributed tests, which
/// the MPI tests inherit in Python, and the tests of the MPI file itself, which
/// split groups.  MPI is the one backend on Apple platforms that can split a
/// group.
///
/// MLX loads Open MPI's `libmpi.dylib` when a program first uses the backend,
/// so mlx-swift links nothing, and an app that doesn't use MPI doesn't need it.
/// Like MLX's CI, run eight processes with `mpirun`.  System Integrity
/// Protection removes `DYLD_LIBRARY_PATH` when it starts `xcrun`, so name the
/// library with `MLX_MPI_LIBNAME` instead:
///
/// ```
/// mpirun --bind-to none -host localhost:8 -np 8 \
///     -x MLX_MPI_LIBNAME=/opt/homebrew/lib/libmpi.dylib \
///     xcrun xctest -XCTest DistributedMPITests /path/to/MLXTests.xctest
/// ```
///
/// Skipped unless `mpirun` started the process and MLX can load Open MPI.
/// XCTest runs the methods in the same order in every process, and they share
/// the group MLX caches.
class DistributedMPITests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["OMPI_COMM_WORLD_SIZE"] != nil,
            "Launch with mpirun to run the MPI tests.")
        try XCTSkipUnless(
            MLXDistributed.isAvailable(.mpi),
            "MLX can't load Open MPI.  Set MLX_MPI_LIBNAME to the path of libmpi.dylib.")
    }

    // MARK: - mlx_distributed_tests.py

    func testAllGather() throws {
        try withMPI { try allGatherBody(world: $0) }
    }

    func testAllReduce() throws {
        try withMPI { try reductionsBody(world: $0) }
    }

    func testAverageGradients() throws {
        try withMPI { try averageGradientsBody(world: $0) }
    }

    func testClipGradNormSharded() throws {
        try withMPI { try clipGradNormShardedBody(world: $0) }
    }

    func testDonation() throws {
        try withMPI { try donationBody(world: $0) }
    }

    func testShardLinear() throws {
        try withMPI(powerOfTwo: true) { try shardLinearBody(world: $0) }
    }

    func testShardPredicate() throws {
        try withMPI(powerOfTwo: true) { try shardPredicateBody(world: $0) }
    }

    // MARK: - mpi_test_distributed.py

    func testGroups() throws {
        try withMPI { world in
            print("[rank \(world.rank)] joined an MPI group of size \(world.size)")

            XCTAssertEqual(world.size, 8)
            XCTAssertTrue(0 <= world.rank && world.rank < 8)

            let world2 = try MLXDistributed.initialize()
            XCTAssertEqual(world.size, world2.size)
            XCTAssertEqual(world.rank, world2.rank)

            var sub = try world.split(color: world.rank % 2)
            XCTAssertEqual(sub.size, 4)
            XCTAssertEqual(sub.rank, world.rank / 2)

            sub = try world.split(color: world.rank / 2)
            XCTAssertEqual(sub.size, 2)
        }
    }

    /// Python reduces the extra dtypes in the world and in a group split from
    /// it.  ``reductionsBody(world:)`` covers every dtype, so
    /// ``testAllReduce()`` reduces them in the world and this in the group.
    func testAllReduceExtra() throws {
        try withMPI { world in
            let group = try world.split(color: world.rank % 2)
            try reductionsBody(world: group)
        }
    }

    /// Python gathers the extra dtypes in the world and in a group split from
    /// it.  ``allGatherBody(world:)`` covers every dtype, so
    /// ``testAllGather()`` gathers them in the world and this in the group.
    func testAllGatherExtra() throws {
        try withMPI { world in
            let sub = try world.split(color: world.rank % 2)
            try allGatherBody(world: sub)
        }
    }

    func testMixed() throws {
        try withMPI { world in
            // Make the following groups:
            // - world: 0 1 2 3 4 5 6 7
            // - sub_1: 0 1 0 1 0 1 0 1
            // - sub_2: 0 0 1 1 2 2 3 3
            //
            // The corresponding colors to make them are
            // - world: N/A
            // - sub_1: 0 0 1 1 2 2 3 3
            // - sub_2: 0 1 0 1 0 1 0 1

            let sub1 = try world.split(color: world.rank / 2)
            let sub2 = try world.split(color: world.rank % 2)

            let x = MLXArray.ones([1, 8]) * world.rank
            let y = MLXDistributed.allSum(x, group: sub1)
            let z = MLXDistributed.allGather(y, group: sub2)
            let zTarget = MLXArray.arange(8).reshaped(4, 2).sum(axis: -1, keepDims: true)
            try checkedEval(z)

            XCTAssertTrue((z .== zTarget).all().item(Bool.self))
        }
    }

    func testSendRecv() throws {
        try withMPI { world in
            let pairs = try world.split(color: world.rank / 2)
            let neighbor = (pairs.rank + 1) % 2
            var send = pairs.rank == 0

            var x = MLXArray.ones([10])
            for _ in 0 ..< 10 {
                if send {
                    try checkedEval(MLXDistributed.send(2 * x, to: neighbor, group: pairs))
                } else {
                    x = MLXDistributed.recvLike(x, from: neighbor, group: pairs)
                    try checkedEval(x)
                }
                send.toggle()
            }

            XCTAssertTrue((x .== (pairs.rank == 0 ? 1024 : 512)).all().item(Bool.self))

            // Check recv and computation in same eval:
            let y = MLXArray.ones([5, 5]) + MLXArray(Float(2))
            if send {
                x = MLXDistributed.send(2 * x, to: neighbor, group: pairs)
            } else {
                x = MLXDistributed.recvLike(x, from: neighbor, group: pairs)
            }
            try checkedEval(y, x)
        }
    }

    // MARK: - Checks the Python tests don't have

    /// `test_quantized_sharded_linear_construction` is a single process test in
    /// `test_nn.py`; this runs it across the ranks.
    func testQuantizedShardedConstruction() throws {
        try withMPI(powerOfTwo: true) { try quantizedShardedConstructionBody(world: $0) }
    }

    func testModuleSubstitution() throws {
        try withMPI(powerOfTwo: true) { try moduleSubstitutionBody(world: $0) }
    }

    func testShardingEdgeCases() throws {
        try withMPI(powerOfTwo: true) { try shardingEdgeCasesBody(world: $0) }
    }

    /// The forward pass of ``FullyShardedModule``.  MPI doesn't implement
    /// `sumScatter`, so `test_fully_shard_grads` can't run here.
    func testFullyShard() throws {
        try withMPI { try fullyShardBody(world: $0) }
    }

    /// Join the MPI world and run `body` with it.
    ///
    /// MLX errors become test failures that name the rank.
    ///
    /// - Parameters:
    ///   - powerOfTwo: the layer tests split dimensions such as 128 and 1024
    ///     among the ranks, as the Python tests do, so they are skipped for
    ///     other numbers of ranks
    ///   - body: the test body
    private func withMPI(
        powerOfTwo: Bool = false, _ body: (MLXDistributed.Group) throws -> Void
    ) throws {
        let rank = ProcessInfo.processInfo.environment["OMPI_COMM_WORLD_RANK"] ?? "?"
        do {
            let world = try MLXDistributed.initialize(backend: .mpi, strict: true)
            if powerOfTwo {
                try XCTSkipUnless(
                    world.size.nonzeroBitCount == 1,
                    "This test needs a power of two number of ranks.")
            }
            try withError {
                try body(world)
            }
        } catch let skip as XCTSkip {
            throw skip
        } catch {
            XCTFail("rank \(rank) failed: \(error)")
            throw error
        }
    }
}
