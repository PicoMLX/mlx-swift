// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

// Ports of the collective tests from the Python distributed tests
// (`python/tests/mlx_distributed_tests.py` and `ring_test_distributed.py`).
// Python runs its common tests on whichever backend its launcher started, so
// each body here takes the group: ``DistributedRingTests`` and
// ``DistributedJacclTests`` run the same ones.

/// The dtypes of the Python tests, including the ones the ring tests add.
let distributedTestDTypes: [DType] = [
    .int8, .uint8, .int16, .uint16, .int32, .uint32, .float32, .float16, .bfloat16,
    .complex64,
]

/// Sizes from the Python tests.  The large ones exercise chunked transfers,
/// which a handful of elements never reach.
let distributedTestShapes = [[7], [10], [1024], [1024, 1024]]

/// Port of `test_groups` from the ring tests.
///
/// Neither the ring nor the JACCL backend can split a group.
func groupsBody(world: MLXDistributed.Group) throws {
    XCTAssertTrue(world.rank >= 0 && world.rank < world.size)

    // initializing again yields the same group
    let again = try MLXDistributed.initialize()
    XCTAssertEqual(again.size, world.size)
    XCTAssertEqual(again.rank, world.rank)

    XCTAssertThrowsError(try world.split(color: world.rank % 2))
}

/// Port of `test_all_reduce` and `test_all_reduce_extra`.
///
/// Every rank builds the same array and contributes its own row, so the
/// reduction over the rows is the expected result in every process.
func reductionsBody(world: MLXDistributed.Group) throws {
    let tolerances: [DType: Float] = [
        .float32: 1e-6, .float16: 5e-3, .bfloat16: 1e-1, .complex64: 1e-6,
    ]
    let key = MLXRandom.key(0)
    var combinations = 0
    defer {
        if world.rank == 0 {
            print("reductions: \(combinations) dtype and shape combinations")
        }
    }

    for dtype in distributedTestDTypes {
        for shape in distributedTestShapes {
            let rtol = tolerances[dtype] ?? 0
            let x = (MLXRandom.uniform(0 ..< 1, [world.size] + shape, key: key) * 10)
                .asType(dtype)
            let name = "\(dtype) \(shape)"
            combinations += 1

            let sum = MLXDistributed.allSum(x[world.rank], group: world)
            let expected = x.sum(axis: 0)
            var error = abs(sum - expected)
            if rtol > 0 {
                error = error / abs(expected)
            }
            try checkedEval(error)
            XCTAssertLessThanOrEqual(
                error.max().asType(.float32).item(Float.self), rtol, "allSum \(name)")

            let maximum = MLXDistributed.allMax(x[world.rank], group: world)
            try checkedEval(maximum)
            XCTAssertTrue(
                (maximum .== x.max(axis: 0)).all().item(Bool.self), "allMax \(name)")

            let minimum = MLXDistributed.allMin(x[world.rank], group: world)
            try checkedEval(minimum)
            XCTAssertTrue(
                (minimum .== x.min(axis: 0)).all().item(Bool.self), "allMin \(name)")
        }
    }
}

/// Port of `test_all_gather` and `test_all_gather_extra`.
func allGatherBody(world: MLXDistributed.Group) throws {
    for dtype in distributedTestDTypes {
        let x = MLXArray.ones([2, 2, 4]).asType(dtype)
        let gathered = MLXDistributed.allGather(x, group: world)
        try checkedEval(gathered)

        XCTAssertEqual(gathered.shape, [world.size * 2, 2, 4], "allGather \(dtype)")
        XCTAssertTrue(
            (gathered .== MLXArray(1).asType(dtype)).all().item(Bool.self),
            "allGather \(dtype)")
    }

    // the shards are concatenated in rank order rather than reduced
    let base = MLXArray([1, 2, 3]).asType(.float32)
    let gathered = MLXDistributed.allGather(base * Float(world.rank + 1), group: world)
    try checkedEval(gathered)
    assertEqual(gathered, concatenated((0 ..< world.size).map { base * Float($0 + 1) }))
}

/// Port of `test_send_recv` from the ring tests.
///
/// Every rank sends to its right neighbor and receives from its left one.
/// Even ranks send first and odd ranks receive first, so that neighbors do
/// not wait on each other.
func sendRecvBody(world: MLXDistributed.Group) throws {
    let rank = world.rank
    let size = world.size
    let right = (rank + 1) % size
    let left = (rank + size - 1) % size
    let key = MLXRandom.key(0)
    var transfers = 0
    defer {
        if rank == 0 {
            print("send/recv: \(transfers) dtype and shape combinations")
        }
    }

    for dtype in distributedTestDTypes {
        for shape in distributedTestShapes {
            let x = (MLXRandom.uniform(0 ..< 1, [size] + shape, key: key) * 10)
                .asType(dtype)
            let name = "\(dtype) \(shape)"
            transfers += 1

            let sent: MLXArray
            let received: MLXArray
            if rank % 2 == 0 {
                sent = MLXDistributed.send(x[rank], to: right, group: world)
                received = MLXDistributed.recvLike(sent, from: left, group: world)
                try checkedEval(sent, received)
            } else {
                received = MLXDistributed.recvLike(x[rank], from: left, group: world)
                sent = MLXDistributed.send(x[rank], to: right, group: world)
                try checkedEval(received, sent)
            }

            XCTAssertTrue((sent .== x[rank]).all().item(Bool.self), "send \(name)")
            XCTAssertTrue((received .== x[left]).all().item(Bool.self), "recv \(name)")
        }
    }
}

/// Port of `test_all_gather_vjp` from the ring tests.
///
/// The gathered array starts with rank 0's contribution, so only rank 0
/// sees a gradient.
func allGatherVJPBody(world: MLXDistributed.Group) throws {
    let gradient = grad { x in
        MLXDistributed.allGather(x, group: world)[0]
    }(MLXArray(1.0))
    try checkedEval(gradient)

    XCTAssertEqual(gradient.item(Float.self), world.rank == 0 ? 1.0 : 0.0)
}

/// Port of `test_sum_scatter` from the NCCL tests.
///
/// JACCL implements `sumScatter` too; the ring backend doesn't.  Every rank
/// draws its own array and has to receive its own chunk of the sum.
func sumScatterBody(world: MLXDistributed.Group) throws {
    let tolerances: [(DType, Float)] = [(.float32, 1e-6), (.float16, 5e-3), (.bfloat16, 1e-1)]
    let shapes = [[8], [64], [1024], [1024, 1024]]
    let key = MLXRandom.key(UInt64(world.rank))

    for (dtype, rtol) in tolerances {
        for shape in shapes {
            let x = (MLXRandom.uniform(0 ..< 1, shape, key: key) * 10).asType(dtype)
            let scattered = MLXDistributed.sumScatter(x, group: world)
            let summed = MLXDistributed.allSum(x, group: world)

            let chunk = shape[0] / world.size
            let start = world.rank * chunk
            let expected = summed[start ..< start + chunk]

            var error = abs(scattered - expected)
            if rtol > 0 {
                error = error / abs(expected)
            }
            try checkedEval(error)
            XCTAssertLessThanOrEqual(
                error.max().asType(.float32).item(Float.self), rtol,
                "sumScatter \(dtype) \(shape)")
        }
    }
}

/// Multi process tests for the ring backend.
///
/// ``DistributedHarness`` runs the bodies above in several processes.  They are
/// skipped unless `MLX_TEST_DISTRIBUTED=1` is set:
///
/// ```
/// MLX_TEST_DISTRIBUTED=1 xcrun xctest -XCTest DistributedRingTests \
///     .../MLXTests.xctest
/// ```
///
/// There is deliberately a single test method: MLX caches the group per
/// process, and a single method also guarantees every rank issues the same
/// operations in the same order, which the ring backend requires.
class DistributedRingTests: XCTestCase {

    static let testName = "DistributedRingTests/testRingCollectives"

    /// Three ranks, so that the left and right neighbors of a rank are
    /// different processes.
    static let rankCount = 3

    func testRingCollectives() throws {
        try DistributedHarness.run(ranks: Self.rankCount, testName: Self.testName) { group in
            try groupsBody(world: group)
            try reductionsBody(world: group)
            try allGatherBody(world: group)
            try sendRecvBody(world: group)
            try allGatherVJPBody(world: group)
        }
    }
}
