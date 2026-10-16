// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

class GatherMMTests: XCTestCase {

    /// One row per batch entry with sorted right-hand indices takes MLX's
    /// grouped gather path on the GPU, which needs the precompiled
    /// `gather_mm_offsets` kernel (MLX v0.32.3).
    func testGatherMMWithSortedRightIndicesMatchesMatmul() {
        MLXRandom.seed(0)
        let a = MLXRandom.normal([6, 1, 32])
        let b = MLXRandom.normal([3, 32, 16])
        let indices = MLXArray([0, 0, 1, 1, 2, 2] as [UInt32])

        let gathered = gatherMM(a, b, rhsIndices: indices, sortedIndices: true, stream: .gpu)
        let expected = matmul(a, b.take(indices, axis: 0))

        XCTAssertEqual(gathered.shape, [6, 1, 16])
        // Either side may run at TF32 precision on chips with neural
        // accelerators; a wrong gather would be off by far more.
        assertEqual(gathered, expected, rtol: 1e-2, atol: 1e-2)
    }
}
