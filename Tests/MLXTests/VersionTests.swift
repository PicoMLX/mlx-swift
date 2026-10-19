// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import XCTest

class VersionTests: XCTestCase {

    /// MLX's CMake reads its version from `mlx/version.h`. mlx-swift passes it to the
    /// compiler itself (Package.swift and Cmlx.xcconfig), so the two have to agree.
    func testMLXVersionMatchesTheMLXSources() throws {
        let header = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "Source/Cmlx/mlx/mlx/version.h")
        let text = try String(contentsOf: header, encoding: .utf8)
        let major = try XCTUnwrap(text.firstMatch(of: /#define MLX_VERSION_MAJOR (\d+)/)).1
        let minor = try XCTUnwrap(text.firstMatch(of: /#define MLX_VERSION_MINOR (\d+)/)).1
        let patch = try XCTUnwrap(text.firstMatch(of: /#define MLX_VERSION_PATCH (\d+)/)).1

        var version = mlx_string_new()
        defer { mlx_string_free(version) }
        mlx_version(&version)

        XCTAssertEqual(String(cString: mlx_string_data(version)), "\(major).\(minor).\(patch)")
    }
}
