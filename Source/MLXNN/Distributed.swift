// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// How a linear layer is sharded across a group.
public enum ShardingType: Sendable {
    /// A common input becomes a sharded output.
    case allToSharded

    /// A sharded input becomes a common output.
    case shardedToAll
}

/// Describes how an unsharded weight is composed.
///
/// A fused QKV matrix, for example, is three segments stacked together and
/// each has to be sharded separately.
public enum Segments: Sendable, Equatable {
    /// The weight is `count` equally sized segments.
    case count(Int)

    /// The weight is split at these indices.
    case indices([Int])

    /// The weight is split at these fractions of the axis.
    case fractions([Double])

    /// The positions that separate the segments of an axis of length `dimension`.
    ///
    /// Throws if `count` equally sized segments do not fit, which
    /// `split(parts:)` would report with a `fatalError`.
    func boundaries(of dimension: Int) throws -> [Int] {
        switch self {
        case .count(let count):
            guard count >= 1, dimension % count == 0 else {
                throw ShardingError.invalidSegments(self, dimension: dimension)
            }
            return (1 ..< count).map { $0 * dimension / count }
        case .indices(let indices):
            return indices
        case .fractions(let fractions):
            return fractions.map { Int($0 * Double(dimension)) }
        }
    }
}

/// A layer cannot be sharded across a group.
public enum ShardingError: Error, CustomStringConvertible, Equatable {
    /// A dimension is not divisible by the size of the group.
    case indivisible(dimension: String, of: Int, across: Int)

    /// A shard would not hold whole quantization groups.
    case quantizationGroup(inputDimensions: Int, groupSize: Int, across: Int)

    /// The segments cannot split a dimension, e.g. `.count(3)` of 8.
    case invalidSegments(Segments, dimension: Int)

    /// A ``QuantizedLinear`` was given to a float sharded layer.
    ///
    /// ``shardLinear(_:sharding:segments:group:)`` returns the quantized flavor.
    case quantizedLayer

    /// The layer being sharded is missing a parameter.
    case missingParameter(String)

    /// A ``FullyShardedModule`` cannot shard a scalar parameter.
    case scalarParameter(String)

    public var description: String {
        switch self {
        case .indivisible(let dimension, let value, let size):
            "Cannot shard the \(dimension) of size \(value) across \(size) processes."
        case .quantizationGroup(let inputDimensions, let groupSize, let size):
            """
            Sharding \(inputDimensions) inputs across \
            \(size == 1 ? "1 process" : "\(size) processes") splits a quantization group of \
            \(groupSize).
            """
        case .invalidSegments(let segments, let dimension):
            "The segments \(segments) do not fit a dimension of size \(dimension)."
        case .quantizedLayer:
            """
            A quantized layer cannot be sharded as a float layer.  Use shardLinear, \
            QuantizedAllToShardedLinear or QuantizedShardedToAllLinear.
            """
        case .missingParameter(let name):
            "The layer being sharded has no \(name)."
        case .scalarParameter(let path):
            "Cannot shard the parameter \(path) because it is a scalar."
        }
    }
}

/// Returns a function that is the identity in the forward pass and sums the
/// gradients across the group in the backward pass.
///
/// - Parameter group: the group to sum across
public func sumGradients(group: MLXDistributed.Group) -> (MLXArray) -> MLXArray {
    if group.size == 1 {
        return { $0 }
    }

    let f = CustomFunction {
        Forward { inputs in
            inputs
        }
        VJP { _, cotangents in
            cotangents.map { MLXDistributed.allSum($0, group: group) }
        }
    }

    return { f([$0])[0] }
}

/// The sharding axis and segments for a parameter, or `nil` to leave it alone.
///
/// The path is the parameter's path in the module, e.g. `layers.0.conv.weight`.
public typealias ShardingPredicate = (String, MLXArray) -> (axis: Int, segments: Segments)?

/// Shard the rows -- the output dimensions -- of every parameter.
///
/// `scales` and `biases` hold one row per output row, so they shard the same
/// way as the weight.  `bias`, the affine bias, is one dimensional.
private func allToShardedPredicate(_ segments: Segments) -> ShardingPredicate {
    { path, weight in
        path.hasSuffix("bias") ? (-1, segments) : (max(weight.ndim - 2, 0), segments)
    }
}

/// Shard the columns -- the input dimensions -- of every parameter.
///
/// The affine bias applies to the summed result, so every process keeps it
/// whole.
private func shardedToAllPredicate(_ segments: Segments) -> ShardingPredicate {
    { path, _ in
        path.hasSuffix("bias") ? nil : (-1, segments)
    }
}

private func predicate(for sharding: ShardingType, segments: Segments) -> ShardingPredicate {
    switch sharding {
    case .allToSharded: allToShardedPredicate(segments)
    case .shardedToAll: shardedToAllPredicate(segments)
    }
}

/// Returns a new parameter tree holding this process' shard of the weights.
///
/// The segments are checked in a group of one as well, where nothing is split,
/// so that segments that cannot work are reported before the model runs on
/// several processes.
private func shard(
    _ parameters: ModuleParameters, group: MLXDistributed.Group,
    _ sharding: ShardingPredicate
) throws -> ModuleParameters {
    let size = group.size
    let rank = group.rank

    let sharded = try parameters.flattened().map { path, weight -> (String, MLXArray) in
        // a scalar -- the NVFP4 global scale, for example -- is the same in
        // every process
        guard weight.ndim > 0, let (axis, segments) = sharding(path, weight) else {
            return (path, weight)
        }

        let boundaries = try segments.boundaries(of: weight.dim(axis))
        guard size > 1 else {
            return (path, weight)
        }

        let segmented =
            boundaries.isEmpty ? [weight] : weight.split(indices: boundaries, axis: axis)
        let parts = try segmented.map { part -> MLXArray in
            // split(parts:) needs equal sections and calls fatalError rather
            // than reporting, so check before asking.  A dimension can divide
            // by the group size while one of its segments does not.
            guard part.dim(axis) % size == 0 else {
                throw ShardingError.indivisible(
                    dimension: "segment of \(path)", of: part.dim(axis), across: size)
            }
            return part.split(parts: size, axis: axis)[rank]
        }

        return (path, concatenated(parts, axis: axis).contiguous())
    }

    // in a group of one every parameter stays whole
    return size == 1 ? parameters : ModuleParameters.unflattened(sharded)
}

/// Shard the input dimensions of a quantized layer.
///
/// The packed weight, the scales and the biases hold the inputs at different
/// resolutions, so a boundary between two segments lands at a different
/// position in each.  Every segment has to hold whole quantization groups on
/// every process.  Then every boundary lies on a group edge and scales to each
/// parameter exactly in integer arithmetic, because every supported mode packs
/// a group into whole `uint32` words.  Scaling through a `Double` could
/// truncate to the position before.
private func quantizedShardedToAllPredicate(
    _ segments: Segments, inputDimensions: Int, groupSize: Int, size: Int
) throws -> ShardingPredicate {
    let boundaries = try segments.boundaries(of: inputDimensions)
    guard boundaries == boundaries.sorted(),
        boundaries.allSatisfy({ (0 ... inputDimensions).contains($0) })
    else {
        throw ShardingError.invalidSegments(segments, dimension: inputDimensions)
    }

    let edges = [0] + boundaries + [inputDimensions]
    for (start, end) in zip(edges, edges.dropFirst()) {
        guard (end - start) % (groupSize * size) == 0 else {
            throw ShardingError.quantizationGroup(
                inputDimensions: end - start, groupSize: groupSize, across: size)
        }
    }

    return { path, weight in
        path.hasSuffix("bias")
            ? nil : (-1, .indices(boundaries.map { $0 * weight.dim(-1) / inputDimensions }))
    }
}

/// Shard a module in place by replacing its parameters with sharded ones.
///
/// The module itself is unchanged, so distributed communication only happens
/// if the module supports it natively.
///
/// - Parameters:
///   - module: the module whose parameters are sharded in place
///   - sharding: the kind of sharding to apply
///   - segments: the segments that comprise each unsharded weight
///   - group: the group to shard across, or `nil` to use the global group
public func shardInPlace(
    _ module: Module, sharding: ShardingType, segments: Segments = .count(1),
    group: MLXDistributed.Group? = nil
) throws {
    try shardInPlace(
        module, predicate: predicate(for: sharding, segments: segments), group: group)
}

/// Shard a module in place, choosing how each parameter is split.
///
/// This is the general form of ``shardInPlace(_:sharding:segments:group:)``, and
/// matches what Python's `shard_inplace` accepts.  The predicate receives a
/// parameter and its path and returns the axis to shard it along, with the
/// segments that make it up, or `nil` to leave the parameter whole.
///
/// The module itself is unchanged, so distributed communication only happens
/// if the module supports it natively.
///
/// - Parameters:
///   - module: the module whose parameters are sharded in place
///   - predicate: chooses how each parameter is sharded
///   - group: the group to shard across, or `nil` to use the global group
public func shardInPlace(
    _ module: Module, predicate: ShardingPredicate, group: MLXDistributed.Group? = nil
) throws {
    let group = try group ?? MLXDistributed.initialize()
    _ = try module.update(parameters: shard(module.parameters(), group: group, predicate))
}

/// Create a new linear layer with sharded parameters that also performs the
/// distributed communication, either in the forward or the backward pass.
///
/// Unlike ``shardInPlace(_:sharding:segments:group:)`` the original layer is
/// not changed.  A ``QuantizedLinear`` yields the quantized flavor of the
/// sharded layer, so a quantized model shards like any other.
///
/// - Parameters:
///   - layer: the linear layer to shard
///   - sharding: the kind of sharding to apply
///   - segments: the segments that comprise each unsharded weight
///   - group: the group to shard across, or `nil` to use the global group
/// - Returns: the sharded layer, which can replace `layer` in a model
public func shardLinear(
    _ layer: Linear, sharding: ShardingType, segments: Segments = .count(1),
    group: MLXDistributed.Group? = nil
) throws -> Linear {
    // QuantizedLinear is a Linear, so it has to be matched first
    if let layer = layer as? QuantizedLinear {
        return switch sharding {
        case .allToSharded:
            try QuantizedAllToShardedLinear(layer, segments: segments, group: group)
        case .shardedToAll:
            try QuantizedShardedToAllLinear(layer, segments: segments, group: group)
        }
    }

    return switch sharding {
    case .allToSharded:
        try AllToShardedLinear(layer, segments: segments, group: group)
    case .shardedToAll:
        try ShardedToAllLinear(layer, segments: segments, group: group)
    }
}

/// Each member of the group applies part of the affine transformation so that
/// the result is sharded across the group.
///
/// The gradients are automatically aggregated from each member of the group.
///
/// This is a ``Linear``, so it can replace one in a model.
open class AllToShardedLinear: Linear {

    /// The group the output dimensions are sharded across.
    public let group: MLXDistributed.Group

    private let aggregateGradients: (MLXArray) -> MLXArray

    /// - Parameters:
    ///   - inputDimensions: number of input dimensions
    ///   - outputDimensions: number of output dimensions, sharded across the group
    ///   - bias: if `true` this layer will apply a bias
    ///   - group: the group to shard across, or `nil` to use the global group
    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true,
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        guard outputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "output", of: outputDimensions, across: group.size)
        }

        self.group = group
        self.aggregateGradients = sumGradients(group: group)

        let scale = sqrt(1.0 / Float(inputDimensions))
        let shardedOutput = outputDimensions / group.size
        super.init(
            weight: MLXRandom.uniform(-scale ..< scale, [shardedOutput, inputDimensions]),
            bias: bias ? MLXRandom.uniform(-scale ..< scale, [shardedOutput]) : nil)
    }

    /// Hold parameters that are already this process' shard.
    init(shardedWeight weight: MLXArray, bias: MLXArray?, group: MLXDistributed.Group) {
        self.group = group
        self.aggregateGradients = sumGradients(group: group)
        super.init(weight: weight, bias: bias)
    }

    /// Create a sharded layer from an existing ``Linear``.
    ///
    /// Throws for a ``QuantizedLinear``: shard it with
    /// ``shardLinear(_:sharding:segments:group:)`` or ``QuantizedAllToShardedLinear``.
    public convenience init(
        _ other: Linear, segments: Segments = .count(1), group: MLXDistributed.Group? = nil
    ) throws {
        // a QuantizedLinear is a Linear, but its packed weight is not a float one
        guard !(other is QuantizedLinear) else {
            throw ShardingError.quantizedLayer
        }

        let group = try group ?? MLXDistributed.initialize()
        let (outputDimensions, _) = other.shape
        guard outputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "output", of: outputDimensions, across: group.size)
        }

        let parameters = try Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group, allToShardedPredicate(segments)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }

        self.init(shardedWeight: weight, bias: parameters["bias"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // every shard reads the same input, so their gradients are summed
        let x = aggregateGradients(x)

        if let bias {
            return addMM(bias, x, weight.T)
        } else {
            return matmul(x, weight.T)
        }
    }

    /// Quantizing keeps the layer sharded.
    ///
    /// Without this a quantized model built by ``quantize(model:groupSize:bits:mode:filter:apply:)``
    /// would hold plain ``QuantizedLinear`` layers that no longer communicate.
    public override func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedAllToShardedLinear(
            shardedWeight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode,
            group: group)
    }
}

/// Each member of the group applies part of the affine transformation and the
/// results are then aggregated.
///
/// Every member of the group ends up with the same result.
///
/// This is a ``Linear``, so it can replace one in a model.
open class ShardedToAllLinear: Linear {

    /// The group the input dimensions are sharded across.
    public let group: MLXDistributed.Group

    /// - Parameters:
    ///   - inputDimensions: number of input dimensions, sharded across the group
    ///   - outputDimensions: number of output dimensions
    ///   - bias: if `true` this layer will apply a bias
    ///   - group: the group to shard across, or `nil` to use the global group
    public init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true,
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        guard inputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "input", of: inputDimensions, across: group.size)
        }

        self.group = group

        // Each process holds its own slice of the weight, but the bias is
        // added after the reduction, so every process has to hold the same
        // one.  Random values would differ from process to process -- their
        // generators are seeded independently -- and the layer would quietly
        // produce a different result in each.  Python's quantized flavor uses
        // zeros for this reason; its float one does not, and diverges.
        let scale = sqrt(1.0 / Float(inputDimensions))
        super.init(
            weight: MLXRandom.uniform(
                -scale ..< scale, [outputDimensions, inputDimensions / group.size]),
            bias: bias ? MLXArray.zeros([outputDimensions]) : nil)
    }

    /// Hold parameters that are already this process' shard.
    init(shardedWeight weight: MLXArray, bias: MLXArray?, group: MLXDistributed.Group) {
        self.group = group
        super.init(weight: weight, bias: bias)
    }

    /// Create a sharded layer from an existing ``Linear``.
    ///
    /// Throws for a ``QuantizedLinear``: shard it with
    /// ``shardLinear(_:sharding:segments:group:)`` or ``QuantizedShardedToAllLinear``.
    public convenience init(
        _ other: Linear, segments: Segments = .count(1), group: MLXDistributed.Group? = nil
    ) throws {
        // a QuantizedLinear is a Linear, but its packed weight is not a float one
        guard !(other is QuantizedLinear) else {
            throw ShardingError.quantizedLayer
        }

        let group = try group ?? MLXDistributed.initialize()
        let (_, inputDimensions) = other.shape
        guard inputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "input", of: inputDimensions, across: group.size)
        }

        let parameters = try Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group, shardedToAllPredicate(segments)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }

        self.init(shardedWeight: weight, bias: parameters["bias"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // each process holds part of the sum, and the bias belongs to the
        // whole, so it is added after the reduction
        var x = matmul(x, weight.T)
        x = MLXDistributed.allSum(x, group: group)

        if let bias {
            x = x + bias
        }
        return x
    }

    /// Quantizing keeps the layer sharded.
    public override func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedShardedToAllLinear(
            shardedWeight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode,
            group: group)
    }
}

/// The quantized flavor of ``AllToShardedLinear``.
///
/// Like ``QuantizedLinear`` its parameters are frozen.
open class QuantizedAllToShardedLinear: QuantizedLinear {

    /// The group the output dimensions are sharded across.
    public let group: MLXDistributed.Group

    private let aggregateGradients: (MLXArray) -> MLXArray

    /// Quantize a weight that is already this process' shard.
    public init(
        shardedWeight weight: MLXArray, bias: MLXArray?, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine, group: MLXDistributed.Group
    ) {
        self.group = group
        self.aggregateGradients = sumGradients(group: group)
        super.init(weight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode)
    }

    /// Hold quantized parameters that are already this process' shard.
    init(
        shardedQuantizedWeight weight: MLXArray, bias: MLXArray?, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int, mode: QuantizationMode,
        globalScale: MLXArray?, group: MLXDistributed.Group
    ) {
        self.group = group
        self.aggregateGradients = sumGradients(group: group)
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode, globalScale: globalScale)
        self.freeze()
    }

    /// Create a sharded layer from an existing ``QuantizedLinear``.
    ///
    /// The packed weight, `scales` and `biases` all carry one row per output,
    /// so they shard together along the output dimension.
    public convenience init(
        _ other: QuantizedLinear, segments: Segments = .count(1),
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        let (outputDimensions, _) = other.shape
        guard outputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "output", of: outputDimensions, across: group.size)
        }

        let parameters = try Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group, allToShardedPredicate(segments)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }
        guard let scales = parameters["scales"] else {
            throw ShardingError.missingParameter("scales")
        }

        self.init(
            shardedQuantizedWeight: weight, bias: parameters["bias"], scales: scales,
            biases: parameters["biases"], groupSize: other.groupSize, bits: other.bits,
            mode: other.mode, globalScale: parameters["global_scale"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // the quantized matmul, the global scale and the bias are all local to
        // this shard, so only the gradient aggregation is added
        super.callAsFunction(aggregateGradients(x))
    }
}

/// The quantized flavor of ``ShardedToAllLinear``.
///
/// Like ``QuantizedLinear`` its parameters are frozen.
open class QuantizedShardedToAllLinear: QuantizedLinear {

    /// The group the input dimensions are sharded across.
    public let group: MLXDistributed.Group

    /// Quantize a weight that is already this process' shard.
    public init(
        shardedWeight weight: MLXArray, bias: MLXArray?, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine, group: MLXDistributed.Group
    ) {
        self.group = group
        super.init(weight: weight, bias: bias, groupSize: groupSize, bits: bits, mode: mode)
    }

    /// Hold quantized parameters that are already this process' shard.
    init(
        shardedQuantizedWeight weight: MLXArray, bias: MLXArray?, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int, mode: QuantizationMode,
        globalScale: MLXArray?, group: MLXDistributed.Group
    ) {
        self.group = group
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode, globalScale: globalScale)
        self.freeze()
    }

    /// Create a sharded layer from an existing ``QuantizedLinear``.
    ///
    /// This shards the input dimension, which is the packed one, so every
    /// segment has to hold whole quantization groups on every process.
    public convenience init(
        _ other: QuantizedLinear, segments: Segments = .count(1),
        group: MLXDistributed.Group? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        let (_, inputDimensions) = other.shape
        guard inputDimensions % group.size == 0 else {
            throw ShardingError.indivisible(
                dimension: "input", of: inputDimensions, across: group.size)
        }

        let parameters = try Dictionary(
            uniqueKeysWithValues: shard(
                other.parameters(), group: group,
                quantizedShardedToAllPredicate(
                    segments, inputDimensions: inputDimensions, groupSize: other.groupSize,
                    size: group.size)
            ).flattened())
        guard let weight = parameters["weight"] else {
            throw ShardingError.missingParameter("weight")
        }
        guard let scales = parameters["scales"] else {
            throw ShardingError.missingParameter("scales")
        }

        self.init(
            shardedQuantizedWeight: weight, bias: parameters["bias"], scales: scales,
            biases: parameters["biases"], groupSize: other.groupSize, bits: other.bits,
            mode: other.mode, globalScale: parameters["global_scale"], group: group)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // this cannot call super: the bias applies to the summed result, not to
        // this process' partial product
        var x = quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize,
            bits: bits, mode: mode)
        x = applyNVFP4GlobalScale(x, globalScale: globalScale)
        x = MLXDistributed.allSum(x, group: group)

        if let bias {
            x = x + bias
        }
        return x
    }
}

/// Returns a function that gathers the shards of parameters in the forward pass
/// and reduce-scatters their gradients in the backward pass.
///
/// The shards travel in one buffer each way.
///
/// - Parameters:
///   - group: the group the parameters are sharded across
///   - fullShapes: the shape of each whole parameter
///   - shardSizes: the number of elements in a shard of each parameter
///   - computeDType: the type to cast the gathered parameters to, or `nil` to
///     leave them alone
private func makeGather(
    group: MLXDistributed.Group, fullShapes: [[Int]], shardSizes: [Int], computeDType: DType?
) -> ([MLXArray]) -> [MLXArray] {
    let size = group.size
    let splitIndices = shardSizes.dropLast().reduce(into: [Int]()) {
        $0.append(($0.last ?? 0) + $1)
    }
    let shardShapes = fullShapes.map { [$0[0] / size] + $0.dropFirst() }

    func maybeCast(_ x: MLXArray, _ dtype: DType?) -> MLXArray {
        guard let dtype, x.dtype != dtype else {
            return x
        }
        return x.asType(dtype)
    }

    func split(_ x: MLXArray) -> [MLXArray] {
        splitIndices.isEmpty ? [x] : x.split(indices: splitIndices, axis: 1)
    }

    return CustomFunction {
        Forward { shards in
            let shard = concatenated(
                shards.map { maybeCast($0.reshaped([1, -1]), computeDType) }, axis: 1)
            let full = MLXDistributed.allGather(shard, group: group)
            return zip(split(full), fullShapes).map { $0.reshaped($1) }
        }
        VJP { shards, cotangents in
            let localFull = concatenated(cotangents.map { $0.reshaped([size, -1]) }, axis: 1)
            let localShard = MLXDistributed.sumScatter(localFull, group: group) / size
            let parts = split(localShard)
            return shards.indices.map { i in
                maybeCast(parts[i].reshaped(shardShapes[i]), shards[i].dtype)
            }
        }
    }
}

/// A ``FullyShardedModule`` whatever module it wraps.
private protocol FullySharded {}

/// ``Module/filterValidParameters``, except that a module that is fully sharded
/// itself gathers its own parameters.
private func filterShardable(module: Module, key: String, item: ModuleItem) -> Bool {
    if case .value(.module(let child)) = item, child is FullySharded {
        return false
    }
    return Module.filterValidParameters(module, key, item)
}

/// Wraps a module so that each member of the group holds only a shard of its
/// parameters.
///
/// The whole parameters are gathered for the forward pass and the gradients
/// are reduce-scattered in the backward pass, so during training each member
/// of the group stores and updates only its own shard.
///
/// Every parameter is sharded along its first axis, so that axis must be
/// divisible by the size of the group.  The parameters of the wrapped module
/// are under `module`.
///
/// ```swift
/// let group = try MLXDistributed.initialize()
/// let model = try fullyShard(MLP(), group: group, computeDType: .bfloat16)
///
/// func loss(model: FullyShardedModule<MLP>, x: MLXArray, y: MLXArray) -> MLXArray {
///     mseLoss(predictions: model(x), targets: y, reduction: .mean)
/// }
///
/// // the gradients hold this process' shard of each parameter
/// let (value, gradients) = valueAndGrad(model: model, loss)(model, x, y)
/// optimizer.update(model: model, gradients: gradients)
/// ```
///
/// A wrapped ``UnaryLayer`` is called like one.  Any other module is called in
/// a closure, `block { $0(x, mask: mask, cache: cache) }`.
///
/// A module that holds fully sharded modules leaves their parameters to them,
/// so a model can gather one layer at a time: wrap the layers, then the model.
///
/// The backward pass needs `MLXDistributed.sumScatter`, which the ring backend
/// doesn't implement.
open class FullyShardedModule<Wrapped: Module>: Module, FullySharded {

    /// The wrapped module, which holds this process' shard of the parameters
    /// outside of a call.
    public let module: Wrapped

    /// The paths of the sharded parameters in ``module``.
    private let paths: [String]

    private let gather: ([MLXArray]) -> [MLXArray]

    /// Shard the parameters of `module` across the group.
    ///
    /// - Parameters:
    ///   - module: the module whose parameters are sharded in place
    ///   - group: the group to shard across, or `nil` to use the global group
    ///   - computeDType: the type the gathered parameters are cast to for the
    ///     forward pass, or `nil` to leave them alone
    public init(
        _ module: Wrapped, group: MLXDistributed.Group? = nil, computeDType: DType? = nil
    ) throws {
        let group = try group ?? MLXDistributed.initialize()
        let size = group.size

        let parameters = module.filterMap(
            filter: filterShardable, map: Module.mapParameters())
        let flat = parameters.flattened()
        for (path, array) in flat {
            guard array.ndim > 0 else {
                throw ShardingError.scalarParameter(path)
            }
            guard array.dim(0) % size == 0 else {
                throw ShardingError.indivisible(
                    dimension: "first axis of \(path)", of: array.dim(0), across: size)
            }
        }

        self.paths = flat.map { $0.0 }
        let fullShapes = flat.map { $0.1.shape }
        let shardSizes = flat.map { $0.1.size / size }

        module.update(parameters: try shard(parameters, group: group) { _, _ in (0, .count(1)) })

        self.module = module
        self.gather = makeGather(
            group: group, fullShapes: fullShapes, shardSizes: shardSizes,
            computeDType: computeDType)
    }

    open override func describeExtra(_ indent: Int) -> String {
        "(shardedParameters=\(paths.count))"
    }

    /// Call `body` with the wrapped module while it holds the whole parameters.
    ///
    /// This calls a module that isn't a ``UnaryLayer``, for example
    /// `block { $0(x, mask: mask, cache: cache) }`.  The shards are put back
    /// when `body` returns or throws.
    public func callAsFunction<Result>(_ body: (Wrapped) throws -> Result) rethrows -> Result {
        guard !paths.isEmpty else {
            return try body(module)
        }

        // update(parameters:) replaces the contents of the module's arrays, so
        // the shards are held in arrays of their own to be put back
        let shards = module.filterMap(
            filter: filterShardable, map: Module.mapParameters { $0.reshaped($0.shape) })
        let fulls = gather(shards.flattened().map { $0.1 })
        module.update(parameters: ModuleParameters.unflattened(Array(zip(paths, fulls))))
        defer {
            module.update(parameters: shards)
        }

        return try body(module)
    }
}

extension FullyShardedModule: UnaryLayer where Wrapped: UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction { $0(x) }
    }
}

extension FullyShardedModule where Wrapped: Embedding {
    /// Call ``Embedding/asLinear(_:)`` with the whole parameters, for a model
    /// whose output projection shares the embedding's weight.
    public func asLinear(_ x: MLXArray) -> MLXArray {
        callAsFunction { $0.asLinear(x) }
    }
}

/// Wrap `module` in a ``FullyShardedModule``.
///
/// In a group of one Python's `fully_shard` returns the module unchanged.  This
/// wraps it all the same, so that the parameter paths, and the type the forward
/// pass computes in, don't depend on the size of the group.
///
/// - Parameters:
///   - module: the module whose parameters are sharded in place
///   - group: the group to shard across, or `nil` to use the global group
///   - computeDType: the type the gathered parameters are cast to for the
///     forward pass, or `nil` to leave them alone
public func fullyShard<Wrapped: Module>(
    _ module: Wrapped, group: MLXDistributed.Group? = nil, computeDType: DType? = nil
) throws -> FullyShardedModule<Wrapped> {
    try FullyShardedModule(module, group: group, computeDType: computeDType)
}

/// A module that is fully sharded already is returned unchanged.
public func fullyShard<Wrapped: Module>(
    _ module: FullyShardedModule<Wrapped>, group: MLXDistributed.Group? = nil,
    computeDType: DType? = nil
) -> FullyShardedModule<Wrapped> {
    module
}

/// Average the gradients across the processes in the group.
///
/// Small gradients are concatenated into batches of at least `allReduceSize`
/// bytes so that they are communicated in one step, which is considerably
/// faster than one call per array.
///
/// - Parameters:
///   - gradients: the gradients, which must have the same structure in every process
///   - group: the group to average across, or `nil` to use the global group
///   - allReduceSize: group arrays until their size in bytes exceeds this,
///     or `0` to disable grouping
///   - stream: stream to evaluate on
public func averageGradients(
    _ gradients: ModuleParameters, group: MLXDistributed.Group? = nil,
    allReduceSize: Int = 32 * 1024 * 1024, stream: StreamOrDevice = .cpu
) throws -> ModuleParameters {
    let group = try group ?? MLXDistributed.initialize()
    let size = group.size
    if size == 1 {
        return gradients
    }

    let flat = gradients.flattened()
    if flat.isEmpty {
        return gradients
    }

    // one all reduce per gradient
    if allReduceSize <= 0 {
        return ModuleParameters.unflattened(
            flat.map {
                ($0.0, MLXDistributed.allSum($0.1, group: group, stream: stream) / size)
            })
    }

    // arrays of mixed types cannot be concatenated
    let dtype = flat[0].1.dtype
    guard flat.allSatisfy({ $0.1.dtype == dtype }) else {
        return try averageGradients(
            gradients, group: group, allReduceSize: 0, stream: stream)
    }

    // gather the gradients into groups that are at least allReduceSize bytes
    let batches = groupBySize(flat.map { $0.1.nbytes }, limit: allReduceSize)

    // concatenate, reduce, split
    var result = [(String, MLXArray)]()
    for batch in batches {
        let sizes = batch.map { flat[$0].1.size }
        var big = concatenated(batch.map { flat[$0].1.reshaped([-1]) })
        big = MLXDistributed.allSum(big, group: group, stream: stream) / size

        let indices = sizes.dropLast().reduce(into: [Int]()) { $0.append(($0.last ?? 0) + $1) }
        let parts = indices.isEmpty ? [big] : big.split(indices: indices)

        for (part, i) in zip(parts, batch) {
            result.append((flat[i].0, part.reshaped(flat[i].1.shape)))
        }
    }

    return ModuleParameters.unflattened(result)
}

/// Group consecutive arrays, given their sizes in bytes, into batches of at
/// least `limit` bytes.  The last batch holds whatever remains.
///
/// This is `_group_by_size` from Python's `mlx.nn.utils`.  It is a function of
/// its own so that the tests can count the batches `averageGradients` forms,
/// the way the Python tests count its `all_sum` calls.
func groupBySize(_ sizes: [Int], limit: Int) -> [[Int]] {
    var batches = [[Int]]()
    var batch = [Int]()
    var batchBytes = 0
    for (i, size) in sizes.enumerated() {
        batch.append(i)
        batchBytes += size
        if batchBytes >= limit {
            batches.append(batch)
            batch = []
            batchBytes = 0
        }
    }
    if !batch.isEmpty {
        batches.append(batch)
    }
    return batches
}

/// Clip the global norm of gradients that are sharded across a group.
///
/// This is the sharded counterpart of `clipGradNorm` in MLXOptimizers: no
/// process holds the whole gradient, so the local squared norms are summed
/// across the group before anything is rescaled.  It clips the gradients of a
/// ``FullyShardedModule``, for example.
///
/// - Parameters:
///   - gradients: this process' shard of the gradients
///   - maxNorm: the maximum allowed global norm
///   - group: the group the gradients are sharded across, or `nil` to use the
///     global group
///   - stream: stream to evaluate on
/// - Returns: the rescaled shard and the global gradient norm
public func clipGradNormSharded(
    gradients: ModuleParameters, maxNorm: Float, group: MLXDistributed.Group? = nil,
    stream: StreamOrDevice = .cpu
) throws -> (ModuleParameters, MLXArray) {
    let group = try group ?? MLXDistributed.initialize()

    let localNormSquared = gradients.reduce(MLXArray(0)) { $0 + $1.square().sum() }
    let totalNorm = sqrt(MLXDistributed.allSum(localNormSquared, group: group, stream: stream))
    let normalizer = minimum(maxNorm / (totalNorm + 1e-6), 1)

    return (gradients.mapValues { $0 * normalizer }, totalNorm)
}
