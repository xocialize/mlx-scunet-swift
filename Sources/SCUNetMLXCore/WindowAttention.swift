//
//  WindowAttention.swift
//  mlx-scunet-swift / SCUNetMLXCore
//
//  WMSA — Swin-style window multi-head self-attention, with the shifted-window (SW) variant.
//  Ported from `models/network_scunet.py`.
//
//  Upstream: https://github.com/cszn/SCUNet (Apache-2.0)
//  Paper:    Zhang et al., "Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis".
//
//  This is the first genuine shifted-window attention in the fleet, and **both of its failure modes
//  are shape-safe** — they produce plausible output rather than an error:
//
//   1. The QKV head split. Upstream's `rearrange('b nw np (threeh c) -> threeh b nw np c')` followed
//      by `.chunk(3, dim=0)` means the projection's output channels are ordered
//      **[all q heads][all k heads][all v heads]**, NOT interleaved per head. Splitting it the
//      obvious way (q,k,v per head) yields the same shapes and silently wrong attention.
//   2. The SW attention mask. It masks only the LAST window row and column — the wrap-around
//      created by the cyclic roll. Omitting it lets opposite edges of the image attend to each
//      other, which looks like mild artefacts rather than a crash.
//
//  Conventions: NHWC. Upstream's WMSA already operates in `b h w c`, so no transposes are needed.
//

import Foundation
import MLX
import MLXNN

/// The constant relative-position gather table, shared across every block with the same window size.
///
/// A class rather than a stored `MLXArray` so `Module` reflection walks past it — see the note on
/// `WMSA.relIndex`.
final class RelativeIndexTable: @unchecked Sendable {
    let indices: MLXArray

    private static let lock = NSLock()
    private nonisolated(unsafe) static var cache: [Int: RelativeIndexTable] = [:]

    static func shared(window: Int) -> RelativeIndexTable {
        lock.lock(); defer { lock.unlock() }
        if let t = cache[window] { return t }
        let t = RelativeIndexTable(window: window)
        cache[window] = t
        return t
    }

    /// `relation[p, q] = coord[p] − coord[q] + (window − 1)` over the window² positions, flattened
    /// into the `(2w−1) × (2w−1)` table so the gather is a single `take` on one axis.
    private init(window: Int) {
        let n = window * window
        var flat = [Int32](repeating: 0, count: n * n)
        for p in 0 ..< n {
            let (pi, pj) = (p / window, p % window)
            for q in 0 ..< n {
                let (qi, qj) = (q / window, q % window)
                let r = pi - qi + window - 1
                let c = pj - qj + window - 1
                flat[p * n + q] = Int32(r * (2 * window - 1) + c)
            }
        }
        self.indices = MLXArray(flat, [n * n])
    }
}

public final class WMSA: Module, UnaryLayer, @unchecked Sendable {

    public enum Kind: String, Sendable { case w = "W", sw = "SW" }

    @ModuleInfo(key: "embedding_layer") public var embedding: Linear
    @ModuleInfo(key: "linear") public var linear: Linear
    /// Stored `(nHeads, 2·window−1, 2·window−1)`.
    ///
    /// ⚠️ That is **not** the shape the declaration suggests. `WMSA.__init__` allocates
    /// `((2w−1)², heads)` and then *re-assigns* the parameter through
    /// `.view(2w−1, 2w−1, heads).transpose(1,2).transpose(0,1)`, so the checkpoint carries the
    /// permuted form. Read the constructor, not the declaration.
    @ParameterInfo(key: "relative_position_params") public var relativePositionParams: MLXArray

    private let inputDim: Int
    private let headDim: Int
    private let nHeads: Int
    private let window: Int
    private let kind: Kind
    private let scale: Float

    /// Flattened `(row, col)` gather indices into the `(2w−1, 2w−1)` bias table.
    ///
    /// Upstream rebuilds this on every forward from a coordinate difference; it is constant, and it
    /// depends only on `window`, so all 28 blocks share one table.
    ///
    /// ⚠️ Boxed deliberately. `Module`'s reflection collects **every stored `MLXArray` property** as
    /// a parameter, so declaring this as a bare `MLXArray` adds 28 phantom tensors that no checkpoint
    /// can satisfy — S0 fails with 28 missing keys. A constant is not a parameter; the box keeps it
    /// out of the tree.
    private let relIndex: RelativeIndexTable

    public init(inputDim: Int, outputDim: Int, headDim: Int, window: Int, kind: Kind) {
        self.inputDim = inputDim
        self.headDim = headDim
        self.nHeads = inputDim / headDim
        self.window = window
        self.kind = kind
        self.scale = pow(Float(headDim), -0.5)

        self._embedding.wrappedValue = Linear(inputDim, 3 * inputDim, bias: true)
        self._linear.wrappedValue = Linear(inputDim, outputDim, bias: true)
        self._relativePositionParams.wrappedValue =
            MLXArray.zeros([inputDim / headDim, 2 * window - 1, 2 * window - 1])

        self.relIndex = RelativeIndexTable.shared(window: window)
    }

    /// `(nHeads, window², window²)` — the learned bias added to every attention logit.
    ///
    /// Public so the S1 gate can compare the table *before* it is consumed. Both of this module's
    /// silent failure modes live in tables, so gating them directly is what turns "the image is
    /// slightly off" into a named cause.
    public func relativeBias() -> MLXArray {
        let n = window * window
        let t = relativePositionParams.reshaped(nHeads, (2 * window - 1) * (2 * window - 1))
        return t.take(relIndex.indices, axis: 1).reshaped(nHeads, n, n)
    }

    /// The shifted-window mask: `true` where attention must be suppressed.
    ///
    /// The cyclic roll wraps content from one edge of the image to the other, so within the LAST
    /// window row (and column) some positions are spatially adjacent only by wrap-around. Those
    /// pairs get `-inf`. Interior windows are unaffected, which is why only index `-1` is touched.
    /// Returns `nil` for a `W` block — which is itself the assertion that the type survived
    /// construction. Public for the S1 gate; see `relativeBias()`.
    public func shiftMask(hWindows: Int, wWindows: Int) -> MLXArray? {
        guard kind == .sw else { return nil }
        let p = window, shift = window / 2, s = p - shift
        var m = [Bool](repeating: false, count: hWindows * wWindows * p * p * p * p)
        @inline(__always) func idx(_ w1: Int, _ w2: Int, _ p1: Int, _ p2: Int, _ p3: Int, _ p4: Int) -> Int {
            (((((w1 * wWindows + w2) * p + p1) * p + p2) * p + p3) * p + p4)
        }
        let lastRow = hWindows - 1, lastCol = wWindows - 1
        for w2 in 0 ..< wWindows {
            for p1 in 0 ..< p { for p2 in 0 ..< p { for p3 in 0 ..< p { for p4 in 0 ..< p {
                if (p1 < s && p3 >= s) || (p1 >= s && p3 < s) { m[idx(lastRow, w2, p1, p2, p3, p4)] = true }
            }}}}
        }
        for w1 in 0 ..< hWindows {
            for p1 in 0 ..< p { for p2 in 0 ..< p { for p3 in 0 ..< p { for p4 in 0 ..< p {
                if (p2 < s && p4 >= s) || (p2 >= s && p4 < s) { m[idx(w1, lastCol, p1, p2, p3, p4)] = true }
            }}}}
        }
        // -> (1, 1, nWindows, window², window²), matching upstream's rearrange target exactly so the
        // golden fixture can be compared without a reshape. Broadcasts over heads and batch.
        return MLXArray(m.map { $0 ? Float(1) : Float(0) },
                        [1, 1, hWindows * wWindows, p * p, p * p])
    }

    /// NHWC in, NHWC out.
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (b, h, w, c) = (input.dim(0), input.dim(1), input.dim(2), input.dim(3))
        let p = window
        var x = input

        // Cyclic shift, negative on the way in. mlx-swift's `roll` takes one shift, so H and W go
        // separately — sequential rolls on independent axes compose to the joint roll upstream does.
        if kind == .sw {
            x = MLX.roll(MLX.roll(x, shift: -(p / 2), axis: 1), shift: -(p / 2), axis: 2)
        }

        let hW = h / p, wW = w / p
        let nW = hW * wW, nP = p * p

        // b (w1 p1) (w2 p2) c -> b (w1 w2) (p1 p2) c
        x = x.reshaped(b, hW, p, wW, p, c)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(b, nW, nP, c)

        let qkv = embedding(x)                              // (b, nW, nP, 3c)

        // ⚠️ THE HEAD SPLIT. Channels are [q heads | k heads | v heads], each head `headDim` wide —
        // upstream reshapes to (3·heads, headDim) and chunks along the HEAD axis, not per-head
        // triples. Reshaping to (heads, 3, headDim) instead would be shape-identical and wrong.
        let heads3 = qkv.reshaped(b, nW, nP, 3 * nHeads, headDim)
            .transposed(3, 0, 1, 2, 4)                      // (3·heads, b, nW, nP, headDim)
        let q = heads3[0 ..< nHeads]
        let k = heads3[nHeads ..< (2 * nHeads)]
        let v = heads3[(2 * nHeads) ..< (3 * nHeads)]

        // (heads, b, nW, nP, nP)
        var sim = MLX.matmul(q, k.transposed(0, 1, 2, 4, 3)) * scale
        sim = sim + relativeBias().reshaped(nHeads, 1, 1, nP, nP)

        if let mask = shiftMask(hWindows: hW, wWindows: wW) {
            // mask is (1, nW, nP, nP); broadcast over heads and batch.
            sim = sim + mask.reshaped(1, 1, nW, nP, nP) * -1e30
        }

        let probs = MLX.softmax(sim, axis: -1)
        let out = MLX.matmul(probs, v)                      // (heads, b, nW, nP, headDim)

        // h b w p c -> b w p (h c)
        let merged = out.transposed(1, 2, 3, 0, 4).reshaped(b, nW, nP, nHeads * headDim)
        var y = linear(merged)

        // b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c
        y = y.reshaped(b, hW, wW, p, p, y.dim(3))
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(b, h, w, y.dim(3))

        if kind == .sw {
            y = MLX.roll(MLX.roll(y, shift: p / 2, axis: 1), shift: p / 2, axis: 2)
        }
        return y
    }
}
