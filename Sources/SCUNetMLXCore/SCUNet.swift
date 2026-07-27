//
//  SCUNet.swift
//  mlx-scunet-swift / SCUNetMLXCore
//
//  Role: MLX-Swift port of SCUNet — **blind** real-world denoising. Unlike DRUNet it takes no noise
//  level: one forward pass, no dial, no estimate to get wrong.
//
//  Upstream: https://github.com/cszn/SCUNet — **Apache-2.0**. Weights are published first-party by
//            the same author in the `cszn/KAIR` v1.0 release, **MIT**.
//  Paper:    Zhang et al., "Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis".
//            17,946,072 parameters with the released config.
//
//  🔴 The config trap: the constructor upstream **defaults to `config=[2]*7`**, which is not what any
//  released checkpoint uses. Every `main_test_scunet_*.py` passes `[4]*7` explicitly. The default
//  loads with 3 missing / 269 unexpected keys — so it fails loudly, but only if you load strictly.
//  Here `[4]*7` is the Swift default, and the wrong one is not reachable by accident.
//
//  Conventions: NHWC; module keys mirror the upstream state dict exactly.
//

import Foundation
import MLX
import MLXNN

public final class SCUNet: Module, @unchecked Sendable {

    public struct Configuration: Sendable {
        public var inChannels = 3
        /// Blocks per stage, in `[down1, down2, down3, body, up3, up2, up1]` order.
        /// **`[4]*7` — the released checkpoints' shape**, not upstream's `[2]*7` default.
        public var config = [4, 4, 4, 4, 4, 4, 4]
        public var dim = 64
        public var headDim = 32
        public var window = 8
        /// Feeds the build-time `SW → W` downgrade rule in `ConvTransBlock`, halving per stage.
        /// It is **not** the size of the image you will run: inference pads to a multiple of 64 and
        /// works at whatever resolution you hand it.
        public var inputResolution = 256
        public init() {}
    }

    @ModuleInfo(key: "m_head") public var head: [Module]
    @ModuleInfo(key: "m_down1") public var down1: [Module]
    @ModuleInfo(key: "m_down2") public var down2: [Module]
    @ModuleInfo(key: "m_down3") public var down3: [Module]
    @ModuleInfo(key: "m_body") public var body: [ConvTransBlock]
    @ModuleInfo(key: "m_up3") public var up3: [Module]
    @ModuleInfo(key: "m_up2") public var up2: [Module]
    @ModuleInfo(key: "m_up1") public var up1: [Module]
    @ModuleInfo(key: "m_tail") public var tail: [Module]

    /// Three stride-2 stages ⇒ 8×; but upstream pads to **64**, not 8. Kept as upstream has it —
    /// the deepest blocks want at least one full 8-px window at 1/8 scale, and 8 × 8 = 64.
    public static let sizeMultiple = 64

    public init(_ cfg: Configuration = Configuration()) {
        let dim = cfg.dim, hd = cfg.headDim, ws = cfg.window

        // Types alternate W, SW, W, SW… by block index, exactly as upstream's `'W' if not i%2 else 'SW'`.
        func blocks(_ ch: Int, _ n: Int, _ res: Int) -> [Module] {
            (0 ..< n).map { i in
                ConvTransBlock(convDim: ch, transDim: ch, headDim: hd, window: ws,
                               kind: i % 2 == 0 ? .w : .sw, inputResolution: res) as Module
            }
        }
        func downsample(_ inCh: Int, _ outCh: Int) -> Module {
            Conv2d(inputChannels: inCh, outputChannels: outCh,
                   kernelSize: 2, stride: 2, padding: 0, bias: false)
        }
        func upsample(_ inCh: Int, _ outCh: Int) -> Module {
            ConvTransposed2d(inputChannels: inCh, outputChannels: outCh,
                             kernelSize: 2, stride: 2, padding: 0, bias: false)
        }

        // `m_head` / `m_tail` are single-element Sequentials upstream, so their weights live at
        // `m_head.0.weight` — the array keeps that index rather than flattening to `m_head.weight`.
        self._head.wrappedValue = [Conv2d(inputChannels: cfg.inChannels, outputChannels: dim,
                                          kernelSize: 3, stride: 1, padding: 1, bias: false)]
        self._down1.wrappedValue = blocks(dim / 2, cfg.config[0], cfg.inputResolution)
            + [downsample(dim, 2 * dim)]
        self._down2.wrappedValue = blocks(dim, cfg.config[1], cfg.inputResolution / 2)
            + [downsample(2 * dim, 4 * dim)]
        self._down3.wrappedValue = blocks(2 * dim, cfg.config[2], cfg.inputResolution / 4)
            + [downsample(4 * dim, 8 * dim)]
        self._body.wrappedValue = (0 ..< cfg.config[3]).map { i in
            ConvTransBlock(convDim: 4 * dim, transDim: 4 * dim, headDim: hd, window: ws,
                           kind: i % 2 == 0 ? .w : .sw, inputResolution: cfg.inputResolution / 8)
        }
        self._up3.wrappedValue = [upsample(8 * dim, 4 * dim)]
            + blocks(2 * dim, cfg.config[4], cfg.inputResolution / 4)
        self._up2.wrappedValue = [upsample(4 * dim, 2 * dim)]
            + blocks(dim, cfg.config[5], cfg.inputResolution / 2)
        self._up1.wrappedValue = [upsample(2 * dim, dim)]
            + blocks(dim / 2, cfg.config[6], cfg.inputResolution)
        self._tail.wrappedValue = [Conv2d(inputChannels: dim, outputChannels: cfg.inChannels,
                                          kernelSize: 3, stride: 1, padding: 1, bias: false)]
    }

    private func runStage(_ stage: [Module], _ x: MLXArray) -> MLXArray {
        stage.reduce(x) { acc, m in
            if let b = m as? ConvTransBlock { return b(acc) }
            if let c = m as? Conv2d { return c(acc) }
            if let t = m as? ConvTransposed2d { return t(acc) }
            return acc
        }
    }

    /// Forward on an NHWC tensor whose H and W are already multiples of `sizeMultiple`.
    ///
    /// Skips are **additive**, and there is no global residual on the input — the model outputs the
    /// clean image directly, not the noise.
    public func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let x1 = runStage(head, x0)
        let x2 = runStage(down1, x1)
        let x3 = runStage(down2, x2)
        let x4 = runStage(down3, x3)
        var x = body.reduce(x4) { $1($0) }
        x = runStage(up3, x + x4)
        x = runStage(up2, x + x3)
        x = runStage(up1, x + x2)
        return runStage(tail, x + x1)
    }

    /// Denoise an NHWC RGB image in [0,1]. Blind — there is nothing to configure.
    public func denoise(_ image: MLXArray) -> MLXArray {
        let (padded, size) = Self.padToMultiple(image, Self.sizeMultiple)
        let out = self(padded)
        return clip(out[0..., 0 ..< size.0, 0 ..< size.1, 0...], min: 0, max: 1)
    }

    /// **Replication** pad to a multiple of `m`, bottom/right only — upstream `ReplicationPad2d`.
    ///
    /// Note this is replicate, *not* the reflect padding the sibling DRUNet port uses. Reflect would
    /// be shape-identical and quietly wrong at the borders of any image whose size is not already a
    /// multiple of 64 — which is nearly all of them.
    public static func padToMultiple(_ x: MLXArray, _ m: Int) -> (MLXArray, (Int, Int)) {
        let (h, w) = (x.dim(1), x.dim(2))
        let ph = (m - h % m) % m, pw = (m - w % m) % m
        if ph == 0 && pw == 0 { return (x, (h, w)) }
        var out = x
        if ph > 0 {
            let edge = out[0..., (h - 1) ..< h, 0..., 0...]
            out = concatenated([out] + Array(repeating: edge, count: ph), axis: 1)
        }
        if pw > 0 {
            let edge = out[0..., 0..., (w - 1) ..< w, 0...]
            out = concatenated([out] + Array(repeating: edge, count: pw), axis: 2)
        }
        return (out, (h, w))
    }

    /// Loads converted safetensors weights under the strict verifier.
    public func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}
