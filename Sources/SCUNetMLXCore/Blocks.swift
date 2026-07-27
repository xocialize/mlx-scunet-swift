//
//  Blocks.swift
//  mlx-scunet-swift / SCUNetMLXCore
//
//  `Block` — a Swin transformer block (LN → WMSA → residual, LN → MLP → residual).
//  `ConvTransBlock` — SCUNet's contribution: a channel-wise split into a conv path and a
//  transformer path, run in parallel and re-fused.
//
//  Both mirror `models/network_scunet.py` key-for-key; `drop_path` is `Identity` at inference so it
//  contributes no parameters and is omitted here.
//

import Foundation
import MLX
import MLXNN

/// Swin transformer block. NHWC in, NHWC out.
public final class Block: Module, UnaryLayer, @unchecked Sendable {

    @ModuleInfo(key: "ln1") public var ln1: LayerNorm
    @ModuleInfo(key: "msa") public var msa: WMSA
    @ModuleInfo(key: "ln2") public var ln2: LayerNorm
    /// `[Linear, GELU, Linear]` — index 1 is the parameter-free activation, held by an `Identity`
    /// so that `mlp.0` / `mlp.2` line up with the upstream `nn.Sequential` key paths.
    @ModuleInfo(key: "mlp") public var mlp: [Module]

    public init(inputDim: Int, outputDim: Int, headDim: Int, window: Int, kind: WMSA.Kind) {
        self._ln1.wrappedValue = LayerNorm(dimensions: inputDim)
        self._msa.wrappedValue = WMSA(inputDim: inputDim, outputDim: inputDim,
                                      headDim: headDim, window: window, kind: kind)
        self._ln2.wrappedValue = LayerNorm(dimensions: inputDim)
        self._mlp.wrappedValue = [
            Linear(inputDim, 4 * inputDim, bias: true),
            Identity(),                                   // the GELU
            Linear(4 * inputDim, outputDim, bias: true),
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x + msa(ln1(x))
        guard let l0 = mlp[0] as? Linear, let l2 = mlp[2] as? Linear else { return y }
        // torch's `nn.GELU()` default is the **exact** erf form, not the tanh approximation.
        y = y + l2(gelu(l0(ln2(y))))
        return y
    }
}

/// Conv path ‖ transformer path, split by channel and re-fused. NHWC in, NHWC out.
///
/// Upstream works in NCHW and splits `dim=1`; here the channel axis is last, so the split and the
/// concat move to `axis: -1` and the two `Rearrange`s around the transformer path vanish — `WMSA`
/// already wants NHWC.
public final class ConvTransBlock: Module, UnaryLayer, @unchecked Sendable {

    @ModuleInfo(key: "conv1_1") public var conv1_1: Conv2d
    @ModuleInfo(key: "conv1_2") public var conv1_2: Conv2d
    /// `[Conv2d, ReLU, Conv2d]`, both convs `bias=false`; index 1 is the ReLU placeholder.
    @ModuleInfo(key: "conv_block") public var convBlock: [Module]
    @ModuleInfo(key: "trans_block") public var transBlock: Block

    private let convDim: Int
    private let transDim: Int

    /// - Parameter kind: the *requested* attention type. It is **downgraded to `.w` here, at
    ///   construction**, when `inputResolution <= window` — exactly as upstream does. That decision
    ///   belongs to the model's build-time resolution schedule, not to the image passed in at
    ///   runtime, so a port that recomputed it from the actual input size would give some blocks the
    ///   wrong attention. (For the released config every resolution is ≥ 32 > 8, so no block is
    ///   actually downgraded — but the rule has to be encoded, not assumed inert.)
    public init(convDim: Int, transDim: Int, headDim: Int, window: Int,
                kind: WMSA.Kind, inputResolution: Int) {
        self.convDim = convDim
        self.transDim = transDim
        let effective: WMSA.Kind = inputResolution <= window ? .w : kind

        self._transBlock.wrappedValue = Block(inputDim: transDim, outputDim: transDim,
                                              headDim: headDim, window: window, kind: effective)
        let total = convDim + transDim
        self._conv1_1.wrappedValue = Conv2d(inputChannels: total, outputChannels: total,
                                            kernelSize: 1, stride: 1, padding: 0, bias: true)
        self._conv1_2.wrappedValue = Conv2d(inputChannels: total, outputChannels: total,
                                            kernelSize: 1, stride: 1, padding: 0, bias: true)
        self._convBlock.wrappedValue = [
            Conv2d(inputChannels: convDim, outputChannels: convDim,
                   kernelSize: 3, stride: 1, padding: 1, bias: false),
            Identity(),                                   // the ReLU
            Conv2d(inputChannels: convDim, outputChannels: convDim,
                   kernelSize: 3, stride: 1, padding: 1, bias: false),
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mixed = conv1_1(x)
        var convX = mixed[0..., 0..., 0..., 0 ..< convDim]
        let transX = mixed[0..., 0..., 0..., convDim ..< (convDim + transDim)]

        guard let c0 = convBlock[0] as? Conv2d, let c2 = convBlock[2] as? Conv2d else { return x }
        convX = c2(relu(c0(convX))) + convX

        let out = transBlock(transX)
        return x + conv1_2(concatenated([convX, out], axis: -1))
    }
}
