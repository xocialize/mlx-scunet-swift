//
//  main.swift — SCUNet parity gates against the PyTorch oracle.
//
//  Gate ladder, in the order a failure is cheapest to diagnose:
//
//    S0  key contract           — shapes/names, strict update
//    S1  attention internals    — stored bias table, gathered bias, SW mask
//    S2  WMSA end-to-end        — W and SW (the QKV head split lands here)
//    S3  Block / ConvTransBlock — residual wiring, channel split, GELU form
//    S4  resamplers + head/tail — the transposed-conv weight layout
//    S5  full model             — 64 / 128 (no pad) and 100 (replication pad + crop)
//
//  S1 exists because the two failure modes of window attention are shape-safe: a wrong relative-bias
//  index or a missing SW mask produces a plausible image, not an error. Gating the *tables* — before
//  they are consumed — turns a diffuse "the output is slightly off" into a named cause.
//

import Foundation
import MLX
import MLXNN
import SCUNetMLXCore

private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()
func fail(_ m: String) -> Never { _ = _unbuffered; print("❌ \(m)"); exit(1) }

func loadedModel(_ p: String) -> SCUNet {
    let m = SCUNet()
    do { try m.loadWeights(from: URL(fileURLWithPath: p)) } catch { fail("weight load: \(error)") }
    return m
}
func g(_ d: String, _ n: String) -> MLXArray {
    do { return try loadNPY("\(d)/\(n).npy") } catch { fail("golden \(n): \(error)") }
}

/// The two `m_down1` blocks the goldens were taken from: index 0 is `W`, index 1 is `SW`.
func downBlocks(_ m: SCUNet) -> (ConvTransBlock, ConvTransBlock) {
    guard let w = m.down1[0] as? ConvTransBlock, let sw = m.down1[1] as? ConvTransBlock else {
        fail("m_down1 layout unexpected")
    }
    return (w, sw)
}

// MARK: - S0

func gateS0(_ w: String) {
    _ = _unbuffered
    print("=== S0 · key contract ===\n")
    let model = SCUNet()
    var keys: [String: [Int]] = [:]; var total = 0
    for (k, v) in model.parameters().flattened() { keys[k] = v.shape; total += v.size }
    print("Swift : \(keys.count) tensors, \(total) params")
    guard let loaded = try? MLX.loadArrays(url: URL(fileURLWithPath: w)) else { fail("load \(w)") }
    print("Ckpt  : \(loaded.count) tensors, \(loaded.values.reduce(0) { $0 + $1.size }) params\n")
    let sk = Set(keys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted(), unused = ck.subtracting(sk).sorted()
    if !missing.isEmpty { print("MISSING (\(missing.count)):"); missing.prefix(10).forEach { print("   \($0) \(keys[$0]!)") } }
    if !unused.isEmpty { print("UNUSED (\(unused.count)):"); unused.prefix(10).forEach { print("   \($0) \(loaded[$0]!.shape)") } }
    var mis: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) where keys[k]! != loaded[k]!.shape { mis.append((k, keys[k]!, loaded[k]!.shape)) }
    if !mis.isEmpty {
        print("SHAPE MISMATCH (\(mis.count)):")
        for (k, a, b) in mis.prefix(10) { print("   \(k)\n     swift \(a) vs ckpt \(b)") }
    }
    guard missing.isEmpty, unused.isEmpty, mis.isEmpty else { fail("S0 FAILED") }
    do { try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all) }
    catch { fail("S0 FAILED at update: \(error)") }
    // 17,946,072 is the Stage-0 figure for `config=[4]*7, dim=64`. The upstream constructor DEFAULT
    // (`[2]*7`) gives 9,662,892 — assert the count so a config regression cannot pass silently on a
    // checkpoint that happens to load.
    guard total == 17_946_072 else { fail("S0 FAILED — expected 17,946,072 params, got \(total)") }
    print("✅ S0 PASSED — \(keys.count) tensors, \(total) params, strict update clean.")
}

// MARK: - S1 · attention internals

func gateS1(_ d: String, _ w: String) -> Bool {
    print("=== S1 · attention internals ===\n")
    let r = GateReport("S1")
    let m = loadedModel(w)
    let (bw, bsw) = downBlocks(m)

    for (tag, blk) in [("w", bw), ("sw", bsw)] {
        let msa = blk.transBlock.msa
        // The stored table. Upstream ALLOCATES it as ((2w-1)², heads) and then re-assigns the
        // parameter through .view(2w-1,2w-1,heads).transpose(1,2).transpose(0,1) — so what the
        // checkpoint carries is (heads, 2w-1, 2w-1). Reading the declaration instead of the
        // constructor gives a transposed table that still loads.
        r.check("\(tag) relparams", msa.relativePositionParams, g(d, "wmsa_\(tag)_relparams"), tol: 0)
        // The gathered (heads, 64, 64) bias. Wrong index arithmetic here is a permutation — same
        // shape, same value range, plausible output.
        r.check("\(tag) relbias", msa.relativeBias(), g(d, "wmsa_\(tag)_relbias"), tol: 0)
    }
    // The SW mask: exact, and only the last window row/column is non-zero.
    guard let mask = bsw.transBlock.msa.shiftMask(hWindows: 4, wWindows: 4) else {
        fail("SW block produced no mask — the W/SW type was lost at construction")
    }
    r.check("sw mask", mask, g(d, "wmsa_sw_mask"), tol: 0)
    return r.summarize()
}

// MARK: - S2 · WMSA end to end

func gateS2(_ d: String, _ w: String) -> Bool {
    print("=== S2 · WMSA ===\n")
    let r = GateReport("S2")
    let m = loadedModel(w)
    let (bw, bsw) = downBlocks(m)
    // NHWC in, NHWC out — no transpose. This is where a wrong QKV head split shows up: upstream's
    // chunk(3, dim=0) over `(3·heads, headDim)` means [all q][all k][all v], not per-head triples.
    r.check("wmsa W", bw.transBlock.msa(g(d, "wmsa_w_in")), g(d, "wmsa_w_out"), tol: 2e-6)
    r.check("wmsa SW", bsw.transBlock.msa(g(d, "wmsa_sw_in")), g(d, "wmsa_sw_out"), tol: 2e-6)
    return r.summarize()
}

// MARK: - S3 · blocks

func gateS3(_ d: String, _ w: String) -> Bool {
    print("=== S3 · Block / ConvTransBlock ===\n")
    let r = GateReport("S3")
    let m = loadedModel(w)
    let (bw, bsw) = downBlocks(m)

    let blockIn = g(d, "block_in")                                   // NHWC
    r.check("block W", bw.transBlock(blockIn), g(d, "block_w_out"), tol: 2e-6)
    r.check("block SW", bsw.transBlock(blockIn), g(d, "block_sw_out"), tol: 2e-6)

    let ctbIn = toNHWC(g(d, "ctb_in"))                               // NCHW golden
    r.check("convtrans W", toNCHW(bw(ctbIn)), g(d, "ctb_w_out"), tol: 2e-6)
    r.check("convtrans SW", toNCHW(bsw(ctbIn)), g(d, "ctb_sw_out"), tol: 2e-6)
    return r.summarize()
}

// MARK: - S4 · resamplers, head, tail

func gateS4(_ d: String, _ w: String) -> Bool {
    print("=== S4 · resamplers + head/tail ===\n")
    let r = GateReport("S4")
    let m = loadedModel(w)
    guard let down = m.down1[4] as? Conv2d,
          let up = m.up3[0] as? ConvTransposed2d,
          let head = m.head[0] as? Conv2d,
          let tail = m.tail[0] as? Conv2d else { fail("stage layout unexpected") }

    r.check("strideconv_down", toNCHW(down(toNHWC(g(d, "down_in")))), g(d, "down_out"), tol: 2e-6)
    // The transposed conv is the one op needing a DIFFERENT weight transpose from every other conv —
    // PyTorch stores it (I,O,k,k). SCUNet has exactly three of them, and `m_down3.4` (512 out) and
    // `m_up3.0` (512 in) are near-mirror shapes, so a swapped layout loads clean. This gate is what
    // catches that, and it lands at exactly 0 when correct.
    r.check("convtranspose_up", toNCHW(up(toNHWC(g(d, "up_in")))), g(d, "up_out"), tol: 2e-6)
    r.check("head", toNCHW(head(toNHWC(g(d, "head_in")))), g(d, "head_out"), tol: 2e-6)
    r.check("tail", toNCHW(tail(toNHWC(g(d, "tail_in")))), g(d, "tail_out"), tol: 2e-6)
    return r.summarize()
}

// MARK: - S5 · full model

func gateS5(_ d: String, _ w: String) -> Bool {
    print("=== S5 · full model ===\n")
    let r = GateReport("S5")
    let m = loadedModel(w)
    // 64 and 128 are already multiples of 64, so they bypass the pad entirely. 100 is the one that
    // matters: it exercises ReplicationPad2d(0, 28, 0, 28) and the crop back. Reflect padding —
    // which the sibling DRUNet port uses — would be shape-identical and wrong only at the borders,
    // so 64/128 alone would not catch it.
    for size in [64, 128, 100] {
        let x = toNHWC(g(d, "full_\(size)_in"))
        let (padded, orig) = SCUNet.padToMultiple(x, SCUNet.sizeMultiple)
        let out = m(padded)[0..., 0 ..< orig.0, 0 ..< orig.1, 0...]
        // No clip — the goldens are the raw forward, and clipping is a presentation decision made
        // in `denoise(_:)`, not part of the model.
        r.check("full \(size)²", toNCHW(out), g(d, "full_\(size)_out"), tol: 2e-5)
    }
    return r.summarize()
}

// MARK: - driver

let args = CommandLine.arguments
let goldens = args.count > 1 ? args[1] : "oracle/goldens"
let weights = args.count > 2 ? args[2] : "oracle/converted/scunet_color_real_psnr/model.safetensors"

MLX.Device.setDefault(device: .cpu)   // parity gates run on the CPU stream

gateS0(weights)
print("")
var allGreen = true
for gate in [gateS1, gateS2, gateS3, gateS4, gateS5] {
    allGreen = gate(goldens, weights) && allGreen
    print("")
}
if !allGreen { fail("one or more gates FAILED") }
print("✅ ALL GATES PASSED")
