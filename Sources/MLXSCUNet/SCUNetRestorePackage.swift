import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
import Hub
import SCUNetMLXCore

/// Errors at the SCUNet package boundary.
public enum SCUNetPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsMissing(String)
}

/// An MLXEngine `imageRestore` package over **SCUNet** — **blind** real-world denoising.
///
/// The distinguishing property is the absence of a knob. NAFNet, FFTformer and Restormer bake a
/// degradation into the checkpoint; DRUNet takes the noise level as a model input and exposes it as
/// a strength dial. SCUNet takes neither: one forward pass, no σ to estimate, nothing for a caller
/// to get wrong. That is why `supportsStrength` is **false** here — declaring a dial this model does
/// not have would be a lie the contract is specifically built to prevent.
///
/// It is also the first **window-attention** model in the fleet (Swin-Conv-UNet), which has a
/// practical consequence: its mixing is strictly local, so tiling is nearly free. Measured at 512²,
/// tiled-vs-full-frame is 71.60 dB with a seam/interior gradient ratio of 1.00× — where Restormer's
/// spatially *global* attention made the same comparison a real trade.
///
/// ⚠️ Positioning is still open (**V4** in the port queue). No primary source reports SCUNet's SIDD
/// or DND — the authors skipped it deliberately — so whether this **replaces or complements**
/// NAFNet is unmeasured. It needs corpus **C5** (ISO brackets + dark frames). The port is sound; the
/// ranking claim is not made.
@InferenceActor
public final class SCUNetRestorePackage: ModelPackage {
    public typealias Configuration = SCUNetConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Weights: MIT — published first-party by the author in the cszn/KAIR v1.0 release.
            //
            // Port code: MIT, matching this repository's LICENSE. Upstream cszn/SCUNet is
            // **Apache-2.0** (no non-commercial text anywhere), and Apache-2.0 §4 permits
            // distributing a derivative under terms of your choice provided the upstream notices are
            // retained — which `LICENSE-upstream` and `NOTICE` do. Upstream's own contribution stays
            // Apache-2.0 regardless of what this field says; the field describes the port.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "cszn/SCUNet", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14) — ✅ MEASURED through the REAL `MLXServeEngine` via
                // `MLXEngineTestKit.ValidationHarness` (`swift run scunet-validate`), process
                // `phys_footprint`, floor read post-load/pre-run:
                //
                //   [scunet-real-psnr] SPLIT floor=0.10GB peak=4.48GB act=4.38GB retain=0.56GB
                //                      engine=0.18GB reserve=2.00GB load=0.0s run=4.9s  @1920x1080
                //
                // Declared with margin: resident 180 MB (floor 103.5 MB), activation 5.0 GB
                // (measured 4.38 GB).
                //
                // ⚠️ The gate's `--bench` read only **1.33 GB** for the same tile size — 3.3× under.
                // Same direction and same cause as every prior port: `--bench` calls the core
                // directly and reads after the fact, so it misses both the engine's decode/encode
                // buffers and the transient peak that 150 ms sampling catches. The harness number is
                // the admission basis; `--bench` is only good for *comparing* tile sizes.
                //
                // Tiling is internal, so this peak is one-tile-sized and flat in resolution. Untiled
                // for comparison: 10.04 GB phys at 1024², scaling linearly in pixels.
                footprints: [
                    QuantFootprint(quant: .fp32,
                                   residentBytes: 180_000_000,
                                   peakActivationBytes: 5_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageRestoreContract.descriptor(
                    name: "scunet-denoise",
                    summary: "SCUNet blind real-world denoising: no noise level to supply, one "
                        + "forward pass. Fidelity (real-psnr) or perceptual (real-gan) by variant.",
                    supportsStrength: false
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: SCUNet?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }

        let url: URL
        if let explicit = configuration.weightsURL {
            guard FileManager.default.fileExists(atPath: explicit.path) else {
                throw SCUNetPackageError.weightsMissing(explicit.path)
            }
            url = explicit
        } else {
            // Since contract 1.24 the engine materializes declared `weightSources` before load().
            // This snapshot is the defensive path — it finds everything already present in the
            // normal flow, and still works for a standalone (engine-less) consumer of the package.
            let repo = configuration.variant.repo
            let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
            let dir = try await hub.snapshot(from: Hub.Repo(id: repo),
                                             matching: ["model.safetensors"]) { progress, speed in
                WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
            }
            url = dir.appendingPathComponent("model.safetensors")
        }

        // Both variants share one architecture and one key set (540 tensors, 17,946,072 params), so
        // unlike the sibling ports there is no per-variant model configuration to get wrong — the
        // variant only selects which weights to fetch.
        let net = SCUNet()
        try net.loadWeights(from: url)
        model = net
    }

    public func unload() async {
        model = nil
        MLX.Memory.clearCache()   // drop the retained MLX pool so eviction frees RSS, not just refs
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the FIRST act of run(), before notLoaded validation.
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .imageRestore,
              let req = request as? ImageRestoreRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let pb = try Self.decodeToPixelBuffer(req.image)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let x = rgbNHWC(from: ensureBGRA(pb), width: w, height: h) else {
            throw SCUNetPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h))")
        }

        // Pre-forward checkpoint: last seam before committing to the eval.
        try Task.checkCancellation()
        let prof = MLXProfiler.shared
        prof.beginRun("scunet imageRestore \(configuration.variant.rawValue) \(w)x\(h)")
        // TILED is the production path: full-frame costs 10.04 GB phys at 1024² and scales linearly
        // in pixels, so 1080p would be ~20 GB. Tiled at 384 it is 1.33 GB, flat in resolution.
        // Tile geometry is 64-aligned — `forward` pads to a multiple of 64 and lays the window grid
        // out from the tile's own origin, so a misaligned origin shifts the window phase and leaves
        // a seam that feathering cannot remove.
        // The tile loop is a REAL iterative seam, so cancellation and progress are checkpointed per
        // tile. The CancellationError propagates unchanged so the engine can tell user-cancel from
        // a package error.
        let restoredNHWC = try prof.region("denoise", "forward") {
            try model.denoiseTiled(x, tile: configuration.tile, overlap: configuration.overlap) { done, total in
                try Task.checkCancellation()
                RunProgress.report(RunPhaseReport(phase: .postprocess, step: done + 1,
                                                  totalSteps: total))
            }
        }
        let outPB = pixelBuffer(fromRGBNHWC: restoredNHWC, width: w, height: h)
        prof.endRun(denominators: ["image": 1])
        guard let outPB else { throw SCUNetPackageError.imageEncodeFailed }

        // Post-forward checkpoint: between materialization and output encode.
        try Task.checkCancellation()
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw SCUNetPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw SCUNetPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        // Blind model: there is no dial, so `appliedStrength` is nil. A caller comparing this to
        // DRUNet can read that directly — nil means "this backer has no strength", not "zero".
        return ImageRestoreResponse(image: outImage, appliedStrength: nil)
    }

    // MARK: - Image codec
    //
    // Same shape as the sibling image packages. Duplicated rather than shared: each `-swift`
    // package stays independently buildable, and the codec is the package's own boundary.

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw SCUNetPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw SCUNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw SCUNetPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no decode.
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw SCUNetPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw SCUNetPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw SCUNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw SCUNetPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes.
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension SCUNetRestorePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(SCUNetRestorePackage.self)
    }
}
