import Foundation
import MLXToolKit
import SCUNetMLXCore

/// A SCUNet checkpoint. Both are **blind real-world** denoisers over the identical architecture —
/// they differ only in training objective, so the choice is fidelity vs. appearance, not strength.
public enum SCUNetVariant: String, Codable, Sendable, CaseIterable {
    /// `scunet_color_real_psnr.pth` — MSE-trained. Conservative: it will leave residual noise rather
    /// than invent detail. The right default when the output feeds another stage.
    case realPSNR = "real-psnr"
    /// `scunet_color_real_gan.pth` — adversarially trained. Crisper and more plausible-looking, and
    /// it *hallucinates* texture where the fidelity model stays smooth. Upstream's own demos favour
    /// it for viewing; it is the wrong pick for anything measured against a reference.
    case realGAN = "real-gan"

    public var repo: String {
        switch self {
        case .realPSNR: return "mlx-community/SCUNet-color-real-psnr-fp32"
        case .realGAN: return "mlx-community/SCUNet-color-real-gan-fp32"
        }
    }

    /// 17.9 M params at fp32 = 71.8 MB — the smallest resident footprint in the image fleet. There
    /// is nothing to gain by lowering the dtype, and denoising is precision-sensitive at exactly the
    /// low-amplitude end where the work happens.
    public var quant: Quant { .fp32 }
}

/// Init-time configuration for `SCUNetRestorePackage` (C9).
public struct SCUNetConfiguration: PackageConfiguration, ModelStorable {
    public var variant: SCUNetVariant

    /// Tile extent for the internal tiled path. Rounded down to a multiple of **64** by the core.
    ///
    /// 384 measured as the knee: at 1920×1080 the whole-frame path costs ~20 GB (extrapolated from
    /// 10.04 GB at 1024²), tile 512 costs 3.60 GB in 3.4 s, tile 384 costs **1.33 GB in 4.9 s**, and
    /// tile 256 costs 0.74 GB but 9.1 s. 384 buys a 2.7× memory saving for 1.4× the time.
    public var tile: Int

    /// Context pixels per tile side, discarded into the feathered blend. Also 64-aligned.
    ///
    /// 64 — exactly one window-grid stride — measured at 71.60 dB against the full-frame reference
    /// with a seam/interior gradient ratio of **1.00×**, versus 58.32 dB and 1.08× at overlap 0.
    public var overlap: Int

    public var modelsRootDirectory: URL?
    public var weightsURL: URL?

    public init(variant: SCUNetVariant = .realPSNR,
                tile: Int = 384,
                overlap: Int = 64,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.variant = variant
        self.tile = tile
        self.overlap = overlap
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }

    private enum CodingKeys: String, CodingKey {
        case variant, tile, overlap
    }
}

extension SCUNetConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

extension SCUNetConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: variant.repo, revision: nil,
                      matching: ["model.safetensors"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}
