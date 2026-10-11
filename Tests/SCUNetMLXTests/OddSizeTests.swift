// OddSizeTests.swift — the tiled path at any H × W, not only multiples of 64.
//
// `denoise` replication-pads to a multiple of 64 and crops back, and `denoiseTiled` runs every
// tile, including the odd-sized edge tiles, through `denoise`.
//
// Offline and CPU-pinned: randomly initialised weights, no download, no Metal. Checked while fixing
// the same class of crash in mlx-cidnet-swift (AB-T-0223), where an unpadded UNet's concatenate hit
// mlx-swift's `fatalError` and took the host process down on a 570×320 image.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import MLXToolKit
import UniformTypeIdentifiers
import XCTest
@testable import SCUNetMLXCore
@testable import MLXSCUNet

final class OddSizeTests: XCTestCase {

    /// Tiled in both axes, with odd edge tiles: 133 × 141 at tile 128 / overlap 0, so the last
    /// tiles are 5 and 13 pixels wide.
    func testTiledOddSizeKeepsExtent() async {
        await withMLXCPU {
            let net = SCUNet()
            let x = MLXRandom.uniform(low: 0, high: 1, [1, 141, 133, 3])
            var tiles = 0
            let out = net.denoiseTiled(x, tile: 128, overlap: 0) { _, _ in tiles += 1 }
            eval(out)
            XCTAssertEqual(tiles, 4)
            XCTAssertEqual(out.shape, [1, 141, 133, 3])
            XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self), "NaN in the output")
        }
    }

    /// Through run(), tiled in both axes with odd edge tiles. The live repro's 570×321 at the default
    /// tile 384 / overlap 64 passes too, but takes ~90 s on the CPU (AB-T-0223 receipt), so the suite
    /// runs it small.
    func testPackageRestoresOddSizePNG() async throws {
        let (width, height) = (133, 141)
        try await withMLXCPU {
            let weights = try randomWeights(SCUNet())
            defer { try? FileManager.default.removeItem(at: weights) }
            let package = SCUNetRestorePackage(configuration: SCUNetConfiguration(
                tile: 128, overlap: 0, weightsURL: weights))
            try await package.load()
            let response = try await package.run(
                ImageRestoreRequest(image: try gradientPNG(width: width, height: height)))
            let restored = try XCTUnwrap(response as? ImageRestoreResponse)
            try assertPNG(restored.image, width: width, height: height)
        }
    }

    /// Runs `body` with MLX pinned to the CPU. `withDefaultDevice` alone scopes only the Swift-side
    /// default stream; mlx core internals still use the global C++ default device, so that is set
    /// too, after latching the Swift default so the process isn't left on the CPU.
    private func withMLXCPU<R>(_ body: () async throws -> R) async rethrows -> R {
        _ = Device.defaultDevice()
        Device.setDefault(device: Device(.cpu))
        defer { Device.setDefault(device: Device(.gpu)) }
        return try await Device.withDefaultDevice(Device(.cpu), body)
    }

    /// Randomly initialised weights, saved where `weightsURL` can load them.
    private func randomWeights(_ model: Module) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "oddsize-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: url)
        return url
    }

    /// The package's response image must come back at the request's W × H, as a decodable PNG.
    private func assertPNG(_ image: Image, width: Int, height: Int,
                           file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(image.format, .png, file: file, line: line)
        XCTAssertEqual(image.width, width, file: file, line: line)
        XCTAssertEqual(image.height, height, file: file, line: line)
        let decoded = try XCTUnwrap(CGImageSourceCreateWithData(image.data as CFData, nil)
            .flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }, file: file, line: line)
        XCTAssertEqual(decoded.width, width, file: file, line: line)
        XCTAssertEqual(decoded.height, height, file: file, line: line)
    }

    /// A smooth gradient PNG.
    private func gradientPNG(width: Int, height: Int) throws -> Image {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let i = (y * width + x) * 4
                pixels[i + 0] = UInt8(255 * x / width)
                pixels[i + 1] = UInt8(255 * y / height)
                pixels[i + 2] = 128
            }
        }
        let cg: CGImage = try pixels.withUnsafeMutableBytes { buffer in
            let ctx = try XCTUnwrap(CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
            return try XCTUnwrap(ctx.makeImage())
        }
        let out = NSMutableData()
        let dest = try XCTUnwrap(CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return Image(format: .png, data: out as Data, width: width, height: height)
    }
}
