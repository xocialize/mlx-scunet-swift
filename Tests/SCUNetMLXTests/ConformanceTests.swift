// ConformanceTests.swift — SCUNet through the engine's offline gates (no MLX kernels run).
//
// The tests that matter most here are the ones about what this package DOESN'T have. SCUNet is the
// blind backer: no noise level, no strength dial. The sibling DRUNet package is the one with the
// dial, and the contract lets a planner tell them apart only if this one keeps saying no.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
import SCUNetMLXCore
@testable import MLXSCUNet

final class ConformanceTests: XCTestCase {

    // MARK: - MAT

    func testMATGate() {
        let report = MaterializationConformance.check(freshConfiguration: SCUNetConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesDeclaredForEveryVariant() {
        var repos = Set<String>()
        for variant in SCUNetVariant.allCases {
            let sources = SCUNetConfiguration(variant: variant).weightSources
            XCTAssertEqual(sources.count, 1, "\(variant)")
            XCTAssertEqual(sources[0].repo, variant.repo)
            XCTAssertEqual(sources[0].matching, ["model.safetensors"], "\(variant)")
            repos.insert(sources[0].repo)
        }
        // The two checkpoints share an architecture and a key set, so nothing about *loading* would
        // complain if both variants pointed at one repo — it would just silently serve the fidelity
        // model to a caller who asked for the perceptual one.
        XCTAssertEqual(repos.count, SCUNetVariant.allCases.count)
    }

    func testExplicitWeightsURLSuppressesMaterialization() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("scunet-\(UUID().uuidString).safetensors")
        try Data([0x00]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertTrue(SCUNetConfiguration(weightsURL: tmp).missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(
            SCUNetConfiguration(weightsURL: tmp.appendingPathExtension("nope"))
                .missingWeightSources(storeRoot: nil).count, 1)
    }

    // MARK: - CAN

    func testCANGatePreCancelledRun() async {
        let package = SCUNetRestorePackage(configuration: SCUNetConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageRestoreRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        let manifest = SCUNetRestorePackage.manifest
        XCTAssertTrue(CancellationConformance.longRunImplied(by: manifest))
        // run() has a REAL iterative seam — the tile loop — and checkpoints once per tile via the
        // core's onTile hook, reporting RunProgress on the same unit.
        let report = CancellationConformance.checkCadence(
            manifest: manifest,
            posture: .cadence([
                .init(phase: .postprocess, unit: .chunk, reportsRunProgress: true),
                .init(phase: .encode, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Manifest

    func testManifestSurfacesAndLicence() {
        let m = SCUNetRestorePackage.manifest
        XCTAssertEqual(m.capabilities, [.imageRestore], "no new capability — same request shape")
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].name, "scunet-denoise")
        // Port code MIT (this repo's LICENSE), weights MIT (cszn/KAIR v1.0). Upstream is
        // Apache-2.0 and its notices are retained in LICENSE-upstream + NOTICE — the declaration
        // must track the repo's own LICENSE file, not upstream's, or the engine's licence gate is
        // reasoning about a document that does not ship here.
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.license.weightLicense, .mit)
        XCTAssertEqual(m.provenance.sourceRepo, "cszn/SCUNet")
    }

    func testFootprintIsSplitAndFlatInResolution() {
        guard let fp = SCUNetRestorePackage.manifest.requirements.footprints
            .first(where: { $0.quant == .fp32 }) else { return XCTFail("no fp32 footprint") }
        // 17.9 M params @ fp32 = 71.8 MB; the floor must cover it without absorbing the activation.
        XCTAssertGreaterThan(fp.residentBytes, 71_800_000)
        XCTAssertLessThan(fp.residentBytes, 500_000_000)
        XCTAssertGreaterThan(fp.peakActivationBytes, fp.residentBytes)
        // Tiled internally, so the peak is one-tile-sized and does NOT track resolution. Measured
        // 4.38 GB at 1080p; untiled at 1024² already costs 10.04 GB and scales linearly in pixels,
        // so ~20 GB at 1080p. The 8 GB bound sits between the two: if this starts failing, someone
        // has removed the tiling, not merely nudged a number.
        XCTAssertLessThan(fp.peakActivationBytes, 8_000_000_000)
    }

    func testQuantConfiguredMatchesADeclaredFootprint() {
        let declared = Set(SCUNetRestorePackage.manifest.requirements.footprints.map(\.quant))
        for v in SCUNetVariant.allCases {
            XCTAssertTrue(declared.contains(SCUNetConfiguration(variant: v).quant), "\(v)")
        }
    }

    // MARK: - Blindness

    /// The point of this package. SCUNet has no noise-level input, so advertising `strength` would
    /// promise a control that cannot exist — and a planner choosing between this and DRUNet reads
    /// exactly this descriptor to tell them apart.
    func testDescriptorDoesNotAdvertiseStrength() {
        let d = SCUNetRestorePackage.manifest.surfaces[0]
        let names = d.parameters.map(\.name)
        XCTAssertTrue(names.contains("image"))
        XCTAssertFalse(names.contains("strength"),
                       "SCUNet is blind — it has no dial to expose: \(names)")
    }

    func testVariantsResolveToDistinctRepos() {
        XCTAssertEqual(Set(SCUNetVariant.allCases.map(\.repo)).count, SCUNetVariant.allCases.count)
    }

    // MARK: - Tile geometry

    /// 64-alignment is a correctness property, not a tuning knob: `forward` pads to a multiple of 64
    /// and lays the window grid out from the tile's own origin, so an unaligned origin shifts the
    /// window phase between neighbouring tiles and leaves a seam feathering cannot remove.
    func testDefaultTileGeometryIs64Aligned() {
        let c = SCUNetConfiguration()
        XCTAssertEqual(c.tile % SCUNet.sizeMultiple, 0)
        XCTAssertEqual(c.overlap % SCUNet.sizeMultiple, 0)
        XCTAssertGreaterThan(c.tile, 2 * c.overlap, "step must be positive")
    }

    func testSizeMultipleMatchesUpstreamPadding() {
        // Upstream pads to 64, not to the 8 the three stride-2 stages would imply — the deepest
        // blocks want a full 8-px attention window at 1/8 scale.
        XCTAssertEqual(SCUNet.sizeMultiple, 64)
    }
}
