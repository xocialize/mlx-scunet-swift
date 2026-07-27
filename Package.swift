// swift-tools-version: 6.2
import PackageDescription

// mlx-scunet-swift — SCUNet blind real-world denoising (Swin-Conv-UNet).
// Upstream: cszn/SCUNet (Apache-2.0); weights first-party from the cszn/KAIR v1.0 release (MIT).
let package = Package(
    name: "mlx-scunet-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "SCUNetMLXCore", targets: ["SCUNetMLXCore"]),
        .executable(name: "scunet-gate", targets: ["SCUNetGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        .target(name: "SCUNetMLXCore", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .executableTarget(name: "SCUNetGate", dependencies: [
            "SCUNetMLXCore", .product(name: "MLX", package: "mlx-swift"),
        ], path: "Sources/Gate", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
