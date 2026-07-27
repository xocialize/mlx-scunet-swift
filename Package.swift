// swift-tools-version: 6.2
import PackageDescription

// mlx-scunet-swift — SCUNet blind real-world denoising (Swin-Conv-UNet).
// Upstream: cszn/SCUNet (Apache-2.0); weights first-party from the cszn/KAIR v1.0 release (MIT).
let package = Package(
    name: "mlx-scunet-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "SCUNetMLXCore", targets: ["SCUNetMLXCore"]),
        .library(name: "MLXSCUNet", targets: ["MLXSCUNet"]),
        .executable(name: "scunet-gate", targets: ["SCUNetGate"]),
        .executable(name: "scunet-validate", targets: ["SCUNetValidate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.39.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(name: "SCUNetMLXCore", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .target(name: "MLXSCUNet", dependencies: [
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            "SCUNetMLXCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "Hub", package: "swift-transformers"),
            .product(name: "MLXProfiling", package: "mlx-profiling"),
        ], swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(name: "SCUNetMLXTests", dependencies: [
            "SCUNetMLXCore", "MLXSCUNet",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
        ], resources: [.copy("Resources/goldens")]),
        .executableTarget(name: "SCUNetValidate", dependencies: [
            "MLXSCUNet",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            .product(name: "MLXEngineTestKit", package: "mlx-engine-swift"),
        ], path: "Sources/Validate", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "SCUNetGate", dependencies: [
            "SCUNetMLXCore", .product(name: "MLX", package: "mlx-swift"),
        ], path: "Sources/Gate", swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
