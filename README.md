# mlx-scunet-swift

**SCUNet** — blind real-world image denoising — ported to MLX-Swift for Apple Silicon, and wrapped
as an MLXEngine `imageRestore` package.

Zhang et al., *Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis*.
Upstream [`cszn/SCUNet`](https://github.com/cszn/SCUNet) (Apache-2.0); weights published
first-party by the same author in the [`cszn/KAIR` v1.0 release](https://github.com/cszn/KAIR/releases/tag/v1.0)
(MIT). **17,946,072 parameters**, 71.8 MB at fp32.

## What makes it worth having

**There is nothing to configure.** Every other restorer in this fleet either bakes a degradation
into the checkpoint (NAFNet, FFTformer, Restormer) or asks you for a noise level
([DRUNet](https://github.com/xocialize/mlx-drunet-swift) takes σ as a model input and exposes it as
a strength dial). SCUNet takes neither: one forward pass on a real photograph whose noise level
nobody has measured.

Two checkpoints, same architecture, different objective:

| variant | objective | use it when |
|---|---|---|
| `.realPSNR` *(default)* | MSE | the output feeds another stage, or is scored against a reference |
| `.realGAN` | adversarial | a human is looking at it — crisper, and it **hallucinates** texture |

## Use

```swift
import SCUNetMLXCore

let model = SCUNet()                       // config=[4]*7 — NOT upstream's [2]*7 default
try model.loadWeights(from: weightsURL)
let clean = model.denoiseTiled(imageNHWC)  // NHWC RGB in [0,1]
```

Or through the engine:

```swift
import MLXSCUNet

engine.register(SCUNetRestorePackage.registration)
let response = try await engine.run(ImageRestoreRequest(image: image))
// response.appliedStrength == nil — this backer is blind, it has no dial
```

Weights: [`mlx-community/SCUNet-color-real-psnr-fp32`](https://huggingface.co/mlx-community/SCUNet-color-real-psnr-fp32)
· [`mlx-community/SCUNet-color-real-gan-fp32`](https://huggingface.co/mlx-community/SCUNet-color-real-gan-fp32).

## Window attention tiles almost for free

SCUNet is a Swin-Conv-UNet: `ConvTransBlock` splits channels between a conv path and a
shifted-window attention path and re-fuses them. The attention is **strictly local** — 8×8 windows,
alternating regular (`W`) and shifted (`SW`) — so tiles are nearly independent. Measured at 512²,
tiled at 256 against the full-frame reference:

| overlap | PSNR | seam / interior gradient |
|---|---|---|
| 0 | 58.32 dB | 1.08× |
| **64** | **71.60 dB** | **1.00×** |

1.00× means a tile boundary is statistically indistinguishable from ordinary image content.
Restormer, whose channel attention reduces over the whole feature map, does not get off this lightly.

Tiling is the production path: full-frame costs 10.04 GB at 1024² and scales linearly in pixels
(~20 GB at 1080p), where tiled at 384 it is flat. **Tile geometry must be 64-aligned** — the forward
pass pads to a multiple of 64 and lays the window grid out from the tile's own origin, so an
unaligned origin shifts the window phase between neighbours and leaves a seam feathering cannot
remove. `denoiseTiled` rounds down to enforce this.

## Parity

Gated against a PyTorch oracle on the CPU stream, fp32, judged on **relative** error
(`swift run -c release scunet-gate --all`). Worst error across the whole ladder: **3.42e-06**.

The gate ladder is shaped by the fact that window attention's two porting errors are *shape-safe* —
they produce a plausible image rather than an error:

- **The QKV head split.** `rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c').chunk(3, dim=0)`
  puts **all q heads, then all k, then all v**, not per-head triples.
- **The SW mask.** Only the *last* window row and column are masked — the wrap-around the cyclic
  roll creates. Omitting it lets opposite edges of the image attend to each other.

So the S1 gate compares the *tables* — the stored relative-position parameters, the gathered
`(heads, 64, 64)` bias, and the mask — at tolerance **0**, before anything consumes them. All five
land at exactly `0.00e+00`, which makes a downstream failure diagnostic instead of diffuse.

Full record, including the three upstream traps and the two that only exist on the Swift side:
[`PORT-STATUS.md`](PORT-STATUS.md).

## ⚠️ Honest positioning

**No primary source reports SCUNet's SIDD or DND** — the authors skipped both deliberately. Whether
this *replaces* or *complements* NAFNet on real sensor noise is unmeasured, not settled. The port is
gated; the ranking claim is not made.

## Licence

Port code MIT ([`LICENSE`](LICENSE)); weights MIT. Upstream `cszn/SCUNet` is **Apache-2.0** and its
notices are retained in [`LICENSE-upstream`](LICENSE-upstream) and [`NOTICE`](NOTICE), as Apache-2.0
§4 requires.
