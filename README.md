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

## Measured on real sensor noise

**No primary source reports SCUNet's SIDD or DND** — the authors skipped both deliberately — so we
measured it ourselves on [NIND](https://commons.wikimedia.org/wiki/Category:Natural_Image_Noise_Dataset)
(CC0): 5 scenes × 4 ISOs on a locked-off camera with a compensating shutter, 768² centre crops, PSNR
against the ISO-100 reference. Pairs were verified pixel-aligned and brightness-matched first, so
only noise differs.

| model | ISO 1600 | ISO 6400 | ISO 25600 |
|---|---|---|---|
| *untouched input* | *34.87* | *29.90* | *23.91* |
| **SCUNet real-psnr** | 36.39 (+1.52) | **34.95 (+5.06)** | **32.28 (+8.38)** |
| Restormer realDenoise | 36.38 (+1.51) | 34.78 (+4.89) | 31.58 (+7.67) |
| SCUNet real-gan | 34.89 (+0.02) | 33.60 (+3.70) | 31.40 (+7.49) |
| **NAFNet-SIDD-width64** | 33.39 (−1.48) 🔴 | 31.99 (+2.09) | 29.38 (+5.48) |
| DRUNet, best σ per row | **37.36 (+2.48)** | 34.47 (+4.58) | 30.74 (+6.83) |

**SCUNet `real-psnr` is the strongest blind denoiser in the set**, and its margin over Restormer
*grows* with noise (+0.01 → +0.17 → +0.70 dB) — consistent with the randomized-degradation training
that motivated the model. A correctly-tuned DRUNet beats it at ISO 1600, but that requires knowing σ;
DRUNet at a wrong σ scores **−2.73 dB**, i.e. worse than leaving the image alone.

The GAN variant costs 0.88–1.50 dB and is **effectively a no-op at ISO 1600** (+0.02 dB). It is a
perceptual mode, never the default, and it should be gated off low-noise input.

**And it replaces the incumbent.** NAFNet-SIDD-width64 is last among the blind denoisers at every
ISO — 3.00 / 2.96 / 2.90 dB behind — despite being **6.5× larger** (116.0 M vs 17.9 M params), and at
ISO 1600 it scores **−1.48 dB, worse than leaving the image alone**. That is precisely the failure
SCUNet's randomized-degradation training targets: NAFNet trains on SIDD's five smartphone sensors,
and NIND is DSLR-class Canon hardware, so being *off the training sensors* is the whole test.

⚠️ What this does *not* establish: it is a **generalization** result. Nothing here speaks to NAFNet's
in-domain SIDD performance, which is what a phone photo would exercise. NIND is DSLR-class hardware,
and PSNR judges the GAN variant on the axis it deliberately trades away.

## Licence

Port code MIT ([`LICENSE`](LICENSE)); weights MIT. Upstream `cszn/SCUNet` is **Apache-2.0** and its
notices are retained in [`LICENSE-upstream`](LICENSE-upstream) and [`NOTICE`](NOTICE), as Apache-2.0
§4 requires.
