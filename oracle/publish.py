"""Publish the converted SCUNet weights to mlx-community (both variants)."""
import os, sys
from huggingface_hub import HfApi
API = HfApi(); HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = {
    "scunet_color_real_psnr": {
        "repo": "mlx-community/SCUNet-color-real-psnr-fp32",
        "objective": "MSE",
        "blurb": (
            "Trained on a fidelity objective. Conservative: it leaves residual noise rather than "
            "inventing detail, which is what you want when the output feeds another stage or is "
            "measured against a reference."),
    },
    "scunet_color_real_gan": {
        "repo": "mlx-community/SCUNet-color-real-gan-fp32",
        "objective": "adversarial",
        "blurb": (
            "Adversarially trained. Crisper and more plausible-looking, and it **hallucinates** "
            "texture where the fidelity model stays smooth. Upstream's own demos favour it for "
            "viewing; it is the wrong pick for anything scored against a ground truth."),
    },
}

CARD = """---
library_name: mlx
license: mit
license_link: https://github.com/cszn/KAIR/blob/master/LICENSE
base_model: cszn/SCUNet
pipeline_tag: image-to-image
tags:
  - mlx
  - denoising
  - image-restoration
  - blind-denoising
  - swin-transformer
  - scunet
---

# {repo}

[SCUNet](https://github.com/cszn/SCUNet) **blind** real-world denoising, converted to **Apple MLX**
for Apple-Silicon inference via [`mlx-scunet-swift`](https://github.com/xocialize/mlx-scunet-swift).

Zhang et al., *Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis*.
**17,946,072 parameters** (71.8 MB) — `config=[4]*7, dim=64`.

This is the **`{objective}`** checkpoint. {blurb}

## 🔑 The point of this model: there is nothing to configure

SCUNet takes **no noise level**. Its sibling DRUNet takes σ as a model input and exposes a strength
dial; NAFNet, FFTformer and Restormer bake a degradation into the checkpoint. SCUNet takes neither —
one forward pass, no σ to estimate, nothing for a caller to get wrong on a real photograph whose
noise level nobody has measured.

```swift
import SCUNetMLXCore

let model = SCUNet()                      // config=[4]*7 — NOT upstream's [2]*7 default
try model.loadWeights(from: weightsURL)
let clean = model.denoiseTiled(imageNHWC) // NHWC RGB in [0,1]
```

Or as an MLXEngine `imageRestore` ModelPackage (`MLXSCUNet.SCUNetRestorePackage`), which declares
`supportsStrength: false` — the contract distinguishing a blind backer from a dialled one.

## Architecture note: window attention tiles almost for free

SCUNet is a Swin-Conv-UNet — `ConvTransBlock` splits channels between a conv path and a
shifted-window attention path and re-fuses them. Because the attention is **strictly local** (8x8
windows, alternating W / SW), tiling barely perturbs the result. Measured at 512^2, tiled at 256
with 64 overlap versus the full-frame reference:

| overlap | PSNR vs full-frame | seam / interior gradient |
|---|---|---|
| 0 | 58.32 dB | 1.08x |
| **64** | **71.60 dB** | **1.00x** |

A ratio of 1.00x means tile boundaries are statistically indistinguishable from ordinary image
content. Restormer, whose attention is spatially *global*, does not get off this lightly.

Tile geometry must be **64-aligned**: the forward pass pads to a multiple of 64 and lays the window
grid out from the tile's own origin, so an unaligned origin shifts the window phase between
neighbouring tiles and leaves a seam feathering cannot remove.

## Conversion

MLX **NHWC**. 540 tensors: 117 `Conv2d`, **3 `ConvTranspose2d`**, 112 `Linear` (passthrough), 308
passthrough. The two 4-D transposes **cannot be told apart by shape**:

| | PyTorch | MLX | transpose |
|---|---|---|---|
| `Conv2d` | `(O, I, kH, kW)` | `(O, kH, kW, I)` | `(0,2,3,1)` |
| `ConvTranspose2d` | **`(I, O, kH, kW)`** | `(O, kH, kW, I)` | **`(1,2,3,0)`** |

Exactly `m_up{{1,2,3}}.0.weight` are the transposed convs; the converter asserts that count.

Two further traps worth knowing if you port this yourself:

- **`relative_position_params` is stored pre-permuted.** The constructor allocates
  `((2w-1)^2, heads)` and then *re-assigns* the parameter through
  `.view(2w-1, 2w-1, heads).transpose(1,2).transpose(0,1)` — so the checkpoint carries
  `(heads, 2w-1, 2w-1)`. Read the constructor, not the declaration.
- **The QKV head split is not per-head triples.** `rearrange(qkv, 'b nw np (threeh c) -> threeh b nw
  np c').chunk(3, dim=0)` puts **all q heads, then all k, then all v**. Splitting it the intuitive
  way is shape-identical and silently wrong.

## Parity

Gated against the PyTorch oracle on the CPU stream, fp32, **relative** error:

- **key contract** — 540 tensors / 17,946,072 params / 0 missing / 0 unused, strict load
- **attention internals** — 5/5 at **exactly 0.00e+00**: the stored bias table, the gathered
  `(heads, 64, 64)` bias, and the SW attention mask
- **WMSA end-to-end** — W 2.36e-07, SW 1.93e-07
- **blocks** — Block and ConvTransBlock, both types, worst 4.57e-07
- **resamplers** — the transposed conv is bit-identical (**0.00e+00**)
- **full model** — 64^2 / 128^2 / 100^2, worst 3.42e-06 (100^2 exercises the internal
  `ReplicationPad2d` and the crop back)

## Measured on real sensor noise

**No primary source reports SCUNet's SIDD or DND** — the authors deliberately skipped both — so we
measured it on [NIND](https://commons.wikimedia.org/wiki/Category:Natural_Image_Noise_Dataset) (CC0):
5 scenes x 4 ISOs on a locked-off camera with a compensating shutter, 768^2 centre crops, PSNR
against the ISO-100 reference. Pairs verified pixel-aligned and brightness-matched first.

| model | ISO 1600 | ISO 6400 | ISO 25600 |
|---|---|---|---|
| *untouched input* | *34.87* | *29.90* | *23.91* |
| **SCUNet real-psnr** | 36.39 (+1.52) | **34.95 (+5.06)** | **32.28 (+8.38)** |
| Restormer realDenoise | 36.38 (+1.51) | 34.78 (+4.89) | 31.58 (+7.67) |
| SCUNet real-gan | 34.89 (+0.02) | 33.60 (+3.70) | 31.40 (+7.49) |
| DRUNet, best sigma per row | **37.36 (+2.48)** | 34.47 (+4.58) | 30.74 (+6.83) |

`real-psnr` is the strongest blind denoiser in the set, and its margin over Restormer **grows with
noise** (+0.01 -> +0.17 -> +0.70 dB) — consistent with the randomized-degradation training that is
the model's whole thesis. A correctly-tuned DRUNet wins at ISO 1600, but that requires knowing sigma;
at a wrong sigma it scores **-2.73 dB**, worse than leaving the image alone.

`real-gan` costs 0.88-1.50 dB and is **effectively a no-op at ISO 1600** (+0.02 dB) — a perceptual
mode, not a default.

⚠️ NAFNet was not in this run, NIND is DSLR-class hardware, and PSNR judges the GAN variant on the
axis it deliberately trades away.

Code: Apache-2.0 ([`cszn/SCUNet`](https://github.com/cszn/SCUNet)). Weights: MIT, published
first-party by the author in the
[`cszn/KAIR` v1.0 release](https://github.com/cszn/KAIR/releases/tag/v1.0).
"""


def main():
    dry = "--dry-run" in sys.argv
    for name, meta in VARIANTS.items():
        card = CARD.format(repo=meta["repo"], objective=meta["objective"], blurb=meta["blurb"])
        if dry:
            print(f"===== {meta['repo']} =====\n{card}\n")
            continue
        w = os.path.join(HERE, "converted", name, "model.safetensors")
        assert os.path.exists(w), w
        print(f"[publish] {meta['repo']} ({os.path.getsize(w)/1e6:.2f} MB) …")
        API.create_repo(meta["repo"], repo_type="model", exist_ok=True)
        API.upload_file(path_or_fileobj=w, path_in_repo="model.safetensors", repo_id=meta["repo"])
        API.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=meta["repo"])
        print(f"[publish]   ok → https://huggingface.co/{meta['repo']}")


main()
