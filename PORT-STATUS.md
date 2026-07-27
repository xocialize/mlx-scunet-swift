# mlx-scunet-swift — port status

**Work order:** P9 in `mlxengine-todo/PORT-QUEUE.md` — SCUNet, blind real-world denoise.

Upstream: [`cszn/SCUNet`](https://github.com/cszn/SCUNet) — **Apache-2.0**, zero non-commercial text.
Weights first-party from the [`cszn/KAIR` v1.0 release](https://github.com/cszn/KAIR/releases/tag/v1.0)
(MIT). **17,946,072 parameters.**

## Stage 0 ✅ PASSED (2026-07-27)

| Fact | Verified |
|---|---|
| Licence | Apache-2.0 code, MIT weights release · no NC anywhere |
| Weights | `scunet_color_real_psnr.pth` (28,300 dl) · `scunet_color_real_gan.pth` (384,940 dl) — 71.98 MB each, 540 tensors |
| Params | **17,946,072** with the released config |
| Load | `strict=True` **clean** for both checkpoints |

### 🔴 Config trap

The constructor **defaults to `config=[2]*7`**, which gives 9,662,892 params and fails with **3
missing / 269 unexpected**. Every released checkpoint uses **`config=[4,4,4,4,4,4,4], dim=64`**, and
every upstream `main_test_scunet_*.py` passes it explicitly. Using the default is the obvious mistake.

### Three porting traps, found by reading source

1. **The relative-position bias is stored pre-permuted.** `WMSA.__init__` re-assigns it as
   `.view(2w-1, 2w-1, heads).transpose(1,2).transpose(0,1)`, so the checkpoint layout is *already*
   rotated relative to the naive `(…, heads)` shape. Indexing must match that, not the intuitive form.
2. **Block type is baked at construction, not runtime.** `ConvTransBlock.__init__` silently downgrades
   `SW` → `W` when `input_resolution <= window_size` — and `input_resolution` is a *constructor
   argument* (default 256) fed by the UNet's decreasing resolution schedule. Whether a deep block uses
   shifted windows is therefore fixed at build time and has nothing to do with the image passed in. A
   port that recomputed this from the runtime size would silently give some blocks the wrong attention.
3. **`models/network_scunet.py` imports `thop` at module scope** purely for FLOP profiling — you must
   install a profiling library just to import the model.

### Why this port is bigger than its four siblings

FFTformer, HVI-CIDNet, Restormer and DRUNet are conv-nets plus channel attention — variations on one
another. SCUNet is the first genuine **window attention with shifted windows**: cyclic roll, a
generated attention mask, and relative-position-bias indexing. `ConvTransBlock` also *splits* channels
between a conv path and a transformer path and concatenates. The queue's note — *"write once,
reusable"* — is the point: this unlocks the Swin family generally.

### On V4, as it looked at Stage 0

**No primary source reports SCUNet's SIDD or DND** — the authors deliberately skipped it — so
whether SCUNet **replaces or complements** NAFNet was unmeasured, and the queue said *"do the
measurement before the port"*.

Stage 0 proceeded anyway on the reasoning that the gate is about **positioning**, not viability: the
licence is clean, the weights are first-party, and the measurement was needed either way. That call
held — see **V4** below, which was answered on NIND rather than SIDD/DND and came out decisively.

## Stage 1 ✅ PASSED (2026-07-27) — core ported, all gates green

`swift run -c release scunet-gate oracle/goldens oracle/converted/scunet_color_real_psnr/model.safetensors`

| Gate | Content | Worst relative error | tol |
|---|---|---|---|
| S0 | key contract · **540 tensors, 17,946,072 params**, strict update clean, both checkpoints | exact | — |
| S1 | stored bias table, gathered bias, SW mask | **0.00e+00** (all five exact) | 0 |
| S2 | WMSA end-to-end, W and SW | 2.36e-07 | 2e-6 |
| S3 | Block W/SW, ConvTransBlock W/SW | 4.57e-07 | 2e-6 |
| S4 | stride-conv down, **convtranspose up (0.00e+00)**, head, tail | 1.11e-06 | 2e-6 |
| S5 | full model 64² / 128² / **100² (replication pad + crop)** | 3.42e-06 | 2e-5 |

### Why the gate ladder is shaped the way it is

Window attention's two porting errors are **shape-safe** — they produce a plausible image rather
than an error, so a single end-to-end gate would report "slightly off" without naming a cause:

- **QKV head split.** `rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c').chunk(3, dim=0)` puts
  **all q heads, then all k, then all v** in the projection's output channels. Splitting per-head
  into (q,k,v) triples is shape-identical and silently wrong. Caught by S2.
- **SW mask.** Only the *last* window row and column are masked — the wrap-around the cyclic roll
  creates. Omitting it lets opposite image edges attend to each other. Caught by S1 (the table,
  exact) before S2 consumes it.

S1 gating the *tables* at `tol: 0` is what makes this ladder diagnostic: all five land at exactly
0.00e+00, so the pre-permuted `(nHeads, 2w−1, 2w−1)` layout and the mask geometry are proven, not
inferred from a downstream number that merely looks small.

### Two traps found while writing the Swift, not in the Python

1. **`Module` reflection collects every stored `MLXArray` property as a parameter.** The constant
   relative-position gather table, declared as a plain `let relIndex: MLXArray`, appeared in
   `parameters()` as **28 phantom tensors** and S0 failed with 28 missing keys. It is now boxed in a
   non-`Module` class (`RelativeIndexTable`) — which also lets all 28 blocks share one table, since
   it depends only on `window`.
2. **`padToMultiple` here is replication, not reflection.** The sibling DRUNet port reflect-pads;
   SCUNet's `ReplicationPad2d` is shape-identical and differs only at the borders. The 64²/128²
   goldens bypass padding entirely, so **only the 100² case discriminates** — that is why it exists.

`mlx-swift`'s `roll` takes a single shift, so the cyclic shift is two sequential rolls on independent
axes; that composes to upstream's joint roll exactly (S2 SW, 1.93e-07).

## Stage 2 ✅ PASSED (2026-07-27) — `MLXSCUNet.SCUNetRestorePackage`, published

`imageRestore`, **`supportsStrength: false`** — see below. Descriptor `scunet-denoise`.
Weights: [`mlx-community/SCUNet-color-real-psnr-fp32`](https://huggingface.co/mlx-community/SCUNet-color-real-psnr-fp32)
· [`mlx-community/SCUNet-color-real-gan-fp32`](https://huggingface.co/mlx-community/SCUNet-color-real-gan-fp32).
Both re-downloaded fresh and re-run through the full gate ladder — all green.

Conformance: 12 offline tests green (MAT, CAN run + cadence, manifest/licence, footprint split,
`QuantConfigured`, distinct repos, tile geometry).

### Footprint — MEASURED, and the gate under-read it by 3.3×

```
[scunet-real-psnr] SPLIT floor=0.10GB peak=4.48GB act=4.38GB retain=0.56GB
                   engine=0.18GB reserve=2.00GB load=0.0s run=4.9s   @1920x1080
```

Declared resident 180 MB (floor 103.5 MB), activation **5.0 GB** (measured 4.38 GB).

The gate's `--bench` read **1.33 GB** for the same tile size. Same direction, same cause, and the
same size of miss as every prior port in this batch: `--bench` calls the core directly and reads
after the fact, so it sees neither the engine's decode/encode buffers nor the transient peak that
the harness's 150 ms sampling catches. **`--bench` is for comparing tile sizes; the harness number
is the admission basis.**

### Tiling: SCUNet gets off far more lightly than Restormer

Untiled is not viable — 10.04 GB phys at 1024², linear in pixels, so ~20 GB at 1080p. Tiled:

| tile | phys @1080p | time |
|---|---|---|
| 256 | 0.74 GB | 9.1 s |
| **384** | **1.33 GB** | **4.9 s** |
| 512 | 3.60 GB | 3.4 s |

Overlap sweep at 512², tiled at 256, against the full-frame reference:

| overlap | PSNR | seam / interior gradient |
|---|---|---|
| 0 (and 16, 32 → rounded down) | 58.32 dB | 1.08× |
| **64** (and 96 → rounded to 64) | **71.60 dB** | **1.00×** |

**1.00× means tile boundaries are statistically indistinguishable from ordinary image content** —
a much better result than the sibling ports got, and the reason is architectural: SCUNet's mixing is
strictly local (8-px windows plus 3×3 convs), whereas Restormer's channel attention reduces over the
whole feature map, so a tile boundary there changes global statistics. Window attention tiles almost
for free.

The 64-alignment is enforced (16, 32 and 96 all round down, visibly) because it is a **correctness**
property: `forward` pads to a multiple of 64 and lays the window grid out from the tile's own
origin, so an unaligned origin shifts the window phase between neighbours and leaves a seam that
feathering cannot remove.

### Why `supportsStrength: false` is the interesting declaration

SCUNet is the fleet's only genuinely **blind** restorer. NAFNet / FFTformer / Restormer bake a
degradation into the checkpoint; DRUNet takes σ as a model input and is the backer that *has* a dial
(contract 1.30.0 exists for it). SCUNet takes neither — so `appliedStrength` comes back `nil`, which
a caller can read as "this backer has no strength", distinct from "strength zero". The conformance
suite asserts the descriptor keeps saying no.

## V4 ✅ MEASURED (2026-07-27) — SCUNet is the strongest blind denoiser we hold

Measured on **NIND** (CC0), 5 scenes × 4 ISOs, 768² centre crops, PSNR vs the ISO-100 reference.
Full record: `mlxengine-image/corpus/nind/RESULTS.md`.

| model | ISO 1600 | ISO 6400 | ISO 25600 |
|---|---|---|---|
| *untouched input* | *34.87* | *29.90* | *23.91* |
| **SCUNet real-psnr** | 36.39 (+1.52) | **34.95 (+5.06)** | **32.28 (+8.38)** |
| Restormer realDenoise | 36.38 (+1.51) | 34.78 (+4.89) | 31.58 (+7.67) |
| SCUNet real-gan | 34.89 (+0.02) | 33.60 (+3.70) | 31.40 (+7.49) |
| **NAFNet-SIDD-width64** | 33.39 (−1.48) 🔴 | 31.99 (+2.09) | 29.38 (+5.48) |
| DRUNet, best σ per row | **37.36 (+2.48)** | 34.47 (+4.58) | 30.74 (+6.83) |

`real-psnr` wins at 6400 and 25600 and ties Restormer at 1600, and **its margin over Restormer grows
with noise** (+0.01 → +0.17 → +0.70 dB) — consistent with the randomized-degradation training that
motivated the row. DRUNet beats it at 1600 *when σ is right*, and scores **−2.73 dB** when it is not.

The GAN variant costs 0.88–1.50 dB and is a **no-op at ISO 1600** (+0.02 dB): a perceptual mode,
never the default, and it should be gated off low-noise input.

**NAFNet-SIDD-width64 is last at every ISO** — 3.00 / 2.96 / 2.90 dB behind SCUNet despite **6.5×
the parameters** (116.0 M vs 17.9 M) — and at ISO 1600 it scores **−1.48 dB, worse than doing
nothing**. Exactly the generalization failure the queue predicted for it: SIDD's five smartphone
sensors vs DSLR-class Canon.

🔴 **This arm was nearly not run, for a bad reason.** It was recorded as blocked on *"no PyTorch
NAFNet in our oracle set"* — a category error. The requirement is a verified **implementation**, not a
PyTorch one, and we hold two (`xocialize/nafnet-mlx`, `xocialize/mlx-nafnet-swift`). It took ~20 min
once posed correctly, and it changed the conclusion from "strongest of the three we hold" to
"replaces the incumbent". **State the requirement, not the artefact you happened to look for.**

⚠️ Limits: a **generalization** result — it says nothing about NAFNet in-domain on SIDD, which is what
a phone photo exercises. NIND is DSLR-class hardware, and PSNR judges the GAN variant on the axis it
deliberately trades away.

## Remaining

Nothing blocking. Product validation for the fleet as a whole still wants corpus C1–C5
(`mlxengine-todo/CORPUS-NEEDS.md`).

## Reproduce Stage 0

```bash
cd oracle
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch numpy einops timm safetensors thop
git clone --depth 1 https://github.com/cszn/SCUNet.git upstream
curl -sLO https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth   # into weights/
.venv/bin/python verify.py
```
