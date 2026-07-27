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

## ⚠️ V4 is still open, and it decides positioning not viability

**No primary source reports SCUNet's SIDD or DND** — the authors deliberately skipped it. That
measurement decides whether SCUNet **replaces or complements** NAFNet. It needs corpus **C5** (ISO
brackets + dark frames), which also serves Restormer's real-denoise and DRUNet's σ selection.

The queue says *"do the measurement before the port"*. That gate is about **positioning**, not about
whether the port is buildable: the licence is clean, the weights are first-party, and the measurement
requires C5 either way. Stage 0 was therefore completed so the row is ready to build the moment the
decision lands.

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

## Remaining

- [ ] Stage 2: `MLXSCUNet` package + conformance suite, publish weights, registry row.
- [ ] **V4** measurement against corpus C5.

## Reproduce Stage 0

```bash
cd oracle
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch numpy einops timm safetensors thop
git clone --depth 1 https://github.com/cszn/SCUNet.git upstream
curl -sLO https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth   # into weights/
.venv/bin/python verify.py
```
