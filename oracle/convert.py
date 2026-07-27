"""P9 weight conversion: scunet_color_real_*.pth -> safetensors in MLX NHWC layout.

Three tensor families, and only two need touching:

  Conv2d           (O, I, kH, kW) -> (O, kH, kW, I)   transpose (0,2,3,1)
  ConvTranspose2d  (I, O, kH, kW) -> (O, kH, kW, I)   transpose (1,2,3,0)   [m_up{1,2,3}.0 ONLY]
  Linear / LayerNorm / relative_position_params        passthrough

`nn.Linear` weight is (out, in) in PyTorch and (out, in) in MLX — no transpose. Getting that wrong
is the mirror of the conv trap: it would load only where out == in, which for the qkv projection
(c -> 3c) it never is, so it fails loudly. The dangerous one is again ConvTranspose vs Conv, which
share a shape.

`relative_position_params` is stored (n_heads, 2w-1, 2w-1) because `WMSA.__init__` re-assigns it
through `.view(2w-1, 2w-1, heads).transpose(1,2).transpose(0,1)` AFTER initialisation. It rides
through untouched — but the Swift side must index it in that layout, not the declared one.
"""
import json, os, re, sys
import numpy as np, torch
from safetensors.numpy import save_file

STEMS = sys.argv[1:] or ["scunet_color_real_psnr", "scunet_color_real_gan"]
TRANSPOSED = re.compile(r"^m_up[123]\.0\.weight$")

for stem in STEMS:
    src = f"weights/{stem}.pth"
    if not os.path.exists(src): print(f"[skip] {src}"); continue
    sd = torch.load(src, map_location="cpu", weights_only=False)
    sd = sd.get("params", sd) if isinstance(sd, dict) and "params" in sd else sd

    out = os.path.join("converted", stem); os.makedirs(out, exist_ok=True)
    conv, stats = {}, {"conv":0, "convtranspose":0, "linear":0, "passthrough":0}
    for k, v in sd.items():
        a = v.detach().cpu().numpy().astype(np.float32)
        if a.ndim == 4:
            if TRANSPOSED.match(k):
                a = np.transpose(a, (1,2,3,0)); stats["convtranspose"] += 1
            else:
                a = np.transpose(a, (0,2,3,1)); stats["conv"] += 1
        elif a.ndim == 2:
            stats["linear"] += 1          # (out,in) in both frameworks — passthrough
        else:
            stats["passthrough"] += 1
        conv[k] = np.ascontiguousarray(a)

    assert stats["convtranspose"] == 3, f"expected 3 transposed convs, got {stats['convtranspose']}"
    total = sum(int(np.prod(v.shape)) for v in conv.values())
    print(f"=== {stem} ===")
    print(f"  tensors {len(conv)}  conv {stats['conv']} · convtranspose {stats['convtranspose']} "
          f"· linear {stats['linear']} · passthrough {stats['passthrough']}")
    print(f"  params  {total:,} ({total*4/1e6:.2f} MB fp32)")
    save_file(conv, os.path.join(out, "model.safetensors"),
              metadata={"format":"pt","source":f"cszn/KAIR v1.0 {stem}.pth","license":"MIT",
                        "layout":"MLX NHWC; conv (O,kH,kW,I); convtranspose from (I,O,k,k); "
                                 "linear (out,in) passthrough; relative_position_params (heads,2w-1,2w-1)",
                        "params":str(total)})
    json.dump({"stem":stem,"transforms":stats,"params":total},
              open(os.path.join(out,"CONVERSION.json"),"w"), indent=2)
    print(f"  -> {out}/model.safetensors\n")
