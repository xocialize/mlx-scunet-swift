"""P9 Stage 0: which config do the released checkpoints actually use?"""
import sys, torch
sys.path.insert(0, "upstream")
from models.network_scunet import SCUNet
torch.set_grad_enabled(False)

for name in ["scunet_color_real_psnr", "scunet_color_real_gan"]:
    sd = torch.load(f"weights/{name}.pth", map_location="cpu", weights_only=False)
    sd = sd.get("params", sd) if isinstance(sd, dict) and "params" in sd else sd
    print(f"=== {name} ===  ({len(sd)} tensors)")
    for cfg in ([2]*7, [4]*7):
        m = SCUNet(in_nc=3, config=cfg, dim=64)
        n = sum(p.numel() for p in m.parameters())
        try:
            m.load_state_dict(sd, strict=True)
            print(f"   config={cfg[0]}x7  params {n:>10,}  ✅ STRICT CLEAN")
            m.eval()
            x = torch.rand(1, 3, 128, 128)
            y = m(x)
            print(f"      forward {tuple(x.shape)} -> {tuple(y.shape)} range [{y.min():.4f},{y.max():.4f}]")
        except Exception as e:
            miss, unexp = m.load_state_dict(sd, strict=False)
            print(f"   config={cfg[0]}x7  params {n:>10,}  ❌ missing {len(miss)} unexpected {len(unexp)}")
