"""P9 oracle — per-sub-op goldens for the SCUNet Swift port.

The window-attention path gets the most coverage: it is the first genuine shifted-window
implementation in this fleet, and both its failure modes (wrong QKV head split, wrong SW mask) are
shape-safe — they produce plausible output rather than an error.
"""
import importlib.util, os, sys
import numpy as np, torch
sys.path.insert(0, "upstream")

torch.set_grad_enabled(False)
OUT="goldens"; os.makedirs(OUT, exist_ok=True)
S = importlib.util.spec_from_file_location("scu", "upstream/models/network_scunet.py")
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)

sd = torch.load("weights/scunet_color_real_psnr.pth", map_location="cpu", weights_only=False)
net = M.SCUNet(in_nc=3, config=[4]*7, dim=64)
net.load_state_dict(sd, strict=True); net.eval()

man=[]
def save(n,a):
    a=np.ascontiguousarray(np.asarray(a,dtype=np.float32)); np.save(f"{OUT}/{n}.npy",a)
    man.append(f"{n+'.npy':34s} {str(a.shape):26s} min={a.min():+.5f} max={a.max():+.5f}")
    print(f"  saved {n}.npy {a.shape}")
def dump(n,t): save(n,t.detach().cpu().numpy())
def seeded(s,*sh): return torch.from_numpy(np.random.default_rng(s).standard_normal(sh,dtype=np.float32))

# --- WMSA, both types. NHWC in/out: upstream's WMSA already works in b h w c.
for tag, blk in [("w", net.m_down1[0].trans_block), ("sw", net.m_down1[1].trans_block)]:
    msa = blk.msa
    print(f"\n=== WMSA type={msa.type} heads={msa.n_heads} window={msa.window_size} ===")
    x = seeded(2100 if tag=="w" else 2101, 1, 32, 32, msa.input_dim)   # 4x4 windows of 8
    dump(f"wmsa_{tag}_in", x)
    dump(f"wmsa_{tag}_out", msa(x))
    # the relative-position table AS STORED (heads, 2w-1, 2w-1), and the gathered bias it produces
    dump(f"wmsa_{tag}_relparams", msa.relative_position_params)
    dump(f"wmsa_{tag}_relbias", msa.relative_embedding())
    if msa.type != "W":
        m = msa.generate_mask(4, 4, msa.window_size, msa.window_size//2)
        save(f"wmsa_{tag}_mask", m.numpy().astype(np.float32))

# --- Block (LN -> MSA -> +x ; LN -> MLP -> +x)
print("\n=== Block ===")
xb = seeded(2110, 1, 32, 32, 32)
dump("block_in", xb)
dump("block_w_out", net.m_down1[0].trans_block(xb))
dump("block_sw_out", net.m_down1[1].trans_block(xb))

# --- ConvTransBlock (NCHW at its boundary)
print("\n=== ConvTransBlock ===")
xc = seeded(2120, 1, 64, 32, 32)
dump("ctb_in", xc)
dump("ctb_w_out", net.m_down1[0](xc))
dump("ctb_sw_out", net.m_down1[1](xc))

# --- resamplers
print("\n=== down / up ===")
xd = seeded(2130, 1, 64, 32, 32); dump("down_in", xd); dump("down_out", net.m_down1[4](xd))
xu = seeded(2131, 1, 512, 8, 8);  dump("up_in", xu);   dump("up_out", net.m_up3[0](xu))

# --- head / tail
print("\n=== head / tail ===")
xh = seeded(2140, 1, 3, 32, 32); dump("head_in", xh); dump("head_out", net.m_head(xh))
xt = seeded(2141, 1, 64, 32, 32); dump("tail_in", xt); dump("tail_out", net.m_tail(xt))

# --- full model. 64-multiple sizes avoid the internal pad; 100 exercises it.
print("\n=== full model ===")
for size in (64, 128, 100):
    xi = torch.from_numpy(np.random.default_rng(2200+size).random((1,3,size,size), dtype=np.float32))
    dump(f"full_{size}_in", xi); dump(f"full_{size}_out", net(xi))

open(f"{OUT}/MANIFEST.txt","w").write(
    "SCUNet PyTorch goldens — fp32, CPU, NCHW except the WMSA/Block ones which are NHWC.\n"
    "checkpoint: weights/scunet_color_real_psnr.pth (cszn/KAIR v1.0, MIT)\n"
    "constructor: SCUNet(in_nc=3, config=[4]*7, dim=64)  head_dim=32 window=8 input_resolution=256\n"
    "input: RGB [0,1]; forward pads to a multiple of 64 (ReplicationPad2d) and crops back.\n\n"
    + "\n".join(man) + "\n")
print(f"\n✅ {len(man)} goldens")
