# Export the pretrained InceptionResnetV1 (facenet-pytorch) for the pure-C++ fused inferencer.
# BN-folds every BasicConv2d (conv+bn->biased conv) and folds last_bn(BN1d) into last_linear.
# Emits: data_net/manifest.txt (ordered op geometry), data_net/weights.bin (float32 blob, w[,b]
# per op in forward order), and ref/{input.bin,embed.bin} for parity.
#   python export_facenet.py [imgsz=160]
import sys, os, struct, numpy as np, torch
from facenet_pytorch import InceptionResnetV1

S = int(sys.argv[1]) if len(sys.argv) > 1 else 160
OUT = os.path.join(os.path.dirname(__file__), "data_net"); os.makedirs(OUT, exist_ok=True)
REF = os.path.dirname(__file__)
m = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

blob = bytearray(); manifest = []   # each: ("C",Co,Ci,kh,kw,sh,sw,ph,pw,hasb) or ("L",out,inn,hasb)
def put(a): blob.extend(np.asarray(a, dtype=np.float32).tobytes())

def fold_conv(conv, bn):
    w = conv.weight.detach().numpy()                       # (Co,Ci,kh,kw)
    g = bn.weight.detach().numpy(); b = bn.bias.detach().numpy()
    rm = bn.running_mean.detach().numpy(); rv = bn.running_var.detach().numpy()
    s = np.sqrt(rv + bn.eps); sc = g / s
    w2 = w * sc[:, None, None, None]; b2 = b - g * rm / s
    return w2, b2

def emit_conv(w, b, conv):
    Co, Ci, kh, kw = w.shape
    sh, sw = conv.stride; ph, pw = conv.padding
    manifest.append(("C", Co, Ci, kh, kw, sh, sw, ph, pw, 1))
    put(w); put(b)

# Walk modules in definition/forward order. BasicConv2d -> folded; Block .conv2d -> plain(bias).
for name, mod in m.named_modules():
    t = type(mod).__name__
    if t == "BasicConv2d":
        w, b = fold_conv(mod.conv, mod.bn); emit_conv(w, b, mod.conv)
    elif t == "Conv2d" and name.endswith(".conv2d"):        # residual projection conv (has bias, no BN)
        emit_conv(mod.weight.detach().numpy(), mod.bias.detach().numpy(), mod)

# head: fold last_bn (BN1d) into last_linear (no bias)
W = m.last_linear.weight.detach().numpy()                   # (512,1792)
bn = m.last_bn; g = bn.weight.detach().numpy(); be = bn.bias.detach().numpy()
rm = bn.running_mean.detach().numpy(); rv = bn.running_var.detach().numpy()
s = np.sqrt(rv + bn.eps); W2 = (g / s)[:, None] * W; b2 = be - g * rm / s
manifest.append(("L", W2.shape[0], W2.shape[1], 1)); put(W2); put(b2)

with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write(f"{len(manifest)}\n")
    for e in manifest: f.write(" ".join(str(x) for x in e) + "\n")
open(os.path.join(OUT, "weights.bin"), "wb").write(blob)

# parity ref: raw linspace input (matches inspect_facenet.py)
x = torch.linspace(-1, 1, 3 * S * S).reshape(1, 3, S, S)
with torch.no_grad(): emb = m(x)
x.numpy().astype(np.float32).tofile(os.path.join(REF, "input.bin"))
emb.numpy().astype(np.float32).tofile(os.path.join(REF, "embed.bin"))
print(f"wrote {len(manifest)} ops, weights.bin {len(blob)/1e6:.1f} MB, imgsz={S}")
print(f"ref embed L2norm={float(emb.norm()):.6f}  embed[:6]={emb[0,:6].numpy()}")
