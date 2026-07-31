# Dump the REAL facenet-pytorch InceptionResnetV1 architecture so the pure-C++ port
# reproduces it exactly (no guessing). Prints: module tree with reprs, every parameter
# name+shape in order, and a fixed-input embedding for parity.
#   pip install facenet-pytorch ; python inspect_facenet.py
import torch, numpy as np
from facenet_pytorch import InceptionResnetV1

torch.manual_seed(0)
m = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

print("=" * 70, "\nMODULE TREE (named_modules, leaf reprs)\n" + "=" * 70)
for name, mod in m.named_modules():
    kids = list(mod.children())
    if kids:  # container: print name + type only
        print(f"[{name or '<root>'}] {type(mod).__name__}")
    else:     # leaf: full repr
        print(f"    {name} = {mod}")

print("=" * 70, "\nPARAMETERS (name: shape) in definition order\n" + "=" * 70)
tot = 0
for n, p in m.named_parameters():
    print(f"  {n:55s} {tuple(p.shape)}")
    tot += p.numel()
print(f"buffers (running stats etc.):")
for n, b in m.named_buffers():
    print(f"  {n:55s} {tuple(b.shape)}")
print(f"TOTAL params: {tot:,}")

# fixed input -> embedding, for C++ parity. facenet expects 160x160 RGB, normalized.
x = torch.linspace(-1, 1, 3 * 160 * 160).reshape(1, 3, 160, 160)
with torch.no_grad():
    emb = m(x)
print("=" * 70, "\nPARITY REF\n" + "=" * 70)
print("input : shape", tuple(x.shape), " (linspace -1..1)")
print("embed : shape", tuple(emb.shape), " L2norm=", float(emb.norm()))
print("embed[:8] =", np.array2string(emb[0, :8].numpy(), precision=6))
