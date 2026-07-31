# InceptionResnetV1 (facenet-pytorch, `vggface2`) — exact spec for the pure-C++ port

Extracted from the real model via `inspect_facenet.py` (full dump: `facenet_arch_dump.txt`).
**27,910,327 params**, 132 Conv2d, 111 BatchNorm2d, 1 BatchNorm1d, 2 Linear. Input `1×3×160×160`,
output **512-D L2-normalized** embedding. BN `eps=1e-3`, momentum 0.1 (eval → running stats).

## Building blocks
- **BasicConv2d(Ci→Co, k, s, p)** = `Conv2d(bias=False) → BatchNorm2d(eps1e-3) → ReLU`. (111 of these.)
- **Block35** (scale **0.17**): `branch0 = BC(256→32,1)`; `branch1 = BC(256→32,1)→BC(32→32,3,p1)`;
  `branch2 = BC(256→32,1)→BC(32→32,3,p1)→BC(32→32,3,p1)`; `cat(96) → conv2d(96→256,1,**bias**)`;
  `out = out*0.17 + x → ReLU`. (repeat_1 = 5×)
- **Block17** (scale **0.10**): `branch0 = BC(896→128,1)`; `branch1 = BC(896→128,1)→BC(128→128,(1,7),p(0,3))
  →BC(128→128,(7,1),p(3,0))`; `cat(256) → conv2d(256→896,1,**bias**)`; `out = out*0.10 + x → ReLU`.
  (repeat_2 = 10×)
- **Block8** (scale **0.20**): `branch0 = BC(1792→192,1)`; `branch1 = BC(1792→192,1)→BC(192→192,(1,3),p(0,1))
  →BC(192→192,(3,1),p(1,0))`; `cat(384) → conv2d(384→1792,1,**bias**)`; `out = out*0.20 + x → ReLU`.
  (repeat_3 = 5×)

## Full forward (definition order == forward order)
```
conv2d_1a  BC(3→32,   3, s2)           # 160 -> 79
conv2d_2a  BC(32→32,  3, s1)           # 79 -> 77
conv2d_2b  BC(32→64,  3, s1, p1)       # 77 -> 77
maxpool_3a MaxPool(3, s2)              # 77 -> 38
conv2d_3b  BC(64→80,  1, s1)           # 38 -> 38
conv2d_4a  BC(80→192, 3, s1)           # 38 -> 36
conv2d_4b  BC(192→256,3, s2)           # 36 -> 17
repeat_1   5× Block35                  # 17×17×256
mixed_6a   Mixed_6a                    # 17 -> 8,  256 -> 896
repeat_2   10× Block17                 # 8×8×896
mixed_7a   Mixed_7a                    # 8 -> 3,   896 -> 1792
repeat_3   5× Block8                   # 3×3×1792
block8     Block8(scale=1.0, noReLU)   # out = conv2d(cat)*1.0 + x   (NO relu)
avgpool_1a AdaptiveAvgPool2d(1)        # 3×3 -> 1×1  (global mean)
dropout    p=0.6 (identity in eval)
last_linear Linear(1792→512, bias=False)
last_bn    BatchNorm1d(512, eps1e-3)
--> F.normalize(p=2, dim=1)            # 512-D unit vector   (classify=False)
# logits = Linear(512→8631) only when classify=True (VGGFace2) — not used for embeddings
```

### Mixed_6a (256→896, 17→8)
`branch0 = BC(256→384,3,s2)`; `branch1 = BC(256→192,1)→BC(192→192,3,p1)→BC(192→256,3,s2)`;
`branch2 = MaxPool(3,s2)`; `cat = [b0(384), b1(256), b2(256)] = 896`.

### Mixed_7a (896→1792, 8→3)
`branch0 = BC(896→256,1)→BC(256→384,3,s2)`; `branch1 = BC(896→256,1)→BC(256→256,3,s2)`;
`branch2 = BC(896→256,1)→BC(256→256,3,p1)→BC(256→256,3,s2)`; `branch3 = MaxPool(3,s2)`;
`cat = [b0(384), b1(256), b2(256), b3(896)] = 1792`.

## Parity reference (from `inspect_facenet.py`)
Input = `linspace(-1,1, 3*160*160).reshape(1,3,160,160)` (raw, no whitening).
Output embedding L2norm = 1.0; `embed[:8] = [0.023061, 0.006832, -0.059693, 0.043314, 0.000663,
0.082477, -0.061489, 0.050746]`.

## Fusion for the pure-C++ inferencer
- Fold each BasicConv2d `conv+bn` → one biased conv: `w' = w·(γ/√(var+eps))`, `b' = β − γ·mean/√(var+eps)`.
- Residual `conv2d` already has bias, no BN → keep as-is.
- Fold `last_bn` (BN1d) into `last_linear`: `W' = diag(γ/√(var+eps))·W`, `b' = β − γ·mean/√(var+eps)`.
- New autograd primitives needed vs the YOLO engine: `relu`, global-avg-pool, `linear` (matmul+bias),
  L2-normalize (√/÷). Training path additionally needs Triplet + ArcFace/softmax loss (planned).
