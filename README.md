# facenet_cpp — pure C++ FaceNet (InceptionResnetV1), no PyTorch / no CMake — WIP

Same approach as the sibling repos ([yolov11_cpp](https://github.com/yomei-o/yolov11_cpp),
[yolo26_cpp](https://github.com/yomei-o/yolo26_cpp), …): a dependency-free C++ reverse-mode autograd
engine that reproduces a **real pretrained model exactly** (parity ~1e-7), with **training and
inference** and no Python at run time.

Reference model = **InceptionResnetV1** from [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
(`vggface2` pretrained, 27.9M params) → a **512-D L2-normalized face embedding**. Recognition =
embed two faces, compare by Euclidean/cosine distance.

## Status — forward parity WORKING ✅
The architecture was extracted from the real model (no guessing) — see
[pure/ref/ARCH.md](pure/ref/ARCH.md) (full dump: `pure/ref/facenet_arch_dump.txt`).

- **fused forward** (`net_facenet.hpp`): **exact parity** vs facenet-pytorch —
  `pure/m1_forward_face.cpp` = **`1.19e-07`** on the 512-D embedding (L2norm 1.0).
- New autograd primitives (`face_ops.hpp`): `relu`, asymmetric `pad_hw` (for the 1×7 / 7×1 Inception
  convs), global-avg-pool `gap`, dense `linear`, row `l2norm_rows` — all with backward for training.

```sh
pip install facenet-pytorch
python pure/ref/inspect_facenet.py                 # dump the real arch (once)
python pure/ref/export_facenet.py 160              # BN-folded weights + parity refs -> pure/ref/data_net/
# Windows (MSVC)
cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m1_forward_face.cpp
m1_forward_face pure/ref/data_net/ pure/ref/ 160   # -> worst 1.19e-07 MATCH
# Linux/macOS
g++ -O2 -std=c++17 -Ipure/third_party pure/m1_forward_face.cpp -o m1 && ./m1
```

## Roadmap
1. ✅ extract InceptionResnetV1 arch from facenet-pytorch + fused forward exact parity (1e-7)
2. **next:** BN-training (unfused) forward + gradcheck
3. losses: **Triplet** (FaceNet paper) + **ArcFace/softmax** (both, per request)
4. training loop (embed dataset → loss → Adam), checkpoints, from-pretrained fine-tune
5. inference/verification CLI (embed a face, 1:1 verify, 1:N identify) + standalone demo
6. all-in-one `facenet <train|embed|verify>` CLI; `.pt` I/O; ONNX
7. (later) Eigen / Thrust device / cuDNN backends, like the sibling repos

Reused verbatim from the sibling engine: `autograd/backend/ops2d/linalg/bn/optim/ptio/dataset/
parallel/dtensor` + stb + flat Eigen. FaceNet-specific: `face_ops.hpp`, `net_facenet.hpp`, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses (stb: public-domain/MIT).
