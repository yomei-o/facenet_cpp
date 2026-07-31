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

## Quick start — face recognition right after `git clone` (no Python)

Pretrained fp16 weights ship in the repo (`weights/facenet/`, ~47 MB). `demo_face` is fully
self-contained — bring two roughly-cropped face images (any size, auto-resized to 160):
```sh
# Windows (MSVC)                                            # Linux/macOS
cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\demo_face.cpp
demo_face verify   a.jpg b.jpg          # same person?     # g++ -O2 -std=c++17 -Ipure/third_party pure/demo_face.cpp -o demo_face
demo_face identify probe.jpg gallery/   # who is it? (gallery = folder-per-identity)
demo_face embed    face.jpg             # 512-D embedding
demo_face selftest                      # fp16 weights vs fp32 ref: worst ~1.9e-04
```
The bundled fp16 weights reproduce the fp32 model to ~2e-04 on the embedding. For higher precision
or training, regenerate fp32 weights with `export_facenet.py` (see below).

## Training + inference (pure C++)
One unified CLI (`facenet.cpp`) — or the standalone `train_face` / `verify_face`:
```sh
facenet train    <dataset_dir> <triplet|arcface|softmax> <steps> [--P 3 --K 4 --lr 1e-4 --ckpt out.bin]
facenet embed    <img>                 [--ckpt f --datanet d --imgsz 160]
facenet verify   <imgA> <imgB>         [--thr 0.5 ...]
facenet identify <probe> <gallery_dir> [...]
facenet export   <out.onnx>            [--imgsz 160]      # fused -> ONNX (opset 13)
```
Dataset = folder-per-identity (`root/<person>/<img>.jpg`). `--ckpt` loads a fine-tuned checkpoint;
without it, the pretrained embedding is used. **CPU speedup:** add `-DUSE_EIGEN
-Ipure/third_party/eigen_flat` (+ `/arch:AVX2` or `-march=native`) to route conv/matmul through
Eigen (same results, ~faster) — the conv/gemm already go through the `bk::` backend seam.
- **losses** (`face_loss.hpp`, all gradchecked to ~1e-4): `triplet_loss` (batch-all/hard, FaceNet
  paper), `arcface_loss` (additive angular margin), `softmax_ce_loss` (cosine softmax).
- **training** (`train_face.cpp` + `face_data.hpp`): Adam, from-pretrained fine-tune, BN train-mode,
  checkpoint save/reload. ArcFace on a 4-identity set: loss 8.86 → 5.37 in 5 steps (decreasing).
- **inference** (`verify_face.cpp`): unit-embedding cosine/euclidean; verify + rank-based identify.
- **real-data eval** (`lfw_eval.cpp`): on **LFW** (300 pairs, pretrained, no MTCNN align) the pure-C++
  embedding gives mean cosine **same 0.73 / diff 0.05** → **97.3% accuracy** at the calibrated
  threshold **cos ≥ 0.48** (now the `verify` default). Proper MTCNN alignment approaches the ~99.6%
  reference. `lfw_eval <pairs.txt>` sweeps the threshold; pairs list = `pathA pathB label` per line.

## Roadmap
1. ✅ extract InceptionResnetV1 arch from facenet-pytorch + fused forward exact parity (1e-7)
2. ✅ BN-training (unfused) forward + gradcheck (numeric vs analytic)
3. ✅ losses: Triplet (FaceNet paper) + ArcFace + cosine-softmax (all gradchecked)
4. ✅ training loop (dataset → loss → Adam), checkpoints, from-pretrained fine-tune
5. ✅ inference/verification CLI (embed, 1:1 verify, 1:N identify)
6. ✅ unified `facenet` CLI (train/embed/verify/identify/export) + CPU Eigen speedup
7. ✅ ONNX I/O — `facenet export` (opset 13, `onnx_build_face.hpp`) verified vs onnxruntime
   (**5.96e-08**) and a pure-C++ ONNX runner (`onnx_run_face.hpp`, **1.19e-07**)
8. ✅ standalone `demo_face` with bundled fp16 weights (clone-and-run, no Python; ~2e-04 vs fp32)
9. ✅ real-data eval on LFW (97.3% @ cos≥0.48, pure C++) + verify-threshold calibration
10. ✅ Thrust **device** engine (`dface_ops.hpp`/`dface.hpp`, one source CPU/GPU) — eval forward
    parity **1.42e-07** on the CPU-thrust backend; `nvcc -DUSE_CUDA` compiles clean (sm_75).
    Colab T4 run: `colab/gpu_check.ipynb` (only the GPU *execution* is pending hardware).
11. (later) device training + cuBLAS/cuDNN fast paths on GPU

Reused verbatim from the sibling engine: `autograd/backend/ops2d/linalg/bn/optim/ptio/dataset/
parallel/dtensor` + stb + flat Eigen. FaceNet-specific: `face_ops.hpp`, `net_facenet.hpp`, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses (stb: public-domain/MIT).
