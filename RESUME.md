# RESUME — facenet_cpp remaining work

Pure-C++ InceptionResnetV1 face recognition. **Done + verified**: forward parity 1.19e-07, unfused +
gradcheck, losses (triplet + ArcFace + softmax, gradchecked), training (COCO/LFW), ONNX I/O
(export + onnxruntime 5.96e-08 + pure-C++ runner 1.19e-07), all sizes, LFW eval **97.3% @ cos≥0.48**,
Thrust device engine (parity + training + cuDNN **verified on a real T4**), and the **WebAssembly
webcam demo** (register + identify, live on GitHub Pages). Mostly complete — remaining is polish:

## ⏭ Nice-to-have
1. **On-device loss** — training currently bridges device heads → host for the TAL/loss each step
   (host↔device copy bound). Move the loss on-device for full GPU training throughput.
2. **Per-size pretrained weights** — ship/verify facenet at other input sizes if needed.
3. **In-browser face detection** — the WASM demo (`wasm/index.html`) uses a center-crop guide; add a
   lightweight detector (BlazeFace/MTCNN, itself WASM-able) to auto-crop and lift accuracy.
4. **ONNX under Colab onnxruntime** — re-run `pure/ref/onnx_verify_face.py` on Colab for good measure
   (already 5.96e-08 locally).

## Notes
- Live demo: https://yomei-o.github.io/facenet_cpp/wasm/  (register faces → identify; fp16 weights
  fetched from `weights/facenet/`). Fixed the earlier Identify crash (`map(Float32Array.from)`).
- Build env: emsdk at C:/emsdk (`EM_CONFIG=C:/emsdk/.emscripten`, PATH += upstream/emscripten +
  node/22.16.0_64bit/bin). nvcc for GPU: VC 14.31 host + `-Xcompiler "/Zc:preprocessor /EHsc /DNOMINMAX"`.
- fine-tune gotcha: pretrained fine-tune wants **lr≈1e-4** (1e-3 drifts confidence → 0 detections).
