# FaceNet in the browser (WebAssembly)

A complete, client-side face-recognition demo: **register** faces from your webcam and **identify**
who is in front of the camera — the pure-C++ InceptionResnetV1 embedding compiled to WASM. No
server, no upload; faces live only in your browser's `localStorage`.

- `facenet_wasm.cpp` — Emscripten C API: `fn_embed(rgba, w, h) -> float[512]` (center-crop → 160 →
  standardize → forward → unit embedding). `fn_ready()` loads the preloaded fp16 weights.
- `index.html` — webcam UI: Register (name → embedding) / Identify (cosine vs registered, thr 0.48).
- `build.sh` — emcc build → `facenet.js` + `facenet.wasm` + `facenet.data` (fp16 weights, ~47 MB).

## Build
```sh
# one-time: get Emscripten
git clone https://github.com/emscripten-core/emsdk && cd emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh          # puts emcc on PATH
# build (from the repo)
cd facenet_cpp/wasm && ./build.sh
```

## Run
The camera needs a **secure context** (localhost or HTTPS):
```sh
cd facenet_cpp/wasm && python -m http.server 8000
# open http://localhost:8000/  -> Enable camera -> Register / Identify
```

## How it works
1. `getUserMedia` → `<video>` → `<canvas>` → `getImageData()` gives an RGBA frame.
2. The frame is copied into the WASM heap; `fn_embed` returns a 512-D L2-normalized embedding.
3. **Register** stores the embedding under a name (multiple samples per person are averaged).
4. **Identify** takes cosine similarity to each registered person; best ≥ threshold (0.48, calibrated
   on LFW) → that name, else *Unknown*.

## Notes / next steps
- No face **detection** yet: center your face in the on-screen circle. Adding a lightweight detector
  (BlazeFace / MTCNN, itself WASM-able) would auto-crop and lift accuracy.
- Single-threaded (no pthreads/SharedArrayBuffer needed). One embedding is ~1 forward of a 27.9M-param
  net — expect a fraction of a second per capture.
- Weights are fp16 (~2e-4 vs fp32). The `.data` file is fetched once and cached by the browser.
