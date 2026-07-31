#!/bin/sh
# Build the FaceNet WASM module (needs Emscripten: `source /path/to/emsdk/emsdk_env.sh`).
# Produces facenet.js + facenet.wasm + facenet.data (fp16 weights, ~47MB, preloaded into MEMFS).
set -e
cd "$(dirname "$0")"
emcc -O3 -std=c++20 -DNDEBUG \
  -I../pure -I../pure/third_party \
  facenet_wasm.cpp \
  -sEXPORTED_FUNCTIONS=_fn_embed,_fn_embed_chw,_fn_ready,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,getValue,HEAPU8,HEAPF32,FS \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB \
  -sMODULARIZE=1 -sEXPORT_NAME=createFaceNet \
  -sENVIRONMENT=web,node \
  -o facenet.js
# weights are NOT baked in: the page fetches ../weights/facenet/ and writes them into MEMFS.
echo "built: facenet.js facenet.wasm (weights loaded at runtime from weights/facenet/)"
