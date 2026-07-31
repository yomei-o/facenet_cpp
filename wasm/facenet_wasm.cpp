// FaceNet embedding compiled to WebAssembly (Emscripten). Exposes a tiny C API the browser calls:
// take an RGBA frame from a <canvas>, center-crop to square, resize to 160, standardize, run the
// pure-C++ InceptionResnetV1 forward, and return the 512-D unit embedding. Weights (fp16 fused) are
// preloaded into MEMFS at /facenet/. All inference is client-side — no image ever leaves the page.
//   build: see build.sh
#include "net_facenet.hpp"
#include <emscripten/emscripten.h>
#include <vector>
#include <cmath>
#include <algorithm>

static FaceProv* g_prov = nullptr;
static float g_emb[512];

extern "C" {

// Load the preloaded fp16 weights once; returns 1 when ready.
EMSCRIPTEN_KEEPALIVE int fn_ready() {
  if (!g_prov) g_prov = new FaceProv(load_facenet_fp16("/facenet/"));
  return g_prov ? 1 : 0;
}

// rgba: w*h*4 bytes (canvas getImageData().data). Returns pointer to 512 floats (unit embedding).
EMSCRIPTEN_KEEPALIVE float* fn_embed(unsigned char* rgba, int w, int h) {
  if (!g_prov) fn_ready();
  const int S = 160;
  int side = std::min(w, h), ox = (w - side) / 2, oy = (h - side) / 2;   // center square crop
  std::vector<float> chw(3 * S * S);
  auto px = [&](int yy, int xx, int c) { yy = std::clamp(yy, 0, h - 1); xx = std::clamp(xx, 0, w - 1); return (float)rgba[(yy * (size_t)w + xx) * 4 + c]; };
  for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
    float sy = (y + 0.5f) * side / S - 0.5f + oy, sx = (x + 0.5f) * side / S - 0.5f + ox;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) {
      float v = px(y0, x0, c) * (1 - fx) * (1 - fy) + px(y0, x0 + 1, c) * fx * (1 - fy)
              + px(y0 + 1, x0, c) * (1 - fx) * fy + px(y0 + 1, x0 + 1, c) * fx * fy;
      chw[(c * S + y) * S + x] = (v - 127.5f) / 128.f;                   // facenet standardization
    }
  }
  Tensor e = facenet_forward(from_data({1, 3, S, S}, chw), *g_prov);
  for (int i = 0; i < 512; ++i) g_emb[i] = e->data[i];
  return g_emb;
}

// pre-standardized CHW 3x160x160 -> 512 embedding (used by the node parity test).
EMSCRIPTEN_KEEPALIVE float* fn_embed_chw(float* chw) {
  if (!g_prov) fn_ready();
  std::vector<float> v(chw, chw + 3 * 160 * 160);
  Tensor e = facenet_forward(from_data({1, 3, 160, 160}, v), *g_prov);
  for (int i = 0; i < 512; ++i) g_emb[i] = e->data[i];
  return g_emb;
}
}
