// Pure-C++ InceptionResnetV1 (facenet-pytorch) fused forward — reproduces the pretrained model
// exactly. Structure is hardcoded (see pure/ref/ARCH.md); every conv/linear is BN-folded and fed
// in forward order from data_net/{manifest.txt,weights.bin} (export_facenet.py). 512-D L2 embedding.
#pragma once
#include "autograd.hpp"
#include "face_ops.hpp"
#include "ops2d.hpp"        // mul_scalar
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

struct FLayer { int kind; int64_t Co, Ci, kh, kw, sh, sw, ph, pw; Tensor w, b; };  // kind 0=conv 1=linear
struct FaceProv { std::vector<FLayer> L; size_t i = 0; FLayer& next() { return L[i++]; } };

// IEEE-754 half -> float (for the bundled fp16 demo weights).
inline float half2float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
  if (exp == 0) { if (mant == 0) f = sign; else { int e = -1; do { ++e; mant <<= 1; } while (!(mant & 0x400)); mant &= 0x3FF; f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13); } }
  else if (exp == 0x1F) f = sign | 0x7F800000u | (mant << 13);
  else f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  float o; std::memcpy(&o, &f, 4); return o;
}

inline FaceProv load_facenet(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  std::ifstream mf(D + "manifest.txt"); if (!mf) { printf("missing %smanifest.txt\n", D.c_str()); std::exit(1); }
  std::ifstream wf(D + "weights.bin", std::ios::binary); if (!wf) { printf("missing %sweights.bin\n", D.c_str()); std::exit(1); }
  auto rd = [&](int64_t n) { std::vector<float> v(n); wf.read((char*)v.data(), n * 4); return v; };
  int N; mf >> N; FaceProv p; p.L.reserve(N);
  for (int i = 0; i < N; ++i) {
    std::string k; mf >> k; FLayer L{};
    if (k == "C") {
      int hasb; mf >> L.Co >> L.Ci >> L.kh >> L.kw >> L.sh >> L.sw >> L.ph >> L.pw >> hasb; L.kind = 0;
      L.w = from_data({L.Co, L.Ci, L.kh, L.kw}, rd(L.Co * L.Ci * L.kh * L.kw), true);
      L.b = from_data({L.Co}, rd(L.Co), true);
    } else {  // "L"
      int hasb; mf >> L.Co >> L.Ci >> hasb; L.kind = 1;
      L.w = from_data({L.Co, L.Ci}, rd(L.Co * L.Ci), true);
      L.b = from_data({L.Co}, rd(L.Co), true);
    }
    p.L.push_back(std::move(L));
  }
  return p;
}

// load fp16 fused weights (bundled demo): manifest.txt + weights_fp16.bin (half), up-converted.
inline FaceProv load_facenet_fp16(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  std::ifstream mf(D + "manifest.txt"); if (!mf) { printf("missing %smanifest.txt\n", D.c_str()); std::exit(1); }
  std::ifstream wf(D + "weights_fp16.bin", std::ios::binary); if (!wf) { printf("missing %sweights_fp16.bin\n", D.c_str()); std::exit(1); }
  auto rd = [&](int64_t n) { std::vector<uint16_t> h(n); wf.read((char*)h.data(), n * 2);
    std::vector<float> v(n); for (int64_t i = 0; i < n; ++i) v[i] = half2float(h[i]); return v; };
  int N; mf >> N; FaceProv p; p.L.reserve(N);
  for (int i = 0; i < N; ++i) {
    std::string k; mf >> k; FLayer L{};
    if (k == "C") { int hb; mf >> L.Co >> L.Ci >> L.kh >> L.kw >> L.sh >> L.sw >> L.ph >> L.pw >> hb; L.kind = 0;
      L.w = from_data({L.Co, L.Ci, L.kh, L.kw}, rd(L.Co * L.Ci * L.kh * L.kw), true); L.b = from_data({L.Co}, rd(L.Co), true); }
    else { int hb; mf >> L.Co >> L.Ci >> hb; L.kind = 1;
      L.w = from_data({L.Co, L.Ci}, rd(L.Co * L.Ci), true); L.b = from_data({L.Co}, rd(L.Co), true); }
    p.L.push_back(std::move(L));
  }
  return p;
}

// --- block helpers (fused: each BasicConv2d = folded conv + ReLU) ---
inline Tensor f_conv(const Tensor& x, FaceProv& p) { auto& L = p.next(); return conv2d_hw(x, L.w, L.b, L.sh, L.ph, L.pw); }
inline Tensor f_bc(const Tensor& x, FaceProv& p) { return relu(f_conv(x, p)); }

inline Tensor f_block35(const Tensor& x, FaceProv& p) {
  Tensor b0 = f_bc(x, p);
  Tensor h1 = f_bc(x, p); Tensor b1 = f_bc(h1, p);
  Tensor h2 = f_bc(x, p); h2 = f_bc(h2, p); Tensor b2 = f_bc(h2, p);
  Tensor m = f_conv(concat_ch({b0, b1, b2}), p);        // residual conv2d (bias, no ReLU)
  return relu(add(x, mul_scalar(m, 0.17f)));
}
inline Tensor f_block17(const Tensor& x, FaceProv& p) {
  Tensor b0 = f_bc(x, p);
  Tensor h = f_bc(x, p); h = f_bc(h, p); Tensor b1 = f_bc(h, p);   // 1x1, 1x7, 7x1
  Tensor m = f_conv(concat_ch({b0, b1}), p);
  return relu(add(x, mul_scalar(m, 0.10f)));
}
inline Tensor f_block8(const Tensor& x, FaceProv& p, float scale, bool do_relu) {
  Tensor b0 = f_bc(x, p);
  Tensor h = f_bc(x, p); h = f_bc(h, p); Tensor b1 = f_bc(h, p);   // 1x1, 1x3, 3x1
  Tensor m = f_conv(concat_ch({b0, b1}), p);
  Tensor out = add(x, mul_scalar(m, scale));
  return do_relu ? relu(out) : out;
}
inline Tensor f_mixed6a(const Tensor& x, FaceProv& p) {
  Tensor b0 = f_bc(x, p);                                // 256->384, 3, s2
  Tensor h = f_bc(x, p); h = f_bc(h, p); Tensor b1 = f_bc(h, p);   // 192,192,256(s2)
  Tensor b2 = maxpool2d(x, 3, 2, 0);
  return concat_ch({b0, b1, b2});
}
inline Tensor f_mixed7a(const Tensor& x, FaceProv& p) {
  Tensor h = f_bc(x, p); Tensor b0 = f_bc(h, p);          // 256,384(s2)
  h = f_bc(x, p); Tensor b1 = f_bc(h, p);                 // 256,256(s2)
  h = f_bc(x, p); h = f_bc(h, p); Tensor b2 = f_bc(h, p); // 256,256,256(s2)
  Tensor b3 = maxpool2d(x, 3, 2, 0);
  return concat_ch({b0, b1, b2, b3});
}

// full forward: input (N,3,160,160) -> (N,512) L2-normalized embedding.
inline Tensor facenet_forward(Tensor x, FaceProv& p) {
  p.i = 0;                                                // rewind provider (safe to call repeatedly)
  x = f_bc(x, p); x = f_bc(x, p); x = f_bc(x, p);         // conv2d_1a/2a/2b
  x = maxpool2d(x, 3, 2, 0);                              // maxpool_3a
  x = f_bc(x, p); x = f_bc(x, p); x = f_bc(x, p);         // conv2d_3b/4a/4b
  for (int i = 0; i < 5; ++i)  x = f_block35(x, p);
  x = f_mixed6a(x, p);
  for (int i = 0; i < 10; ++i) x = f_block17(x, p);
  x = f_mixed7a(x, p);
  for (int i = 0; i < 5; ++i)  x = f_block8(x, p, 0.20f, true);
  x = f_block8(x, p, 1.0f, false);                        // final block8, noReLU
  x = gap(x);                                             // (N,1792)
  auto& L = p.next(); x = linear(x, L.w, L.b);            // last_linear+last_bn (folded) -> (N,512)
  return l2norm_rows(x);
}
