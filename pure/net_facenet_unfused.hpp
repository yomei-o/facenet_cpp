// BN-training (unfused) InceptionResnetV1: conv(no bias) + BatchNorm2d kept separate so training
// uses batch stats and eval uses running stats. Same structure as net_facenet.hpp. Used for
// training + gradcheck; eval-mode output matches the fused forward (~1e-6).
#pragma once
#include "autograd.hpp"
#include "face_ops.hpp"
#include "bn.hpp"           // batchnorm2d
#include "ops2d.hpp"        // mul_scalar, reshape
#include <fstream>
#include <string>
#include <vector>

struct ULayer { int kind; int64_t Co, Ci, kh, kw, sh, sw, ph, pw;
                Tensor w, b, gamma, beta; std::vector<float> rm, rv; };  // kind B0 C1 L2 N3
struct UProv { std::vector<ULayer> L; size_t i = 0; float eps = 1e-3f; ULayer& next() { return L[i++]; } };

inline UProv load_facenet_unfused(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  std::ifstream mf(D + "manifest_unfused.txt"); if (!mf) { printf("missing %smanifest_unfused.txt\n", D.c_str()); std::exit(1); }
  std::ifstream wf(D + "weights_unfused.bin", std::ios::binary); if (!wf) { printf("missing %sweights_unfused.bin\n", D.c_str()); std::exit(1); }
  auto rd = [&](int64_t n) { std::vector<float> v(n); wf.read((char*)v.data(), n * 4); return v; };
  int N; mf >> N; UProv p; p.L.reserve(N);
  for (int i = 0; i < N; ++i) {
    std::string k; mf >> k; ULayer L{};
    if (k == "B" || k == "C") {
      mf >> L.Co >> L.Ci >> L.kh >> L.kw >> L.sh >> L.sw >> L.ph >> L.pw;
      L.w = from_data({L.Co, L.Ci, L.kh, L.kw}, rd(L.Co * L.Ci * L.kh * L.kw), true);
      if (k == "B") { L.kind = 0;
        L.gamma = from_data({L.Co}, rd(L.Co), true); L.beta = from_data({L.Co}, rd(L.Co), true);
        L.rm = rd(L.Co); L.rv = rd(L.Co);
      } else { L.kind = 1; L.b = from_data({L.Co}, rd(L.Co), true); }
    } else if (k == "L") { L.kind = 2; mf >> L.Co >> L.Ci; L.w = from_data({L.Co, L.Ci}, rd(L.Co * L.Ci), true); }
    else { L.kind = 3; mf >> L.Co; L.gamma = from_data({L.Co}, rd(L.Co), true); L.beta = from_data({L.Co}, rd(L.Co), true);
           L.rm = rd(L.Co); L.rv = rd(L.Co); }
    p.L.push_back(std::move(L));
  }
  return p;
}

// BasicConv2d (conv -> BN -> ReLU); residual conv (conv+bias, no BN/ReLU)
inline Tensor u_bc(const Tensor& x, UProv& p, bool tr) {
  auto& L = p.next(); Tensor nob;
  Tensor y = conv2d_hw(x, L.w, nob, L.sh, L.ph, L.pw);
  y = batchnorm2d(y, L.gamma, L.beta, L.rm, L.rv, p.eps, tr, 0.1f);
  return relu(y);
}
inline Tensor u_conv(const Tensor& x, UProv& p) { auto& L = p.next(); return conv2d_hw(x, L.w, L.b, L.sh, L.ph, L.pw); }

inline Tensor u_block35(const Tensor& x, UProv& p, bool tr) {
  Tensor b0 = u_bc(x, p, tr);
  Tensor h1 = u_bc(x, p, tr); Tensor b1 = u_bc(h1, p, tr);
  Tensor h2 = u_bc(x, p, tr); h2 = u_bc(h2, p, tr); Tensor b2 = u_bc(h2, p, tr);
  Tensor m = u_conv(concat_ch({b0, b1, b2}), p);
  return relu(add(x, mul_scalar(m, 0.17f)));
}
inline Tensor u_block17(const Tensor& x, UProv& p, bool tr) {
  Tensor b0 = u_bc(x, p, tr);
  Tensor h = u_bc(x, p, tr); h = u_bc(h, p, tr); Tensor b1 = u_bc(h, p, tr);
  Tensor m = u_conv(concat_ch({b0, b1}), p);
  return relu(add(x, mul_scalar(m, 0.10f)));
}
inline Tensor u_block8(const Tensor& x, UProv& p, bool tr, float scale, bool do_relu) {
  Tensor b0 = u_bc(x, p, tr);
  Tensor h = u_bc(x, p, tr); h = u_bc(h, p, tr); Tensor b1 = u_bc(h, p, tr);
  Tensor m = u_conv(concat_ch({b0, b1}), p);
  Tensor out = add(x, mul_scalar(m, scale));
  return do_relu ? relu(out) : out;
}
inline Tensor u_mixed6a(const Tensor& x, UProv& p, bool tr) {
  Tensor b0 = u_bc(x, p, tr);
  Tensor h = u_bc(x, p, tr); h = u_bc(h, p, tr); Tensor b1 = u_bc(h, p, tr);
  Tensor b2 = maxpool2d(x, 3, 2, 0);
  return concat_ch({b0, b1, b2});
}
inline Tensor u_mixed7a(const Tensor& x, UProv& p, bool tr) {
  Tensor h = u_bc(x, p, tr); Tensor b0 = u_bc(h, p, tr);
  h = u_bc(x, p, tr); Tensor b1 = u_bc(h, p, tr);
  h = u_bc(x, p, tr); h = u_bc(h, p, tr); Tensor b2 = u_bc(h, p, tr);
  Tensor b3 = maxpool2d(x, 3, 2, 0);
  return concat_ch({b0, b1, b2, b3});
}

// full forward -> (N,512) L2-normalized embedding.
inline Tensor facenet_forward_u(Tensor x, UProv& p, bool tr) {
  p.i = 0;                                                // rewind provider (safe to call repeatedly)
  x = u_bc(x, p, tr); x = u_bc(x, p, tr); x = u_bc(x, p, tr);
  x = maxpool2d(x, 3, 2, 0);
  x = u_bc(x, p, tr); x = u_bc(x, p, tr); x = u_bc(x, p, tr);
  for (int i = 0; i < 5; ++i)  x = u_block35(x, p, tr);
  x = u_mixed6a(x, p, tr);
  for (int i = 0; i < 10; ++i) x = u_block17(x, p, tr);
  x = u_mixed7a(x, p, tr);
  for (int i = 0; i < 5; ++i)  x = u_block8(x, p, tr, 0.20f, true);
  x = u_block8(x, p, tr, 1.0f, false);
  x = gap(x);                                            // (N,1792)
  auto& Ll = p.next();                                   // last_linear (no bias)
  x = matmul(x, transpose2d(Ll.w));                      // (N,512)
  auto& Lb = p.next();                                   // last_bn (BN1d) as 4D
  x = reshape(x, {x->shape[0], x->shape[1], 1, 1});
  x = batchnorm2d(x, Lb.gamma, Lb.beta, Lb.rm, Lb.rv, p.eps, tr, 0.1f);
  x = reshape(x, {x->shape[0], x->shape[1]});
  return l2norm_rows(x);
}

// save weights in load order (B: w,gamma,beta,rm,rv | C: w,b | L: w | N: gamma,beta,rm,rv).
inline void save_facenet_unfused(UProv& p, const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  auto wr = [&](const std::vector<float>& v) { f.write((const char*)v.data(), v.size() * 4); };
  auto wt = [&](const Tensor& t) { wr(t->data); };
  for (auto& L : p.L) {
    if (L.kind == 0) { wt(L.w); wt(L.gamma); wt(L.beta); wr(L.rm); wr(L.rv); }
    else if (L.kind == 1) { wt(L.w); wt(L.b); }
    else if (L.kind == 2) { wt(L.w); }
    else { wt(L.gamma); wt(L.beta); wr(L.rm); wr(L.rv); }
  }
}
// reload weights (same manifest) from a checkpoint written by save_facenet_unfused.
inline void load_facenet_weights(UProv& p, const std::string& path) {
  std::ifstream f(path, std::ios::binary); if (!f) { printf("missing %s\n", path.c_str()); std::exit(1); }
  auto rd = [&](std::vector<float>& v) { f.read((char*)v.data(), v.size() * 4); };
  for (auto& L : p.L) {
    if (L.kind == 0) { rd(L.w->data); rd(L.gamma->data); rd(L.beta->data); rd(L.rm); rd(L.rv); }
    else if (L.kind == 1) { rd(L.w->data); rd(L.b->data); }
    else if (L.kind == 2) { rd(L.w->data); }
    else { rd(L.gamma->data); rd(L.beta->data); rd(L.rm); rd(L.rv); }
  }
}

// collect trainable params (conv/linear weights, residual bias, BN affine) for the optimizer.
inline std::vector<Tensor> facenet_params_u(UProv& p) {
  std::vector<Tensor> ps;
  for (auto& L : p.L) { if (L.w) ps.push_back(L.w); if (L.b) ps.push_back(L.b);
    if (L.gamma) ps.push_back(L.gamma); if (L.beta) ps.push_back(L.beta); }
  return ps;
}
