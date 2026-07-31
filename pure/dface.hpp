// Device-resident (Thrust) InceptionResnetV1 eval forward — one source, CPU under
// THRUST_DEVICE_SYSTEM_CPP / GPU under nvcc -DUSE_CUDA (+cuBLAS/+cuDNN). Mirrors
// net_facenet_unfused.hpp eval (dbn_eval running stats) so it reproduces the fp32 reference
// embedding. Loads the unfused weights (manifest_unfused.txt + weights_unfused.bin).
#pragma once
#include "dface_ops.hpp"
#include <fstream>
#include <string>
#include <vector>

struct DFLayer { int kind; int64_t Co, Ci, kh, kw, sh, sw, ph, pw; DT w, b, gamma, beta, rm, rv; };
struct DFaceProv { std::vector<DFLayer> L; size_t i = 0; float eps = 1e-3f; DFLayer& next() { return L[i++]; } };

inline DFaceProv dface_build(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  std::ifstream mf(D + "manifest_unfused.txt"); if (!mf) { printf("missing %smanifest_unfused.txt\n", D.c_str()); std::exit(1); }
  std::ifstream wf(D + "weights_unfused.bin", std::ios::binary); if (!wf) { printf("missing %sweights_unfused.bin\n", D.c_str()); std::exit(1); }
  auto rd = [&](int64_t n) { std::vector<float> v(n); wf.read((char*)v.data(), n * 4); return v; };
  int N; mf >> N; DFaceProv p; p.L.reserve(N);
  for (int i = 0; i < N; ++i) {
    std::string k; mf >> k; DFLayer L{};
    if (k == "B" || k == "C") {
      mf >> L.Co >> L.Ci >> L.kh >> L.kw >> L.sh >> L.sw >> L.ph >> L.pw;
      L.w = dfrom({L.Co, L.Ci, L.kh, L.kw}, rd(L.Co * L.Ci * L.kh * L.kw));
      if (k == "B") { L.kind = 0; L.gamma = dfrom({L.Co}, rd(L.Co)); L.beta = dfrom({L.Co}, rd(L.Co));
        L.rm = dfrom({L.Co}, rd(L.Co)); L.rv = dfrom({L.Co}, rd(L.Co)); }
      else { L.kind = 1; L.b = dfrom({L.Co}, rd(L.Co)); }
    } else if (k == "L") { L.kind = 2; mf >> L.Co >> L.Ci; L.w = dfrom({L.Co, L.Ci}, rd(L.Co * L.Ci)); }
    else { L.kind = 3; mf >> L.Co; L.gamma = dfrom({L.Co}, rd(L.Co)); L.beta = dfrom({L.Co}, rd(L.Co));
           L.rm = dfrom({L.Co}, rd(L.Co)); L.rv = dfrom({L.Co}, rd(L.Co)); }
    p.L.push_back(std::move(L));
  }
  return p;
}

inline DT d_bc(DT x, DFaceProv& p) { auto& L = p.next(); DT y = dconv2d_hw(x, L.w, DT(), L.sh, L.ph, L.pw);
  y = dbn_eval(y, L.gamma, L.beta, L.rm, L.rv, p.eps); return drelu(y); }
inline DT d_cv(DT x, DFaceProv& p) { auto& L = p.next(); return dconv2d_hw(x, L.w, L.b, L.sh, L.ph, L.pw); }

inline DT d_block35(DT x, DFaceProv& p) { DT b0 = d_bc(x, p), h1 = d_bc(x, p), c1 = d_bc(h1, p), h2 = d_bc(x, p); h2 = d_bc(h2, p); DT c2 = d_bc(h2, p);
  return drelu(dadd(x, dmul_scalar(d_cv(dconcat({b0, c1, c2}), p), 0.17f))); }
inline DT d_block17(DT x, DFaceProv& p) { DT b0 = d_bc(x, p), h = d_bc(x, p); h = d_bc(h, p); DT b1 = d_bc(h, p);
  return drelu(dadd(x, dmul_scalar(d_cv(dconcat({b0, b1}), p), 0.10f))); }
inline DT d_block8(DT x, DFaceProv& p, float sc, bool relu) { DT b0 = d_bc(x, p), h = d_bc(x, p); h = d_bc(h, p); DT b1 = d_bc(h, p);
  DT o = dadd(x, dmul_scalar(d_cv(dconcat({b0, b1}), p), sc)); return relu ? drelu(o) : o; }
inline DT d_mixed6a(DT x, DFaceProv& p) { DT b0 = d_bc(x, p), h = d_bc(x, p); h = d_bc(h, p); DT b1 = d_bc(h, p), b2 = dmaxpool2d(x, 3, 2, 0); return dconcat({b0, b1, b2}); }
inline DT d_mixed7a(DT x, DFaceProv& p) { DT h = d_bc(x, p), b0 = d_bc(h, p); h = d_bc(x, p); DT b1 = d_bc(h, p); h = d_bc(x, p); h = d_bc(h, p); DT b2 = d_bc(h, p), b3 = dmaxpool2d(x, 3, 2, 0); return dconcat({b0, b1, b2, b3}); }

// full eval forward -> (N,512) L2-normalized device embedding.
inline DT dface_forward(DT x, DFaceProv& p) {
  p.i = 0;
  x = d_bc(x, p); x = d_bc(x, p); x = d_bc(x, p); x = dmaxpool2d(x, 3, 2, 0); x = d_bc(x, p); x = d_bc(x, p); x = d_bc(x, p);
  for (int i = 0; i < 5; ++i)  x = d_block35(x, p);
  x = d_mixed6a(x, p);
  for (int i = 0; i < 10; ++i) x = d_block17(x, p);
  x = d_mixed7a(x, p);
  for (int i = 0; i < 5; ++i)  x = d_block8(x, p, 0.20f, true);
  x = d_block8(x, p, 1.0f, false);
  x = dgap(x);                                           // (N,1792)
  auto& Ll = p.next(); x = dmatmul(x, dtranspose2d(Ll.w));  // last_linear (no bias) -> (N,512)
  auto& Lb = p.next();                                   // last_bn (BN1d) as 4D eval
  x = dreshape(x, {x->shape[0], x->shape[1], 1, 1});
  x = dbn_eval(x, Lb.gamma, Lb.beta, Lb.rm, Lb.rv, p.eps);
  x = dreshape(x, {x->shape[0], x->shape[1]});
  return dl2norm_rows(x);
}
