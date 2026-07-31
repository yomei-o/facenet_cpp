// Gradcheck the training losses (triplet + ArcFace + softmax) on tiny random embeddings — fast,
// no net. Verifies each loss backward wrt the (pre-normalized) embedding and the ArcFace weights.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m_loss_face.cpp
#include "face_loss.hpp"
#include "net_facenet_unfused.hpp"   // (unused, but keeps include set consistent)
#include <cstdio>
#include <cmath>
#include <functional>
#include <random>

int main() {
  const int B = 6, D = 8, C = 3;
  std::mt19937 rng(1); std::normal_distribution<float> nd(0, 1);
  std::vector<float> ev(B * D), wv(C * D); for (auto& v : ev) v = nd(rng); for (auto& v : wv) v = nd(rng);
  std::vector<int> lab = {0, 0, 1, 1, 2, 2};

  auto run = [&](const char* name, std::function<Tensor(Tensor&, Tensor&)> mk, bool checkW) {
    Tensor E = from_data({B, D}, ev, true), W = from_data({C, D}, wv, true);
    Tensor emb = l2norm_rows(E);                 // net emits unit embeddings
    Tensor L = mk(emb, W); backward(L);
    struct Ck { const char* n; float* d; float ana; };   // d points into ev/wv (what FD rebuilds from)
    std::vector<Ck> cks = {{"E[3]", &ev[3], E->grad[3]}, {"E[20]", &ev[20], E->grad[20]}};
    if (checkW) cks.push_back({"W[5]", &wv[5], W->grad[5]});
    const float eps = 1e-3f; int ok = 0;
    printf("[%s] L=%.5f\n", name, L->data[0]);
    for (auto& c : cks) {
      float base = *c.d;
      auto f = [&](float v) { *c.d = v; Tensor E2 = from_data({B, D}, ev, true), W2 = from_data({C, D}, wv, true);
        Tensor e2 = l2norm_rows(E2); Tensor l2 = mk(e2, W2); return (double)l2->data[0]; };
      double num = (f(base + eps) - f(base - eps)) / (2 * eps); *c.d = base;
      double rel = std::fabs(num - c.ana) / (std::fabs(num) + std::fabs(c.ana) + 1e-9);
      bool good = rel < 1e-2 || (std::fabs(c.ana) < 1e-6 && std::fabs(num) < 1e-6); ok += good;
      printf("   %-6s ana=% .5e num=% .5e rel=%.2e %s\n", c.n, c.ana, num, rel, good ? "ok" : "BAD");
    }
    return ok == (int)cks.size();
  };

  int pass = 0, tot = 0;
  tot++; pass += run("triplet(batch-all)", [&](Tensor& e, Tensor&) { return triplet_loss(e, lab, 0.5f, false); }, false);
  tot++; pass += run("triplet(batch-hard)", [&](Tensor& e, Tensor&) { return triplet_loss(e, lab, 0.5f, true); }, false);
  tot++; pass += run("arcface", [&](Tensor& e, Tensor& W) { return arcface_loss(e, W, lab, 0.5f, 16.f); }, true);
  tot++; pass += run("softmax", [&](Tensor& e, Tensor& W) { return softmax_ce_loss(e, W, lab, 16.f); }, true);
  printf("%d/%d loss gradchecks pass\n", pass, tot);
  return pass == tot ? 0 : 1;
}
