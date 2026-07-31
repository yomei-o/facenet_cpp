// Numerical gradient check for the full unfused InceptionResnetV1 backward (train mode, so BN-train,
// conv, relu, pad, concat, gap, linear, l2norm backward are all exercised). Scalar loss L = sum(emb).
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\gradcheck_face.cpp
#include "net_facenet_unfused.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <functional>
#include <vector>

static std::vector<float> readbin(const std::string& p, int64_t n) {
  std::ifstream f(p, std::ios::binary); std::vector<float> v(n);
  if (!f) { printf("missing %s\n", p.c_str()); std::exit(1); } f.read((char*)v.data(), n * 4); return v;
}

int main(int argc, char** argv) {
  std::string DN = argc > 1 ? argv[1] : "pure/ref/data_net/", RF = argc > 2 ? argv[2] : "pure/ref/";
  int64_t S = argc > 3 ? atoll(argv[3]) : 160; if (DN.back() != '/') DN += '/'; if (RF.back() != '/') RF += '/';
  UProv p = load_facenet_unfused(DN);
  auto xin = readbin(RF + "input.bin", 3 * S * S);

  auto loss = [&]() { p.i = 0; Tensor xt = from_data({1, 3, S, S}, xin, true);
                      Tensor e = facenet_forward_u(xt, p, false); double s = 0;
                      for (int i = 0; i < 512; ++i) s += e->data[i]; return std::pair<double, Tensor>{s, xt}; };

  // analytic grads from one backward
  p.i = 0; Tensor xt = from_data({1, 3, S, S}, xin, true);
  Tensor e = facenet_forward_u(xt, p, false); Tensor L = sum(e); backward(L);

  ULayer& B0 = p.L[0]; ULayer* LL = nullptr; for (auto& l : p.L) if (l.kind == 2) LL = &l;
  struct C { const char* n; std::function<void(float)> set; std::function<float()> get; float ana; };
  std::vector<C> chks = {
    {"input[100]",     [&](float v){ xin[100] = v; },        [&]{ return xin[100]; },        xt->grad[100]},
    {"input[5000]",    [&](float v){ xin[5000] = v; },       [&]{ return xin[5000]; },       xt->grad[5000]},
    {"conv1a.w[10]",   [&](float v){ B0.w->data[10] = v; },  [&]{ return B0.w->data[10]; },  B0.w->grad[10]},
    {"conv1a.w[500]",  [&](float v){ B0.w->data[500] = v; }, [&]{ return B0.w->data[500]; }, B0.w->grad[500]},
    {"conv1a.gamma[5]",[&](float v){ B0.gamma->data[5]=v; }, [&]{ return B0.gamma->data[5];},B0.gamma->grad[5]},
    {"lastlin.w[77]",  [&](float v){ LL->w->data[77] = v; }, [&]{ return LL->w->data[77]; }, LL->w->grad[77]},
  };

  const float eps = 1e-3f; int ok = 0;
  printf("gradcheck (train mode, L=sum(emb), 2-sided eps=%.0e):\n", eps);
  printf("  %-16s %13s %13s %10s\n", "param", "analytic", "numeric", "rel.err");
  for (auto& c : chks) {
    float base = c.get();
    c.set(base + eps); double Lp = loss().first;
    c.set(base - eps); double Lm = loss().first;
    c.set(base);
    double num = (Lp - Lm) / (2 * eps);
    double rel = std::fabs(num - c.ana) / (std::fabs(num) + std::fabs(c.ana) + 1e-9);
    bool good = rel < 3e-2 || (std::fabs(c.ana) < 1e-6 && std::fabs(num) < 1e-6);  // fp32 2-sided FD on a 27.9M-param net is ~1-3% noisy
    ok += good;
    printf("  %-16s % .5e % .5e %9.2e %s\n", c.n, c.ana, num, rel, good ? "ok" : "BAD");
  }
  printf("%d/%zu gradients match\n", ok, chks.size());
  return ok == (int)chks.size() ? 0 : 1;
}
