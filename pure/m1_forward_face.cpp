// Parity: pure-C++ fused InceptionResnetV1 forward vs facenet-pytorch reference embedding.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m1_forward_face.cpp
//   run:   m1_forward_face [data_net_dir] [ref_dir] [imgsz]
#include "net_facenet.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>

static std::vector<float> readbin(const std::string& p, int64_t n) {
  std::ifstream f(p, std::ios::binary); std::vector<float> v(n);
  if (!f) { printf("missing %s\n", p.c_str()); std::exit(1); }
  f.read((char*)v.data(), n * 4); return v;
}

int main(int argc, char** argv) {
  std::string DN = argc > 1 ? argv[1] : "pure/ref/data_net/";
  std::string RF = argc > 2 ? argv[2] : "pure/ref/";
  int64_t S = argc > 3 ? atoll(argv[3]) : 160;
  if (!DN.empty() && DN.back() != '/') DN += '/';
  if (!RF.empty() && RF.back() != '/') RF += '/';

  FaceProv p = load_facenet(DN);
  auto xin = readbin(RF + "input.bin", 1 * 3 * S * S);
  auto ref = readbin(RF + "embed.bin", 512);

  Tensor x = from_data({1, 3, S, S}, xin);
  Tensor e = facenet_forward(x, p);

  float worst = 0; double l2 = 0;
  for (int i = 0; i < 512; ++i) { worst = std::max(worst, std::fabs(e->data[i] - ref[i])); l2 += (double)e->data[i] * e->data[i]; }
  printf("facenet forward: emb L2norm=%.6f  worst |pure - facenet-pytorch| = %.3e  %s\n",
         std::sqrt(l2), worst, worst < 1e-3f ? "MATCH" : "MISMATCH");
  printf("  pure[:6] = "); for (int i = 0; i < 6; ++i) printf("% .6f ", e->data[i]); printf("\n");
  printf("  ref [:6] = "); for (int i = 0; i < 6; ++i) printf("% .6f ", ref[i]); printf("\n");
  return 0;
}
