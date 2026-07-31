// Parity: unfused (BN eval-mode, running stats) forward vs facenet-pytorch reference embedding.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m6_unfused_face.cpp
#include "net_facenet_unfused.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>

static std::vector<float> readbin(const std::string& p, int64_t n) {
  std::ifstream f(p, std::ios::binary); std::vector<float> v(n);
  if (!f) { printf("missing %s\n", p.c_str()); std::exit(1); } f.read((char*)v.data(), n * 4); return v;
}
int main(int argc, char** argv) {
  std::string DN = argc > 1 ? argv[1] : "pure/ref/data_net/", RF = argc > 2 ? argv[2] : "pure/ref/";
  int64_t S = argc > 3 ? atoll(argv[3]) : 160;
  if (DN.back() != '/') DN += '/'; if (RF.back() != '/') RF += '/';
  UProv p = load_facenet_unfused(DN);
  auto xin = readbin(RF + "input.bin", 3 * S * S), ref = readbin(RF + "embed.bin", 512);
  Tensor e = facenet_forward_u(from_data({1, 3, S, S}, xin), p, false);   // eval mode
  float worst = 0; for (int i = 0; i < 512; ++i) worst = std::max(worst, std::fabs(e->data[i] - ref[i]));
  printf("facenet unfused(eval) forward: worst |pure - facenet-pytorch| = %.3e  %s\n",
         worst, worst < 1e-3f ? "MATCH" : "MISMATCH");
  return 0;
}
