// Device (Thrust) facenet eval forward parity vs the facenet-pytorch reference embedding.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dface_test.cpp
//   GPU (Colab nvcc): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party pure/dface_test.cpp -o dface
#include "dface.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>

static std::vector<float> readbin(const std::string& p, int64_t n) {
  std::ifstream f(p, std::ios::binary); std::vector<float> v(n);
  if (!f) { printf("missing %s\n", p.c_str()); std::exit(1); } f.read((char*)v.data(), n * 4); return v;
}
int main(int argc, char** argv) {
  std::string DN = argc > 1 ? argv[1] : "pure/ref/data_net/", RF = argc > 2 ? argv[2] : "pure/ref/";
  int64_t S = argc > 3 ? atoll(argv[3]) : 160; if (DN.back() != '/') DN += '/'; if (RF.back() != '/') RF += '/';
  DFaceProv p = dface_build(DN);
  auto xin = readbin(RF + "input.bin", 3 * S * S), ref = readbin(RF + "embed.bin", 512);
  DT e = dface_forward(dfrom({1, 3, S, S}, xin), p); bk::sync();
  auto host = dto_host(e);
  float worst = 0; double l2 = 0; for (int i = 0; i < 512; ++i) { worst = std::max(worst, std::fabs(host[i] - ref[i])); l2 += (double)host[i] * host[i]; }
  printf("device facenet eval forward: L2norm=%.6f worst |device - facenet-pytorch| = %.3e  %s\n",
         std::sqrt(l2), worst, worst < 1e-3f ? "MATCH" : "MISMATCH");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host, THRUST_DEVICE_SYSTEM_CPP)\n");
#endif
  return 0;
}
