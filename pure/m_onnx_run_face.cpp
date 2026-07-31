// Pure-C++ round-trip: load facenet.onnx and run it on the shared engine; compare to the
// facenet-pytorch reference embedding.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m_onnx_run_face.cpp
//   run:   m_onnx_run_face [facenet.onnx] [ref_dir] [imgsz]
#include "onnx_run_face.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>

static std::vector<float> readbin(const std::string& p, int64_t n) {
  std::ifstream f(p, std::ios::binary); std::vector<float> v(n);
  if (!f) { printf("missing %s\n", p.c_str()); std::exit(1); } f.read((char*)v.data(), n * 4); return v;
}
int main(int argc, char** argv) {
  std::string onnx = argc > 1 ? argv[1] : "facenet.onnx", RF = argc > 2 ? argv[2] : "pure/ref/";
  int64_t S = argc > 3 ? atoll(argv[3]) : 160; if (RF.back() != '/') RF += '/';
  onx::Graph g = onx::load_onnx(onnx);
  auto xin = readbin(RF + "input.bin", 3 * S * S), ref = readbin(RF + "embed.bin", 512);
  auto out = onx::run_onnx_face(g, from_data({1, 3, S, S}, xin));
  Tensor e = out.at("embedding");
  float worst = 0; for (int i = 0; i < 512; ++i) worst = std::max(worst, std::fabs(e->data[i] - ref[i]));
  printf("pure-C++ ONNX run vs facenet-pytorch: worst |diff| = %.3e  %s\n", worst, worst < 1e-3f ? "MATCH" : "MISMATCH");
  return 0;
}
