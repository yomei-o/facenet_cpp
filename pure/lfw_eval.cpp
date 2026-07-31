// LFW verification eval for the pure-C++ facenet embedding + cosine-threshold calibration.
// Reads a pairs list (one "pathA pathB label" per line; label 1=same, 0=different), embeds both
// faces, and sweeps the cosine threshold to report the best accuracy + the calibrated threshold.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\lfw_eval.cpp
//   run:   lfw_eval <pairs.txt> [--datanet d --imgsz 160 --fp16 weights/facenet/]
#define STB_IMAGE_IMPLEMENTATION
#include "face_data.hpp"
#include "net_facenet.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

static std::string opt(int c, char** v, const std::string& k, const std::string& d) {
  for (int i = 1; i < c - 1; ++i) if (k == v[i]) return v[i + 1]; return d;
}
static std::vector<float> embed(const std::string& path, FaceProv& p, int64_t S) {
  auto v = load_face(path, S); Tensor e = facenet_forward(from_data({1, 3, S, S}, v), p);
  return std::vector<float>(e->data.begin(), e->data.begin() + 512);
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) { printf("usage: lfw_eval <pairs.txt> [--datanet d --fp16 dir --imgsz 160]\n"); return 1; }
  std::string pairs = argv[1];
  std::string DN = opt(argc, argv, "--datanet", "pure/ref/data_net/"); if (DN.back() != '/') DN += '/';
  std::string fp16 = opt(argc, argv, "--fp16", "");
  int64_t S = (int64_t)atoll(opt(argc, argv, "--imgsz", "160").c_str());

  FaceProv p = fp16.empty() ? load_facenet(DN) : load_facenet_fp16(fp16);
  printf("weights: %s\n", fp16.empty() ? DN.c_str() : fp16.c_str());

  std::ifstream f(pairs); if (!f) { printf("cannot open %s\n", pairs.c_str()); return 1; }
  std::vector<std::pair<float, int>> scored; std::string line; int n = 0;
  double smean = 0, dmean = 0; int sc = 0, dc = 0;
  while (std::getline(f, line)) {
    if (line.empty()) continue; std::istringstream ss(line); std::string a, b; int lab;
    if (!(ss >> a >> b >> lab)) continue;
    float cs = 0; { auto ea = embed(a, p, S), eb = embed(b, p, S); for (int i = 0; i < 512; ++i) cs += ea[i] * eb[i]; }
    scored.push_back({cs, lab}); if (lab) { smean += cs; ++sc; } else { dmean += cs; ++dc; }
    if (++n % 50 == 0) printf("  ...%d pairs\n", n);
  }
  if (scored.empty()) { printf("no pairs\n"); return 1; }

  // sweep cosine threshold for best accuracy
  float bestT = 0; double bestAcc = 0;
  for (float t = -1.f; t <= 1.f; t += 0.002f) {
    int ok = 0; for (auto& s : scored) ok += ((s.first >= t) == (s.second == 1));
    double acc = (double)ok / scored.size(); if (acc > bestAcc) { bestAcc = acc; bestT = t; }
  }
  printf("\nLFW eval: %zu pairs (%d same, %d diff)\n", scored.size(), sc, dc);
  printf("  mean cosine  same=%.4f  diff=%.4f\n", sc ? smean / sc : 0, dc ? dmean / dc : 0);
  printf("  best threshold cos>=%.3f  ->  accuracy %.2f%%\n", bestT, 100 * bestAcc);
  return 0;
}
