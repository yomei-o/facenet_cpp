// Self-contained FaceNet demo: loads the bundled fp16 fused weights (weights/facenet/) — no Python,
// no export, no download — and runs face embedding / 1:1 verify / 1:N identify. Bring your own
// (roughly cropped) face images; any size is resized to 160 and standardized.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\demo_face.cpp
//   demo_face verify   <imgA> <imgB>          [--thr 0.5 --weights weights/facenet/ --imgsz 160]
//   demo_face identify <probe> <gallery_dir>  [...]
//   demo_face embed    <img>                  [...]
//   demo_face selftest                        (checks fp16 weights vs the fp32 reference embedding)
#define STB_IMAGE_IMPLEMENTATION
#include "face_data.hpp"
#include "net_facenet.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>

static std::string opt(int c, char** v, const std::string& k, const std::string& d) {
  for (int i = 1; i < c - 1; ++i) if (k == v[i]) return v[i + 1]; return d;
}
static std::vector<float> embed(const std::string& path, FaceProv& p, int64_t S) {
  auto v = load_face(path, S); Tensor e = facenet_forward(from_data({1, 3, S, S}, v), p);
  return std::vector<float>(e->data.begin(), e->data.begin() + 512);
}
static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0; for (int i = 0; i < 512; ++i) s += (double)a[i] * b[i]; return (float)s;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: demo_face <verify|identify|embed|selftest> ...  (bundled weights, no Python)\n"); return 1; }
  std::string cmd = argv[1];
  std::string WD = opt(argc, argv, "--weights", "weights/facenet/"); if (WD.back() != '/') WD += '/';
  int64_t S = (int64_t)atoll(opt(argc, argv, "--imgsz", "160").c_str());
  float thr = (float)atof(opt(argc, argv, "--thr", "0.5").c_str());

  FaceProv p = load_facenet_fp16(WD);
  printf("loaded fp16 weights from %s\n", WD.c_str());

  if (cmd == "selftest") {
    std::string RF = opt(argc, argv, "--ref", "pure/ref/"); if (RF.back() != '/') RF += '/';
    std::ifstream fi(RF + "input.bin", std::ios::binary), fe(RF + "embed.bin", std::ios::binary);
    if (!fi || !fe) { printf("selftest needs %sinput.bin + embed.bin (run export_facenet.py)\n", RF.c_str()); return 1; }
    std::vector<float> xin(3 * S * S), ref(512); fi.read((char*)xin.data(), xin.size() * 4); fe.read((char*)ref.data(), 512 * 4);
    Tensor e = facenet_forward(from_data({1, 3, S, S}, xin), p);
    float worst = 0; for (int i = 0; i < 512; ++i) worst = std::max(worst, std::fabs(e->data[i] - ref[i]));
    printf("fp16 demo vs fp32 facenet-pytorch ref: worst=%.3e  %s\n", worst, worst < 5e-3f ? "OK" : "HIGH");
    return 0;
  }
  if (cmd == "embed") {
    auto e = embed(argv[2], p, S); printf("embedding[512] L2=%.4f [:8]=", std::sqrt(cosine(e, e)));
    for (int i = 0; i < 8; ++i) printf("% .4f ", e[i]); printf("\n");
  } else if (cmd == "verify") {
    if (argc < 4) { printf("verify needs <imgA> <imgB>\n"); return 1; }
    auto a = embed(argv[2], p, S), b = embed(argv[3], p, S); float cs = cosine(a, b);
    printf("cosine=%.4f euclidean=%.4f -> %s (thr cos=%.2f)\n", cs, std::sqrt(std::max(0.f, 2 - 2 * cs)), cs >= thr ? "SAME" : "DIFFERENT", thr);
  } else if (cmd == "identify") {
    if (argc < 4) { printf("identify needs <probe> <gallery_dir>\n"); return 1; }
    auto ep = embed(argv[2], p, S); FaceDS g = scan_faces(argv[3]);
    std::vector<std::vector<float>> cen(g.num_classes, std::vector<float>(512, 0.f));
    for (size_t i = 0; i < g.paths.size(); ++i) { auto e = embed(g.paths[i], p, S); for (int d = 0; d < 512; ++d) cen[g.label[i]][d] += e[d]; }
    int best = -1; float bs = -1e9f;
    for (int c = 0; c < g.num_classes; ++c) { float n = 0; for (int d = 0; d < 512; ++d) n += cen[c][d] * cen[c][d]; n = std::sqrt(n) + 1e-9f;
      for (int d = 0; d < 512; ++d) cen[c][d] /= n; float s = cosine(ep, cen[c]);
      printf("  %-16s cosine=%.4f\n", g.names[c].c_str(), s); if (s > bs) { bs = s; best = c; } }
    printf("-> best: %s (cosine=%.4f)\n", best >= 0 ? g.names[best].c_str() : "?", bs);
  } else { printf("unknown command %s\n", cmd.c_str()); return 1; }
  return 0;
}
