// FaceNet inference CLI: embed a face, 1:1 verify (same/different), or 1:N identify against a
// gallery. Uses the unfused net in eval mode so it loads the pretrained weights (data_net) or a
// fine-tuned checkpoint (--ckpt from train_face). Images are standardized+resized like training.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\verify_face.cpp
//   run:   verify_face embed    <img> [--ckpt f] [--datanet d] [--imgsz 160]
//          verify_face verify   <imgA> <imgB> [--thr 0.5] ...
//          verify_face identify <probe> <gallery_dir> ...
#define STB_IMAGE_IMPLEMENTATION
#include "face_data.hpp"
#include "net_facenet_unfused.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <map>

static std::vector<float> embed(const std::string& path, UProv& p, int64_t S) {
  auto v = load_face(path, S); p.i = 0;
  Tensor e = facenet_forward_u(from_data({1, 3, S, S}, v), p, false);   // eval mode
  std::vector<float> out(e->data.begin(), e->data.begin() + 512); return out;
}
static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0; for (int i = 0; i < 512; ++i) s += (double)a[i] * b[i]; return (float)s;   // unit vectors
}
static std::string opt(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 1; i < argc - 1; ++i) if (key == argv[i]) return argv[i + 1]; return def;
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: verify_face <embed|verify|identify> <args> [--ckpt f --datanet d --imgsz 160 --thr 0.5]\n"); return 1; }
  std::string mode = argv[1];
  std::string DN = opt(argc, argv, "--datanet", "pure/ref/data_net/"); if (DN.back() != '/') DN += '/';
  std::string ckpt = opt(argc, argv, "--ckpt", "");
  int64_t S = (int64_t)atoll(opt(argc, argv, "--imgsz", "160").c_str());
  float thr = (float)atof(opt(argc, argv, "--thr", "0.48").c_str());

  UProv p = load_facenet_unfused(DN);
  if (!ckpt.empty()) { load_facenet_weights(p, ckpt); printf("loaded checkpoint %s\n", ckpt.c_str()); }

  if (mode == "embed") {
    auto e = embed(argv[2], p, S);
    printf("embedding[512], L2=%.4f, [:8]=", std::sqrt(cosine(e, e)));
    for (int i = 0; i < 8; ++i) printf("% .4f ", e[i]); printf("\n");
  } else if (mode == "verify") {
    if (argc < 4) { printf("verify needs <imgA> <imgB>\n"); return 1; }
    auto ea = embed(argv[2], p, S), eb = embed(argv[3], p, S);
    float cs = cosine(ea, eb), dist = std::sqrt(std::max(0.f, 2 - 2 * cs));
    printf("cosine=%.4f  euclidean=%.4f  -> %s (thr cos=%.2f)\n", cs, dist, cs >= thr ? "SAME" : "DIFFERENT", thr);
  } else if (mode == "identify") {
    if (argc < 4) { printf("identify needs <probe> <gallery_dir>\n"); return 1; }
    auto ep = embed(argv[2], p, S);
    FaceDS g = scan_faces(argv[3]);
    std::vector<std::vector<float>> cen(g.num_classes, std::vector<float>(512, 0.f)); std::vector<int> cnt(g.num_classes, 0);
    for (size_t i = 0; i < g.paths.size(); ++i) { auto e = embed(g.paths[i], p, S); int c = g.label[i];
      for (int d = 0; d < 512; ++d) cen[c][d] += e[d]; cnt[c]++; }
    int best = -1; float bs = -1e9f;
    for (int c = 0; c < g.num_classes; ++c) { float n = 0; for (int d = 0; d < 512; ++d) n += cen[c][d] * cen[c][d];
      n = std::sqrt(n); for (int d = 0; d < 512; ++d) cen[c][d] /= (n + 1e-9f);
      float s = cosine(ep, cen[c]); printf("  %-16s cosine=%.4f\n", g.names[c].c_str(), s);
      if (s > bs) { bs = s; best = c; } }
    printf("-> best match: %s (cosine=%.4f)\n", best >= 0 ? g.names[best].c_str() : "?", bs);
  } else { printf("unknown mode %s\n", mode.c_str()); return 1; }
  return 0;
}
