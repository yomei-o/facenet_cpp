// Unified FaceNet CLI: train / embed / verify / identify in one binary (pure C++, CPU).
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\facenet.cpp
//   facenet train    <dataset_dir> <triplet|arcface|softmax> <steps> [--P 3 --K 4 --imgsz 160 --lr 1e-4 --ckpt out.bin]
//   facenet embed    <img>                 [--ckpt f --datanet d --imgsz 160]
//   facenet verify   <imgA> <imgB>         [--thr 0.5 ...]
//   facenet identify <probe> <gallery_dir> [...]
#define STB_IMAGE_IMPLEMENTATION
#include "face_data.hpp"
#include "net_facenet_unfused.hpp"
#include "face_loss.hpp"
#include "optim.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>
#include <string>

static std::string opt(int c, char** v, const std::string& k, const std::string& d) {
  for (int i = 1; i < c - 1; ++i) if (k == v[i]) return v[i + 1]; return d;
}
static std::vector<float> embed(const std::string& path, UProv& p, int64_t S) {
  auto v = load_face(path, S); Tensor e = facenet_forward_u(from_data({1, 3, S, S}, v), p, false);
  return std::vector<float>(e->data.begin(), e->data.begin() + 512);
}
static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0; for (int i = 0; i < 512; ++i) s += (double)a[i] * b[i]; return (float)s;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) { printf("usage: facenet <train|embed|verify|identify> ... (see header)\n"); return 1; }
  std::string cmd = argv[1];
  std::string DN = opt(argc, argv, "--datanet", "pure/ref/data_net/"); if (DN.back() != '/') DN += '/';
  int64_t S = (int64_t)atoll(opt(argc, argv, "--imgsz", "160").c_str());
  std::string ckpt = opt(argc, argv, "--ckpt", "");

  UProv p = load_facenet_unfused(DN);

  if (cmd == "train") {
    if (argc < 5) { printf("train <dataset_dir> <triplet|arcface|softmax> <steps> [--P --K --lr --ckpt]\n"); return 1; }
    std::string dir = argv[2], loss = argv[3]; int steps = atoi(argv[4]);
    int P = atoi(opt(argc, argv, "--P", "3").c_str()), K = atoi(opt(argc, argv, "--K", "4").c_str());
    float lr = (float)atof(opt(argc, argv, "--lr", "1e-4").c_str());
    std::string out = ckpt.empty() ? "facenet_ft.bin" : ckpt;
    FaceDS ds = scan_faces(dir);
    printf("dataset: %zu images, %d identities\n", ds.paths.size(), ds.num_classes);
    if (ds.num_classes < 2) { printf("need >=2 identities\n"); return 1; }
    std::vector<Tensor> params = facenet_params_u(p); Tensor W;
    if (loss != "triplet") { std::mt19937 r(0); std::normal_distribution<float> nd(0, 0.01f);
      std::vector<float> wv(ds.num_classes * 512); for (auto& x : wv) x = nd(r);
      W = from_data({ds.num_classes, 512}, wv, true); params.push_back(W); }
    Adam opt_(params, lr);
    printf("train loss=%s steps=%d batch=%dx%d lr=%.0e\n", loss.c_str(), steps, P, K, lr);
    std::mt19937 rng(1234); double avg = 0;
    for (int it = 1; it <= steps; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      FBatch b = sample_pk(ds, P, K, S, rng); opt_.zero_grad();
      Tensor emb = facenet_forward_u(b.x, p, true), L;
      if (loss == "triplet")      L = triplet_loss(emb, b.lab, 0.2f, true);
      else if (loss == "arcface") L = arcface_loss(emb, W, b.lab, 0.5f, 16.f);
      else                        L = softmax_ce_loss(emb, W, b.lab, 16.f);
      backward(L); opt_.step();
      double sc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      avg = it == 1 ? L->data[0] : 0.9 * avg + 0.1 * L->data[0];
      if (it % 5 == 0 || it == 1) printf("  step %4d/%d loss %.4f (avg %.4f) %.2f s/step\n", it, steps, L->data[0], avg, sc);
      free_graph(L);
    }
    save_facenet_unfused(p, out); printf("saved -> %s\n", out.c_str()); return 0;
  }

  if (!ckpt.empty()) { load_facenet_weights(p, ckpt); printf("loaded checkpoint %s\n", ckpt.c_str()); }

  if (cmd == "embed") {
    auto e = embed(argv[2], p, S); printf("embedding[512] L2=%.4f [:8]=", std::sqrt(cosine(e, e)));
    for (int i = 0; i < 8; ++i) printf("% .4f ", e[i]); printf("\n");
  } else if (cmd == "verify") {
    float thr = (float)atof(opt(argc, argv, "--thr", "0.5").c_str());
    auto a = embed(argv[2], p, S), b = embed(argv[3], p, S); float cs = cosine(a, b);
    printf("cosine=%.4f euclidean=%.4f -> %s (thr cos=%.2f)\n", cs, std::sqrt(std::max(0.f, 2 - 2 * cs)), cs >= thr ? "SAME" : "DIFFERENT", thr);
  } else if (cmd == "identify") {
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
