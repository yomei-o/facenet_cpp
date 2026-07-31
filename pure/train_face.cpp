// FaceNet training (CPU): fine-tune the pretrained InceptionResnetV1 embedding on a folder-per-
// identity dataset with Triplet / ArcFace / cosine-softmax loss. Saves a checkpoint reloadable by
// the unfused net (and thus by verify_face / detect).
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_face.cpp
//   run:   train_face <dataset_dir> <triplet|arcface|softmax> <steps> [P K imgsz lr datanet ckpt]
#define STB_IMAGE_IMPLEMENTATION
#include "face_data.hpp"
#include "net_facenet_unfused.hpp"
#include "face_loss.hpp"
#include "optim.hpp"
#include <cstdio>
#include <chrono>
#include <random>

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) { printf("usage: train_face <dataset_dir> <triplet|arcface|softmax> <steps> [P K imgsz lr datanet ckpt]\n"); return 1; }
  std::string dir = argv[1], loss = argv[2];
  int steps = argc > 3 ? atoi(argv[3]) : 100;
  int P = argc > 4 ? atoi(argv[4]) : 3, K = argc > 5 ? atoi(argv[5]) : 4;
  int64_t S = argc > 6 ? atoll(argv[6]) : 160;
  float lr = argc > 7 ? (float)atof(argv[7]) : 1e-4f;
  std::string DN = argc > 8 ? argv[8] : "pure/ref/data_net/"; if (DN.back() != '/') DN += '/';
  std::string ckpt = argc > 9 ? argv[9] : "facenet_ft.bin";

  FaceDS ds = scan_faces(dir);
  printf("dataset: %zu images, %d identities (%s ...)\n", ds.paths.size(), ds.num_classes,
         ds.names.empty() ? "" : ds.names[0].c_str());
  if (ds.num_classes < 2) { printf("need >=2 identities\n"); return 1; }

  UProv prov = load_facenet_unfused(DN);
  std::vector<Tensor> params = facenet_params_u(prov);
  Tensor W;                                              // ArcFace/softmax class centers
  if (loss != "triplet") { std::mt19937 r(0); std::normal_distribution<float> nd(0, 0.01f);
    std::vector<float> wv(ds.num_classes * 512); for (auto& v : wv) v = nd(r);
    W = from_data({ds.num_classes, 512}, wv, true); params.push_back(W); }
  Adam opt(params, lr);
  printf("training: loss=%s steps=%d batch=%dx%d=%d imgsz=%lld lr=%.0e params=%zu\n",
         loss.c_str(), steps, P, K, P * K, (long long)S, lr, params.size());

  std::mt19937 rng(1234); double ravg = 0;
  for (int it = 1; it <= steps; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    FBatch b = sample_pk(ds, P, K, S, rng);
    opt.zero_grad();
    Tensor emb = facenet_forward_u(b.x, prov, true);     // train mode (batch>1 -> BN ok)
    Tensor L;
    if (loss == "triplet")      L = triplet_loss(emb, b.lab, 0.2f, true);   // batch-hard
    else if (loss == "arcface") L = arcface_loss(emb, W, b.lab, 0.5f, 16.f);
    else                        L = softmax_ce_loss(emb, W, b.lab, 16.f);
    backward(L); opt.step();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ravg = it == 1 ? L->data[0] : 0.9 * ravg + 0.1 * L->data[0];
    if (it % 5 == 0 || it == 1) printf("  step %4d/%d  loss %.4f  (avg %.4f)  %.2f s/step\n", it, steps, L->data[0], ravg, secs);
    free_graph(L);
  }
  save_facenet_unfused(prov, ckpt);
  printf("saved checkpoint -> %s  (reload with load_facenet_weights)\n", ckpt.c_str());
  return 0;
}
