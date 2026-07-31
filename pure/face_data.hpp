// Face dataset loader: folder-per-identity (root/<person>/<img>.jpg). Loads images, bilinear-resizes
// to S×S, applies facenet standardization (x-127.5)/128, CHW. Samples P-identities × K-images batches
// (for triplet) or plain labeled batches (for ArcFace/softmax). Requires STB_IMAGE_IMPLEMENTATION in
// the including .cpp.
#pragma once
#include "autograd.hpp"
#include "stb_image.h"
#include <string>
#include <vector>
#include <filesystem>
#include <random>
#include <algorithm>

struct FaceDS { std::vector<std::string> paths; std::vector<int> label; int num_classes = 0;
                std::vector<std::vector<int>> by_class; std::vector<std::string> names; };

inline FaceDS scan_faces(const std::string& root) {
  namespace fs = std::filesystem; FaceDS ds;
  std::vector<fs::path> dirs;
  for (auto& e : fs::directory_iterator(root)) if (e.is_directory()) dirs.push_back(e.path());
  std::sort(dirs.begin(), dirs.end());
  for (auto& d : dirs) {
    int cls = ds.num_classes; bool any = false;
    for (auto& f : fs::directory_iterator(d)) {
      auto ext = f.path().extension().string(); for (auto& ch : ext) ch = (char)tolower(ch);
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
        ds.paths.push_back(f.path().string()); ds.label.push_back(cls); any = true;
      }
    }
    if (any) { ds.by_class.push_back({}); ds.names.push_back(d.filename().string()); ds.num_classes++; }
  }
  for (int i = 0; i < (int)ds.label.size(); ++i) ds.by_class[ds.label[i]].push_back(i);
  return ds;
}

// load one image -> [3*S*S] float, RGB, bilinear-resized, standardized (x-127.5)/128.
inline std::vector<float> load_face(const std::string& path, int64_t S) {
  int w, h, c; unsigned char* im = stbi_load(path.c_str(), &w, &h, &c, 3);
  std::vector<float> out(3 * S * S, 0.f);
  if (!im) { printf("warn: cannot load %s (using zeros)\n", path.c_str()); return out; }
  for (int64_t y = 0; y < S; ++y) for (int64_t x = 0; x < S; ++x) {
    float sy = (y + 0.5f) * h / S - 0.5f, sx = (x + 0.5f) * w / S - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    auto px = [&](int yy, int xx, int ch) { yy = std::clamp(yy, 0, h - 1); xx = std::clamp(xx, 0, w - 1);
      return (float)im[(yy * w + xx) * 3 + ch]; };
    for (int ch = 0; ch < 3; ++ch) {
      float v = px(y0, x0, ch) * (1 - fx) * (1 - fy) + px(y0, x0 + 1, ch) * fx * (1 - fy)
              + px(y0 + 1, x0, ch) * (1 - fx) * fy + px(y0 + 1, x0 + 1, ch) * fx * fy;
      out[(ch * S + y) * S + x] = (v - 127.5f) / 128.f;
    }
  }
  stbi_image_free(im); return out;
}

struct FBatch { Tensor x; std::vector<int> lab; };

// P identities × K images (triplet-friendly); falls back gracefully if a class has < K images.
inline FBatch sample_pk(const FaceDS& ds, int P, int K, int64_t S, std::mt19937& rng) {
  std::vector<int> cls(ds.num_classes); for (int i = 0; i < ds.num_classes; ++i) cls[i] = i;
  std::shuffle(cls.begin(), cls.end(), rng); if (P > (int)cls.size()) P = (int)cls.size();
  std::vector<int> idx; std::vector<int> lab;
  for (int pi = 0; pi < P; ++pi) { auto& pool = ds.by_class[cls[pi]];
    for (int k = 0; k < K; ++k) { idx.push_back(pool[rng() % pool.size()]); lab.push_back(cls[pi]); } }
  int B = (int)idx.size(); std::vector<float> xb(B * 3 * S * S);
  for (int b = 0; b < B; ++b) { auto v = load_face(ds.paths[idx[b]], S); std::copy(v.begin(), v.end(), xb.begin() + b * 3 * S * S); }
  return {from_data({B, 3, S, S}, xb), lab};
}
