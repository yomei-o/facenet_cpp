// FaceNet training losses on unit embeddings emb[B,D]:
//  - triplet_loss   (FaceNet paper): batch-all / batch-hard, margin on ‖a-p‖²-‖a-n‖².
//  - arcface_loss   (additive angular margin softmax) + softmax_ce_loss (cosine softmax, m=0).
// Built on the shared autograd so grads flow to emb (and to the class weights W for ArcFace).
#pragma once
#include "autograd.hpp"
#include "face_ops.hpp"      // l2norm_rows
#include "linalg.hpp"        // matmul, transpose2d
#include "ops2d.hpp"         // mul_scalar, log_softmax_rows, gather_row
#include <vector>
#include <cmath>
#include <memory>

// Gram G[B,B] = emb·embᵀ (rows unit → cosine similarities).
inline Tensor gram(const Tensor& emb) { return matmul(emb, transpose2d(emb)); }

// Triplet loss from unit embeddings. Dᵢⱼ = 2−2Gᵢⱼ, term = margin + D_ap − D_an = margin −2G_ap +2G_an.
// batch-all: mean over all violating (a,p,n); batch-hard: hardest negative per (a,p).
inline Tensor triplet_loss(const Tensor& emb, const std::vector<int>& lab, float margin = 0.2f, bool hard = false) {
  Tensor G = gram(emb); int64_t B = emb->shape[0]; const float* g = G->data.data();
  struct Tri { int a, p, n; }; auto tris = std::make_shared<std::vector<Tri>>();
  double loss = 0; int cnt = 0;
  for (int a = 0; a < B; ++a) for (int p = 0; p < B; ++p) {
    if (p == a || lab[p] != lab[a]) continue;
    if (hard) {
      int bn = -1; float bv = -1e30f;
      for (int n = 0; n < B; ++n) if (lab[n] != lab[a] && g[a * B + n] > bv) { bv = g[a * B + n]; bn = n; }
      if (bn >= 0) { float t = margin - 2 * g[a * B + p] + 2 * g[a * B + bn]; if (t > 0) { loss += t; tris->push_back({a, p, bn}); ++cnt; } }
    } else {
      for (int n = 0; n < B; ++n) { if (lab[n] == lab[a]) continue;
        float t = margin - 2 * g[a * B + p] + 2 * g[a * B + n]; if (t > 0) { loss += t; tris->push_back({a, p, n}); ++cnt; } }
    }
  }
  int denom = cnt > 0 ? cnt : 1;
  auto o = make_tensor({1}, true); o->data[0] = (float)(loss / denom);
  o->parents = {G}; Node* op = o.get();
  o->backward_fn = [G, op, tris, denom, B] { float go = op->grad[0] / denom;
    for (auto& t : *tris) { G->grad[t.a * B + t.p] += -2 * go; G->grad[t.a * B + t.n] += 2 * go; } };
  return o;
}

// cosine logits [B,C] = unit-emb · unit-Wⱼ (W = class centers [C,D]).
inline Tensor cosine_logits(const Tensor& emb, const Tensor& W) { return matmul(emb, transpose2d(l2norm_rows(W))); }

// ArcFace: add angular margin m to the TARGET class, scale by s; returns logits[B,C] for CE.
inline Tensor arcface_margin(const Tensor& cos, const std::vector<int>& lab, float m = 0.5f, float s = 64.f) {
  int64_t B = cos->shape[0], C = cos->shape[1]; auto o = make_tensor(cos->shape, true);
  float cm = std::cos(m), sm = std::sin(m);
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < C; ++j) {
    float c = cos->data[i * C + j];
    if ((int)j == lab[i]) { float sn = std::sqrt(std::max(0.f, 1 - c * c)); o->data[i * C + j] = s * (c * cm - sn * sm); }
    else o->data[i * C + j] = s * c;
  }
  o->parents = {cos}; Node* op = o.get();
  o->backward_fn = [cos, op, B, C, lab, cm, sm, s] {
    for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < C; ++j) { float c = cos->data[i * C + j], go = op->grad[i * C + j];
      if ((int)j == lab[i]) { float sn = std::sqrt(std::max(1e-6f, 1 - c * c)); cos->grad[i * C + j] += s * (cm + c * sm / sn) * go; }
      else cos->grad[i * C + j] += s * go;
    }
  };
  return o;
}

// cross-entropy from logits[B,C] + int labels -> scalar (−mean log softmax[y]).
inline Tensor cross_entropy(const Tensor& logits, const std::vector<int>& lab) {
  std::vector<int64_t> idx(lab.begin(), lab.end());
  return mul_scalar(mean(gather_row(log_softmax_rows(logits), idx)), -1.f);
}

inline Tensor arcface_loss(const Tensor& emb, const Tensor& W, const std::vector<int>& lab, float m = 0.5f, float s = 64.f) {
  return cross_entropy(arcface_margin(cosine_logits(emb, W), lab, m, s), lab);
}
inline Tensor softmax_ce_loss(const Tensor& emb, const Tensor& W, const std::vector<int>& lab, float s = 64.f) {
  return cross_entropy(mul_scalar(cosine_logits(emb, W), s), lab);   // cosine softmax (ArcFace with m=0)
}
