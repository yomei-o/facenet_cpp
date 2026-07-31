// Device (Thrust) versions of the facenet-specific ops, mirroring face_ops.hpp on DT tensors:
// relu, asymmetric spatial pad (1x7/7x1 convs), conv-with-per-dim-pad, global avg pool, dense
// linear bias-add, row L2-normalize, and an eval-mode BatchNorm (running stats). One source:
// CPU under THRUST_DEVICE_SYSTEM_CPP, GPU under nvcc -DUSE_CUDA. Same backward-scatter idiom as
// dtensor.hpp (per-input-element accumulation = race-free).
#pragma once
#include "dtensor.hpp"

inline DT drelu(DT x) {
  DT y = dmake(x->shape); int64_t n = x->numel(); const float* X = x->dp(); float* Y = y->dp();
  bk::parallel_for(n, [=] BK_HD (int64_t i) { Y[i] = X[i] > 0 ? X[i] : 0.f; });
  y->parents = {x};
  y->backward_fn = [x, n, yo = y.get()]() { const float* X = x->dp(); const float* GY = yo->gp(); float* GX = x->gp();
    bk::parallel_for(n, [=] BK_HD (int64_t i) { if (X[i] > 0) GX[i] += GY[i]; }); };
  return y;
}

// zero-pad H by ph each side, W by pw each side (symmetric per dim).
inline DT dpad_hw(DT x, int64_t ph, int64_t pw) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3], OH = H + 2 * ph, OW = W + 2 * pw;
  DT y = dmake({N, C, OH, OW}); const float* X = x->dp(); float* Y = y->dp();
  bk::parallel_for(N * C * OH * OW, [=] BK_HD (int64_t idx) {
    int64_t ow = idx % OW, t = idx / OW, oh = t % OH, t2 = t / OH, c = t2 % C, n = t2 / C, ih = oh - ph, iw = ow - pw;
    Y[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? X[((n * C + c) * H + ih) * W + iw] : 0.f;
  });
  y->parents = {x};
  y->backward_fn = [x, N, C, H, W, OH, OW, ph, pw, yo = y.get()]() { const float* GY = yo->gp(); float* GX = x->gp();
    bk::parallel_for(N * C * H * W, [=] BK_HD (int64_t idx) {
      int64_t iw = idx % W, t = idx / W, ih = t % H, t2 = t / H, c = t2 % C, n = t2 / C;
      GX[idx] += GY[((n * C + c) * OH + (ih + ph)) * OW + (iw + pw)]; }); };
  return y;
}
inline DT dconv2d_hw(DT x, DT w, DT b, int64_t stride, int64_t ph, int64_t pw) {
  DT px = (ph || pw) ? dpad_hw(x, ph, pw) : x; return dconv2d(px, w, b, stride, 0, 1);
}

// global average pool: (N,C,H,W) -> (N,C).
inline DT dgap(DT x) {
  int64_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
  DT y = dmake({N, C}); const float* X = x->dp(); float* Y = y->dp();
  bk::parallel_for(N * C, [=] BK_HD (int64_t nc) { float s = 0; const float* p = X + nc * HW; for (int64_t i = 0; i < HW; ++i) s += p[i]; Y[nc] = s / (float)HW; });
  y->parents = {x};
  y->backward_fn = [x, N, C, HW, yo = y.get()]() { const float* GY = yo->gp(); float* GX = x->gp();
    bk::parallel_for(N * C * HW, [=] BK_HD (int64_t idx) { GX[idx] += GY[idx / HW] / (float)HW; }); };
  return y;
}

// y[n,d] = x[n,d] + b[d]  (x:[N,D], b:[D])
inline DT dadd_rowvec(DT x, DT b) {
  int64_t N = x->shape[0], D = x->shape[1];
  DT y = dmake(x->shape); const float* X = x->dp(); const float* B = b->dp(); float* Y = y->dp();
  bk::parallel_for(N * D, [=] BK_HD (int64_t i) { Y[i] = X[i] + B[i % D]; });
  y->parents = {x, b};
  y->backward_fn = [x, b, N, D, yo = y.get()]() { const float* GY = yo->gp(); float* GX = x->gp(); float* GB = b->gp();
    bk::parallel_for(N * D, [=] BK_HD (int64_t i) { GX[i] += GY[i]; });
    bk::parallel_for(D, [=] BK_HD (int64_t d) { float s = 0; for (int64_t n = 0; n < N; ++n) s += GY[n * D + d]; GB[d] += s; }); };
  return y;
}
// dense layer via matmul(x, Wᵀ) + bias  (W:[out,in])
inline DT dlinear(DT x, DT W, DT b) { return dadd_rowvec(dmatmul(x, dtranspose2d(W)), b); }

// row L2 normalize: y = x / max(‖x‖, eps)
inline DT dl2norm_rows(DT x, float eps = 1e-12f) {
  int64_t N = x->shape[0], D = x->shape[1];
  DT y = dmake(x->shape); auto inv = std::make_shared<thrust::device_vector<float>>(N, 0.f);
  const float* X = x->dp(); float* Y = y->dp(); float* INV = thrust::raw_pointer_cast(inv->data());
  bk::parallel_for(N, [=] BK_HD (int64_t n) { const float* p = X + n * D; float ss = 0; for (int64_t d = 0; d < D; ++d) ss += p[d] * p[d];
    float iv = 1.f / fmaxf(sqrtf(ss), eps); INV[n] = iv; float* q = Y + n * D; for (int64_t d = 0; d < D; ++d) q[d] = p[d] * iv; });
  y->parents = {x};
  y->backward_fn = [x, inv, N, D, yo = y.get()]() { const float* Yv = yo->dp(); const float* GY = yo->gp(); float* GX = x->gp();
    const float* INV = thrust::raw_pointer_cast(inv->data());
    bk::parallel_for(N, [=] BK_HD (int64_t n) { const float* yy = Yv + n * D; const float* gy = GY + n * D; float* gx = GX + n * D;
      float dot = 0; for (int64_t d = 0; d < D; ++d) dot += yy[d] * gy[d];
      for (int64_t d = 0; d < D; ++d) gx[d] += INV[n] * (gy[d] - yy[d] * dot); }); };
  return y;
}

// eval-mode BatchNorm: y = gamma*(x-rm)/sqrt(rv+eps) + beta  (per channel, running stats — matches
// net_facenet_unfused eval so the device forward reproduces the fp32 reference embedding).
inline DT dbn_eval(DT x, DT gamma, DT beta, DT rm, DT rv, float eps) {
  int64_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
  DT y = dmake(x->shape); const float* X = x->dp(); const float* G = gamma->dp(); const float* B = beta->dp();
  const float* RM = rm->dp(); const float* RV = rv->dp(); float* Y = y->dp();
  bk::parallel_for(N * C * HW, [=] BK_HD (int64_t idx) { int64_t c = (idx / HW) % C;
    Y[idx] = (X[idx] - RM[c]) * (G[c] / sqrtf(RV[c] + eps)) + B[c]; });
  y->parents = {x};
  y->backward_fn = [x, gamma, rv, N, C, HW, eps, yo = y.get()]() { const float* G = gamma->dp(); const float* RV = rv->dp();
    const float* GY = yo->gp(); float* GX = x->gp();
    bk::parallel_for(N * C * HW, [=] BK_HD (int64_t idx) { int64_t c = (idx / HW) % C; GX[idx] += GY[idx] * G[c] / sqrtf(RV[c] + eps); }); };
  return y;
}
