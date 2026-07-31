// Minimal pure-C++ ONNX interpreter for the facenet graph: loads an .onnx (via onnx.hpp) and runs
// it on the shared autograd engine. Handles per-dim Conv pads (1x7/7x1 Inception convs) plus Relu,
// MaxPool, Concat, Mul, Add, GlobalAveragePool, Flatten, Gemm, LpNormalization.
#pragma once
#include "onnx.hpp"
#include "autograd.hpp"
#include "face_ops.hpp"      // relu, conv2d_hw, gap, add_rowvec, l2norm_rows
#include "ops2d.hpp"         // reshape, mul_scalar
#include "linalg.hpp"        // matmul, transpose2d
#include <map>
#include <string>
#include <vector>

namespace onx {
inline const Attr* find_attr_(const Node& n, const std::string& k) { for (auto& a : n.attr) if (a.name == k) return &a; return nullptr; }
inline int64_t attr_i(const Node& n, const std::string& k, int64_t d) { auto* a = find_attr_(n, k); if (!a) return d; return a->ints.empty() ? a->i : a->ints[0]; }
inline std::vector<int64_t> attr_ints(const Node& n, const std::string& k) { auto* a = find_attr_(n, k); return a ? a->ints : std::vector<int64_t>{}; }

inline std::map<std::string, Tensor> run_onnx_face(const Graph& g, const Tensor& x) {
  std::map<std::string, Tensor> vals;
  for (auto& t : g.init_f) vals[t.name] = from_data(t.dims, t.data);
  std::string in_name; for (auto& vi : g.inputs) if (!vals.count(vi.name)) { in_name = vi.name; break; }
  if (in_name.empty() && !g.inputs.empty()) in_name = g.inputs[0].name;
  vals[in_name] = x;
  auto get = [&](const std::string& n) -> Tensor { return vals.at(n); };

  for (const auto& nd : g.nodes) {
    const std::string& op = nd.op_type; Tensor y;
    if (op == "Conv") {
      Tensor w = get(nd.input[1]); Tensor b = (nd.input.size() >= 3 && !nd.input[2].empty()) ? get(nd.input[2]) : Tensor();
      auto st = attr_ints(nd, "strides"); auto pd = attr_ints(nd, "pads");
      int64_t s = st.empty() ? 1 : st[0], ph = pd.size() > 0 ? pd[0] : 0, pw = pd.size() > 1 ? pd[1] : 0;
      y = conv2d_hw(get(nd.input[0]), w, b, s, ph, pw);
    } else if (op == "Relu") y = relu(get(nd.input[0]));
    else if (op == "Mul") { Tensor a = get(nd.input[0]), b = get(nd.input[1]);
      y = b->numel() == 1 ? mul_scalar(a, b->data[0]) : (a->numel() == 1 ? mul_scalar(b, a->data[0]) : mul(a, b)); }
    else if (op == "Add") y = add(get(nd.input[0]), get(nd.input[1]));
    else if (op == "Concat") { std::vector<Tensor> xs; for (auto& s : nd.input) xs.push_back(get(s)); y = concat_ch(xs); }
    else if (op == "MaxPool") { auto k = attr_ints(nd, "kernel_shape"), st = attr_ints(nd, "strides"), pd = attr_ints(nd, "pads");
      y = maxpool2d(get(nd.input[0]), k.empty() ? 1 : k[0], st.empty() ? 1 : st[0], pd.empty() ? 0 : pd[0]); }
    else if (op == "GlobalAveragePool") { Tensor t = gap(get(nd.input[0])); y = reshape(t, {t->shape[0], t->shape[1], 1, 1}); }
    else if (op == "Flatten") { Tensor t = get(nd.input[0]); y = reshape(t, {t->shape[0], t->numel() / t->shape[0]}); }
    else if (op == "Gemm") { Tensor a = get(nd.input[0]), W = get(nd.input[1]), b = get(nd.input[2]);
      Tensor prod = attr_i(nd, "transB", 0) ? matmul(a, transpose2d(W)) : matmul(a, W); y = add_rowvec(prod, b); }
    else if (op == "LpNormalization") y = l2norm_rows(get(nd.input[0]));
    else if (op == "Identity") y = get(nd.input[0]);
    else { printf("unsupported ONNX op: %s\n", op.c_str()); std::exit(1); }
    vals[nd.output[0]] = y;
  }
  return vals;
}
}  // namespace onx
