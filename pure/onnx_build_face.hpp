// InceptionResnetV1 (facenet) -> ONNX (opset 13) graph builder. Takes the FUSED FaceProv (BN folded)
// and emits Conv(with per-dim pads)/Relu/MaxPool/Concat/Mul/Add/GlobalAveragePool/Flatten/Gemm/
// LpNormalization — a faithful 1:1 of net_facenet.hpp. Output = 512-D L2-normalized embedding.
#pragma once
#include "net_facenet.hpp"
#include "onnx.hpp"
#include <string>
using namespace onx;

inline Graph build_facenet_onnx(FaceProv& prov, int64_t IMG = 160) {
  Graph g; g.opset = 13;
  g.inputs.push_back({"input", {1, 3, IMG, IMG}});
  g.outputs.push_back({"embedding", {1, 512}});
  int uid = 0; auto U = [&](const char* p) { return std::string(p) + std::to_string(uid++); };

  auto conv = [&](const std::string& in, bool do_relu) -> std::string {
    FLayer& L = prov.next();
    std::string wn = U("w"), bn = U("b"), yn = U("c");
    g.init_f.push_back({wn, {L.Co, L.Ci, L.kh, L.kw}, L.w->data});
    g.init_f.push_back({bn, {L.Co}, L.b->data});
    onx::Node n; n.op_type = "Conv"; n.name = yn; n.input = {in, wn, bn}; n.output = {yn};
    n.attr.push_back({"kernel_shape", A_INTS, 0, 0, "", {L.kh, L.kw}, {}});
    n.attr.push_back({"strides", A_INTS, 0, 0, "", {L.sh, L.sw}, {}});
    n.attr.push_back({"pads", A_INTS, 0, 0, "", {L.ph, L.pw, L.ph, L.pw}, {}});
    n.attr.push_back({"group", A_INT, 1, 0, "", {}, {}});
    g.nodes.push_back(n);
    if (do_relu) { std::string r = U("r"); g.nodes.push_back({"Relu", r, {yn}, {r}, {}}); return r; }
    return yn;
  };
  auto bc = [&](const std::string& x) { return conv(x, true); };   // BasicConv2d (conv+relu)
  auto cv = [&](const std::string& x) { return conv(x, false); };  // residual conv2d (no relu)
  auto rl = [&](const std::string& in) { std::string y = U("r"); g.nodes.push_back({"Relu", y, {in}, {y}, {}}); return y; };
  auto add = [&](const std::string& a, const std::string& b) { std::string y = U("ad"); g.nodes.push_back({"Add", y, {a, b}, {y}, {}}); return y; };
  auto mulc = [&](const std::string& in, float v) { std::string s = U("k"), y = U("ms"); g.init_f.push_back({s, {1}, {v}}); g.nodes.push_back({"Mul", y, {in, s}, {y}, {}}); return y; };
  auto concat = [&](const std::vector<std::string>& xs) { std::string y = U("ct"); onx::Node n{"Concat", y, xs, {y}, {}}; n.attr.push_back({"axis", A_INT, 1, 0, "", {}, {}}); g.nodes.push_back(n); return y; };
  auto maxpool = [&](const std::string& in) { std::string y = U("mp"); onx::Node n{"MaxPool", y, {in}, {y}, {}};
    n.attr.push_back({"kernel_shape", A_INTS, 0, 0, "", {3, 3}, {}}); n.attr.push_back({"strides", A_INTS, 0, 0, "", {2, 2}, {}});
    n.attr.push_back({"pads", A_INTS, 0, 0, "", {0, 0, 0, 0}, {}}); g.nodes.push_back(n); return y; };

  auto block35 = [&](const std::string& x) { std::string b0 = bc(x), h1 = bc(x), c1 = bc(h1), h2 = bc(x); h2 = bc(h2); std::string c2 = bc(h2);
    return rl(add(x, mulc(cv(concat({b0, c1, c2})), 0.17f))); };
  auto block17 = [&](const std::string& x) { std::string b0 = bc(x), h = bc(x); h = bc(h); std::string b1 = bc(h);
    return rl(add(x, mulc(cv(concat({b0, b1})), 0.10f))); };
  auto block8 = [&](const std::string& x, float sc, bool relu) { std::string b0 = bc(x), h = bc(x); h = bc(h); std::string b1 = bc(h);
    std::string o = add(x, mulc(cv(concat({b0, b1})), sc)); return relu ? rl(o) : o; };
  auto mixed6a = [&](const std::string& x) { std::string b0 = bc(x), h = bc(x); h = bc(h); std::string b1 = bc(h), b2 = maxpool(x); return concat({b0, b1, b2}); };
  auto mixed7a = [&](const std::string& x) { std::string h = bc(x), b0 = bc(h); h = bc(x); std::string b1 = bc(h); h = bc(x); h = bc(h); std::string b2 = bc(h), b3 = maxpool(x); return concat({b0, b1, b2, b3}); };

  std::string x = bc("input"); x = bc(x); x = bc(x); x = maxpool(x); x = bc(x); x = bc(x); x = bc(x);
  for (int i = 0; i < 5; ++i)  x = block35(x);
  x = mixed6a(x);
  for (int i = 0; i < 10; ++i) x = block17(x);
  x = mixed7a(x);
  for (int i = 0; i < 5; ++i)  x = block8(x, 0.20f, true);
  x = block8(x, 1.0f, false);
  // head: GlobalAveragePool -> Flatten -> Gemm(last_linear folded) -> LpNormalization
  std::string gp = U("gp"); g.nodes.push_back({"GlobalAveragePool", gp, {x}, {gp}, {}});
  std::string fl = U("fl"); { onx::Node n{"Flatten", fl, {gp}, {fl}, {}}; n.attr.push_back({"axis", A_INT, 1, 0, "", {}, {}}); g.nodes.push_back(n); }
  FLayer& Ll = prov.next(); std::string wn = U("w"), bn = U("b"), gm = U("gm");
  g.init_f.push_back({wn, {Ll.Co, Ll.Ci}, Ll.w->data}); g.init_f.push_back({bn, {Ll.Co}, Ll.b->data});
  { onx::Node n{"Gemm", gm, {fl, wn, bn}, {gm}, {}}; n.attr.push_back({"transB", A_INT, 1, 0, "", {}, {}}); g.nodes.push_back(n); }
  { onx::Node n{"LpNormalization", "lp", {gm}, {"embedding"}, {}}; n.attr.push_back({"axis", A_INT, 1, 0, "", {}, {}}); n.attr.push_back({"p", A_INT, 2, 0, "", {}, {}}); g.nodes.push_back(n); }
  return g;
}
