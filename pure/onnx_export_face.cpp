// Export the fused InceptionResnetV1 to facenet.onnx (opset 13).
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\onnx_export_face.cpp
//   run:   onnx_export_face [data_net_dir] [out.onnx] [imgsz]
#include "onnx_build_face.hpp"
#include <cstdio>

int main(int argc, char** argv) {
  std::string DN = argc > 1 ? argv[1] : "pure/ref/data_net/";
  std::string out = argc > 2 ? argv[2] : "facenet.onnx";
  int64_t S = argc > 3 ? atoll(argv[3]) : 160;
  if (DN.back() != '/') DN += '/';
  FaceProv prov = load_facenet(DN);
  Graph g = build_facenet_onnx(prov, S);
  save_onnx(g, out);
  printf("wrote %s  (%zu nodes, %zu float initializers, imgsz=%lld)\n", out.c_str(), g.nodes.size(), g.init_f.size(), (long long)S);
  return 0;
}
