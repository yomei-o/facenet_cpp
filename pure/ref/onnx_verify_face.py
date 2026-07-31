# Verify a facenet.onnx (exported by onnx_export_face / `facenet export`) against the
# facenet-pytorch reference embedding under onnxruntime.
#   python onnx_verify_face.py [facenet.onnx] [imgsz]
import sys, os, numpy as np, onnxruntime as ort
onnx = sys.argv[1] if len(sys.argv) > 1 else "facenet.onnx"
S = int(sys.argv[2]) if len(sys.argv) > 2 else 160
ref_dir = os.path.dirname(__file__)
x = np.fromfile(os.path.join(ref_dir, "input.bin"), np.float32).reshape(1, 3, S, S)
ref = np.fromfile(os.path.join(ref_dir, "embed.bin"), np.float32)
out = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0].ravel()
worst = float(np.abs(out - ref).max())
print(f"onnxruntime vs facenet-pytorch: L2norm={np.linalg.norm(out):.6f} worst={worst:.3e} "
      f"{'MATCH' if worst < 1e-3 else 'MISMATCH'}")
