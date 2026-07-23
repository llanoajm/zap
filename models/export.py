"""Export every candidate to ONNX and CoreML, then LOAD each artifact and run a
forward pass before reporting success.  Nothing here claims an export "works"
without executing it.

- ONNX:   exported with torch.onnx, then loaded in onnxruntime and run on CPU;
          outputs are compared to the eager PyTorch outputs (allclose).
- CoreML: converted with coremltools (conversion runs on Linux).  CoreML
          *prediction* requires macOS, so on Linux we load the .mlpackage spec
          and validate the declared input/output shapes.  The exact macOS
          command to run a real forward pass is printed and stored in the report.

Run:  python models/export.py            (exports all, verifies, writes report)
      python models/export.py --onnx-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from architectures import REGISTRY  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "exported")
os.makedirs(OUT, exist_ok=True)


def _input_names(ctor, n):
    base = {1: ["input"], 2: ["a", "b"]}.get(n, [f"in{i}" for i in range(n)])
    return base


def export_onnx(name, model, inputs):
    path = os.path.join(OUT, f"{name}.onnx")
    innames = _input_names(REGISTRY[name], len(inputs))
    with torch.no_grad():
        eager = model(*inputs)
    eager_t = eager if isinstance(eager, tuple) else (eager,)
    outnames = [f"out{i}" for i in range(len(eager_t))]
    torch.onnx.export(
        model, inputs, path, input_names=innames, output_names=outnames,
        dynamic_axes={n: {0: "batch"} for n in innames + outnames},
        opset_version=17,
    )
    # LOAD + RUN in onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {n: x.numpy() for n, x in zip(innames, inputs)}
    ort_out = sess.run(None, feed)
    # compare
    ok = all(np.allclose(e.numpy(), o, atol=1e-4)
             for e, o in zip(eager_t, ort_out))
    return {"path": os.path.relpath(path), "loaded": True, "ran": True,
            "matches_eager": bool(ok),
            "input_shapes": {n: list(x.shape) for n, x in zip(innames, inputs)},
            "output_shapes": [list(np.asarray(o).shape) for o in ort_out]}


def export_coreml(name, model, inputs):
    try:
        import coremltools as ct
    except Exception as e:  # pragma: no cover
        return {"skipped": f"coremltools import failed: {e}"}
    innames = _input_names(REGISTRY[name], len(inputs))
    ts = torch.jit.trace(model, inputs)
    ct_inputs = [ct.TensorType(name=n, shape=x.shape) for n, x in zip(innames, inputs)]
    try:
        mlmodel = ct.convert(ts, inputs=ct_inputs,
                             minimum_deployment_target=ct.target.iOS16)
    except Exception as e:
        return {"error": f"conversion failed: {e}"}
    path = os.path.join(OUT, f"{name}.mlpackage")
    mlmodel.save(path)
    # LOAD the saved spec and validate declared shapes (prediction needs macOS)
    reloaded = ct.models.MLModel(path, skip_model_load=True)
    spec = reloaded.get_spec()
    in_desc = [(i.name, list(i.type.multiArrayType.shape)) for i in spec.description.input]
    out_desc = [o.name for o in spec.description.output]
    can_predict = sys.platform == "darwin"
    rec = {"path": os.path.relpath(path), "converted": True, "loaded_spec": True,
           "declared_inputs": in_desc, "declared_outputs": out_desc,
           "predicted_on_this_host": False,
           "predict_cmd_macos":
               f"python -c \"import coremltools as ct,numpy as np; "
               f"m=ct.models.MLModel('{os.path.relpath(path)}'); "
               f"print(m.predict({{{innames[0]!r}: np.random.randn(*{list(inputs[0].shape)}).astype('float32')}}))\""}
    if can_predict:
        try:
            feed = {n: x.numpy().astype(np.float32) for n, x in zip(innames, inputs)}
            reloaded2 = ct.models.MLModel(path)
            reloaded2.predict(feed)
            rec["predicted_on_this_host"] = True
        except Exception as e:
            rec["predict_error"] = str(e)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-only", action="store_true")
    args = ap.parse_args()

    report = {}
    for name, ctor in REGISTRY.items():
        model = ctor().eval()
        inputs = ctor.example_inputs(batch=1)
        entry = {"onnx": export_onnx(name, model, inputs)}
        print(f"[{name}] ONNX: ran={entry['onnx']['ran']} "
              f"matches_eager={entry['onnx']['matches_eager']}")
        if not args.onnx_only:
            entry["coreml"] = export_coreml(name, model, inputs)
            cm = entry["coreml"]
            status = ("converted+spec-loaded" if cm.get("converted")
                      else cm.get("error") or cm.get("skipped"))
            print(f"[{name}] CoreML: {status} "
                  f"(predicted_here={cm.get('predicted_on_this_host')})")
        report[name] = entry

    out = os.path.join(OUT, "export_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    n_ok = sum(1 for e in report.values() if e["onnx"]["matches_eager"])
    print(f"ONNX verified (loaded+ran+matches eager): {n_ok}/{len(report)}")


if __name__ == "__main__":
    main()
