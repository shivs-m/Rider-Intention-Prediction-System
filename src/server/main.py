# src/server/main.py

import os
import re
import yaml
import numpy as np
import torch
import shutil
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from features.lstm_classifier import LSTMFeatureClassifier
from features.dataset import FeatureSequenceDataset, CLASS_NAMES, _load_npy_features


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Rider Intention Prediction (Task 1)", version="1.0.1")

# Allow local CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Model load with auto-shape inference from checkpoint
# -----------------------------------------------------------------------------
def infer_lstm_shape_from_state(state_dict: dict):
    """
    Infer (hidden_size, num_layers, bidirectional) from LSTM parameter names/shapes.
    """
    # Hidden size: use weight_hh_l0: shape = [4*hidden, hidden]
    w_hh_l0 = state_dict.get("lstm.weight_hh_l0", None)
    if w_hh_l0 is None:
        raise RuntimeError("Checkpoint missing 'lstm.weight_hh_l0' – cannot infer hidden size.")
    hidden = w_hh_l0.shape[1]  # second dim is hidden

    # Bidirectional: presence of reverse weights
    bidirectional = any(k.startswith("lstm.weight_ih_l0_reverse") for k in state_dict.keys())

    # Layers: count layers by scanning keys like lstm.weight_hh_l{n}
    layer_idxs = set()
    pat = re.compile(r"^lstm\.weight_hh_l(\d+)(?:_reverse)?$")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            layer_idxs.add(int(m.group(1)))
    num_layers = (max(layer_idxs) + 1) if layer_idxs else 1

    return hidden, num_layers, bidirectional


# -----------------------------------------------------------------------------
# Initialize config, dataset, and model
# -----------------------------------------------------------------------------
_cfg = load_config()
_train_ds = FeatureSequenceDataset(os.path.join(_cfg["data"]["root"], "train"),
                                   _cfg["data"]["view"])
_input_dim = _train_ds.feat_dim

_ckpt_path = "dist/model.pt"
if not os.path.exists(_ckpt_path):
    raise RuntimeError("dist/model.pt not found. Train first, then start the server.")

_state = torch.load(_ckpt_path, map_location="cpu")
if not isinstance(_state, dict):
    raise RuntimeError("Unexpected checkpoint format; expected a PyTorch state_dict.")

_hidden, _layers, _bi = infer_lstm_shape_from_state(_state)

_model = LSTMFeatureClassifier(
    input_dim=_input_dim,
    num_classes=len(CLASS_NAMES),
    hidden=_hidden,
    layers=_layers,
    bi=_bi,
)
# strict=True is fine now because we match shapes
_model.load_state_dict(_state, strict=True)
_model.eval()


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def _resample_to_len(x: torch.Tensor, T_out: int) -> torch.Tensor:
    """Pad or truncate features to a fixed temporal length."""
    T_in, D = x.shape
    if T_in == T_out:
        return x
    if T_in > T_out:
        idx = np.linspace(0, T_in - 1, num=T_out).round().astype(np.int64)
        return x[idx]
    out = torch.zeros(T_out, D, dtype=x.dtype)
    out[:T_in] = x
    return out


# -----------------------------------------------------------------------------
# Static UI (front-end)
# -----------------------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return f.read()
    return HTMLResponse("<h2>Frontend not found. Please add static/index.html</h2>")


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "classes": CLASS_NAMES,
        "model_hidden": _hidden,
        "model_layers": _layers,
        "model_bidirectional": _bi,
        "feat_dim": _input_dim,
    }


@app.post("/predict_feature")
async def predict_feature(
    file: UploadFile = File(None),
    npy_path: str = Query(default=None),
    clip_len: int = Query(default=64)
):
    try:
        # Load features
        if file is not None:
            tmp = "temp_feat.npy"
            with open(tmp, "wb") as f:
                shutil.copyfileobj(file.file, f)
            arr = _load_npy_features(tmp)
            os.remove(tmp)
        elif npy_path is not None:
            arr = _load_npy_features(npy_path)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload a .npy file or provide npy_path."}
            )

        # Preprocess & run
        x = torch.from_numpy(arr)                   # (T, D)
        x = _resample_to_len(x, clip_len).unsqueeze(0)  # (1, T, D)
        with torch.no_grad():
            probs = torch.softmax(_model(x), dim=1)[0].numpy().tolist()

        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        one_hot = [1 if i == idx else 0 for i in range(len(CLASS_NAMES))]

        return {
            "label": label,
            "confidence": float(max(probs)),
            "probs": probs,
            "one_hot": one_hot,
            "classes": CLASS_NAMES,
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})