# server/main.py
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from features.lstm_classifier import LSTMFeatureClassifier
from features.dataset import FeatureSequenceDataset, CLASS_NAMES, _load_npy_features

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Rider Intention Prediction (Task 1)", version="1.0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        # Provide a sensible fallback so the app can still boot
        return {
            "data": {
                "root": "data",   # expected tree: data/train, data/val, ...
                "view": "front"   # or whatever your default view is
            }
        }
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}

# -----------------------------------------------------------------------------
# Model load with auto-shape inference from checkpoint
# -----------------------------------------------------------------------------
def infer_lstm_shape_from_state(state_dict: dict):
    """
    Infer (hidden_size, num_layers, bidirectional) from LSTM parameter names/shapes.
    Expects keys like:
      - lstm.weight_hh_l0
      - lstm.weight_ih_l0_reverse  (if bidirectional)
    """
    w_hh_l0 = state_dict.get("lstm.weight_hh_l0")
    if w_hh_l0 is None:
        raise RuntimeError("Checkpoint missing 'lstm.weight_hh_l0' – cannot infer hidden size.")
    hidden = int(w_hh_l0.shape[1])  # second dim is hidden

    # Bidirectional: presence of any *_reverse params
    bidirectional = any(k.startswith("lstm.weight_ih_l0_reverse") for k in state_dict.keys())

    # Layers: scan keys like lstm.weight_hh_l{n} and take max+1
    layer_idxs = set()
    pat = re.compile(r"^lstm\.weight_hh_l(\d+)(?:_reverse)?$")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            layer_idxs.add(int(m.group(1)))
    num_layers = int(max(layer_idxs) + 1) if layer_idxs else 1

    return hidden, num_layers, bidirectional

# -----------------------------------------------------------------------------
# Initialize config, dataset, and model
# -----------------------------------------------------------------------------
_cfg = load_config()
_data_root = Path(_cfg["data"]["root"])
_view = _cfg["data"]["view"]

# We only instantiate the dataset to learn the input feature dimension
_train_dir = _data_root / "train"
if not _train_dir.exists():
    # Don’t crash the app if data isn’t present in the container; set a default feat dim
    # You can still /healthz and serve UI. Predict will error if no .npy uploaded.
    _input_dim = 2048
else:
    _train_ds = FeatureSequenceDataset(str(_train_dir), _view)
    _input_dim = int(_train_ds.feat_dim)

_ckpt_path = Path("dist/model.pt")
if not _ckpt_path.exists():
    raise RuntimeError("dist/model.pt not found. Train first, then start the server.")

_state = torch.load(str(_ckpt_path), map_location="cpu")
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
_model.load_state_dict(_state, strict=True)
_model.eval()

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def _resample_to_len(x: torch.Tensor, T_out: int) -> torch.Tensor:
    """Pad or truncate features to a fixed temporal length T_out."""
    if x.ndim != 2:
        raise ValueError(f"Expected features of shape [T, D], got {tuple(x.shape)}")
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
_here = Path(__file__).resolve().parent
static_dir = _here / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return HTMLResponse("<h2>Frontend not found. Please add static/index.html</h2>")

# Render often sends HEAD; don’t 405 it
@app.head("/", response_class=PlainTextResponse)
async def root_head():
    return ""

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "classes": CLASS_NAMES,
        "model_hidden": int(_hidden),
        "model_layers": int(_layers),
        "model_bidirectional": bool(_bi),
        "feat_dim": int(_input_dim),
        "checkpoint": str(_ckpt_path),
    }

@app.head("/healthz", response_class=PlainTextResponse)
async def healthz_head():
    return ""

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.post("/predict_feature")
async def predict_feature(
    file: UploadFile = File(None),
    npy_path: Optional[str] = Query(default=None, description="Path to .npy on server"),
    clip_len: int = Query(default=64, ge=1, le=512)
):
    """
    Predict from a sequence of pre-extracted frame features (shape [T, D]) in a .npy file.
    - Upload the .npy via form-data (key: 'file'), OR
    - Provide ?npy_path=/absolute/or/relative/path.npy (file must exist in container FS)
    """
    try:
        # -------------------------
        # Load features
        # -------------------------
        if file is not None:
            tmp = Path("temp_feat.npy")
            with tmp.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            arr = _load_npy_features(str(tmp))
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        elif npy_path is not None:
            arr = _load_npy_features(npy_path)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload a .npy file or provide ?npy_path=..."}
            )

        if arr.ndim != 2:
            return JSONResponse(
                status_code=400,
                content={"error": f"Expected 2D features [T, D], got shape {list(arr.shape)}"}
            )

        # -------------------------
        # Preprocess & run
        # -------------------------
        x = torch.from_numpy(arr)                # [T, D]
        x = _resample_to_len(x, clip_len)        # [clip_len, D]
        x = x.unsqueeze(0)                       # [1, T, D]

        with torch.no_grad():
            logits = _model(x)                   # [1, C]
            probs_t = torch.softmax(logits, dim=1)[0]
            probs = probs_t.numpy().tolist()

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