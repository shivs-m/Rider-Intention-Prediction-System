import os, yaml, numpy as np, torch
from typing import List
from features.lstm_classifier import LSTMFeatureClassifier
from features.dataset import CLASS_NAMES, FeatureSequenceDataset

def load_config(path="config.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

def _resample_to_len(x: torch.Tensor, T_out: int) -> torch.Tensor:
    T_in, D = x.shape
    if T_in == T_out: return x
    if T_in > T_out:
        idx = np.linspace(0, T_in - 1, num=T_out).round().astype(np.int64)
        return x[idx]
    out = torch.zeros(T_out, D, dtype=x.dtype)
    out[:T_in] = x
    return out

def predict_path(npy_path: str, clip_len: int = 64):
    # infer input dim from train set
    cfg = load_config()
    ds = FeatureSequenceDataset(cfg["data"]["root"] + "/train", cfg["data"]["view"])
    model = LSTMFeatureClassifier(input_dim=ds.feat_dim, num_classes=len(CLASS_NAMES))
    state = torch.load("dist/model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # use the same robust loader inside dataset
    from features.dataset import _load_npy_features
    arr = _load_npy_features(npy_path)          # (T, D)
    x = torch.from_numpy(arr)                   # (T, D)
    x = _resample_to_len(x, clip_len).unsqueeze(0)  # (1, T, D)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy().tolist()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(max(probs)), probs

if __name__ == "__main__":
    # pick any validation file path here:
    # e.g. data/features/resnet50/val/frontal_view/Left Turn/XXXXX.npy
    sample = None
    cfg = load_config()
    val_view = os.path.join(cfg["data"]["root"], "val", cfg["data"]["view"])
    # try first file from first class
    for cname in CLASS_NAMES:
        cdir = os.path.join(val_view, cname)
        if os.path.isdir(cdir):
            files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith(".npy")]
            if files:
                sample = files[0]
                break
    if sample is None:
        raise SystemExit("No validation files found.")
    label, conf, probs = predict_path(sample, clip_len=64)
    print("File:", sample)
    print("Pred:", label, "| conf:", round(conf, 4))