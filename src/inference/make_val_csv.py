import os, csv, yaml, numpy as np, torch
from features.dataset import FeatureSequenceDataset, CLASS_NAMES, _load_npy_features
from features.lstm_classifier import LSTMFeatureClassifier


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resample_to_len(x: torch.Tensor, T_out: int) -> torch.Tensor:
    """Pad or truncate feature sequence to T_out frames."""
    import numpy as np
    T_in, D = x.shape
    if T_in == T_out:
        return x
    if T_in > T_out:
        idx = np.linspace(0, T_in - 1, num=T_out).round().astype(np.int64)
        return x[idx]
    out = torch.zeros(T_out, D, dtype=x.dtype)
    out[:T_in] = x
    return out


if __name__ == "__main__":
    cfg = load_config()
    clip_len = 64
    val_view = os.path.join(cfg["data"]["root"], "val", cfg["data"]["view"])

    # Load model and dataset info
    ds = FeatureSequenceDataset(cfg["data"]["root"] + "/train", cfg["data"]["view"])
    model = LSTMFeatureClassifier(input_dim=ds.feat_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("dist/model.pt", map_location="cpu"))
    model.eval()

    out_csv = "task1_val_result_format.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["video_name"] + CLASS_NAMES
        writer.writerow(header)

        for cname in CLASS_NAMES:
            cdir = os.path.join(val_view, cname)
            if not os.path.isdir(cdir):
                continue

            for fname in sorted(os.listdir(cdir)):
                if not fname.endswith(".npy"):
                    continue

                fpath = os.path.join(cdir, fname)
                arr = _load_npy_features(fpath)
                x = torch.from_numpy(arr)
                x = _resample_to_len(x, clip_len).unsqueeze(0)

                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0].numpy().tolist()

                idx = int(np.argmax(probs))
                one_hot = [1 if i == idx else 0 for i in range(len(CLASS_NAMES))]
                writer.writerow([fname] + one_hot)

    print("✅ Wrote:", out_csv)