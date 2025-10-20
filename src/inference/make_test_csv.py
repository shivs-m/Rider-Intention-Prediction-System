import os, csv, yaml, numpy as np, torch
from typing import List
from features.dataset import FeatureSequenceDataset, CLASS_NAMES, _load_npy_features
from features.lstm_classifier import LSTMFeatureClassifier

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _resample_to_len(x: torch.Tensor, T_out: int) -> torch.Tensor:
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
    test_view = os.path.join(cfg["data"]["root"], "test", cfg["data"]["view"])

    # model + feature dim (from train set)
    ds = FeatureSequenceDataset(cfg["data"]["root"] + "/train", cfg["data"]["view"])
    feat_dim = ds.feat_dim
    model = LSTMFeatureClassifier(input_dim=feat_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("dist/model.pt", map_location="cpu"))
    model.eval()

    out_csv = "task1_test_result_format.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["video_name"] + CLASS_NAMES
        writer.writerow(header)

        # Walk test directory (flat or nested)
        test_files: List[str] = []
        for root, _, files in os.walk(test_view):
            for fname in files:
                if fname.endswith(".npy"):
                    test_files.append(os.path.join(root, fname))

        test_files.sort()

        skipped = 0
        for fpath in test_files:
            fname = os.path.basename(fpath)
            try:
                arr = _load_npy_features(fpath)        # (T, D) normally
                # sanity: wrong feat_dim -> coerce
                if arr.shape[1] != feat_dim:
                    arr = arr.reshape(arr.shape[0], -1)
                    if arr.shape[1] != feat_dim:
                        # fallback if still wrong
                        arr = np.zeros((1, feat_dim), dtype="float32")
            except Exception as e:
                # empty/bad file -> fallback zeros
                skipped += 1
                arr = np.zeros((1, feat_dim), dtype="float32")

            x = torch.from_numpy(arr)
            x = _resample_to_len(x, clip_len).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].numpy().tolist()
            idx = int(np.argmax(probs))
            one_hot = [1 if i == idx else 0 for i in range(len(CLASS_NAMES))]

            writer.writerow([fname] + one_hot)

    msg = f"✅ Wrote: {out_csv}"
    if skipped:
        msg += f" (used zero-fallback for {skipped} file(s))"
    print(msg)