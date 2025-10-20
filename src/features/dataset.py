import os
import glob
from typing import Optional, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES: List[str] = [
    "Left Lane Change",
    "Right Lane Change",
    "Left Turn",
    "Right Turn",
    "Slow-Stop",
    "Straight",
]

# preferred keys when features are inside dicts
_FEATURE_KEYS = ["features", "feats", "embedding", "embeddings", "feature", "data", "x", "X"]


def _extract_from_dict(d: dict) -> Any:
    """Pick a sensible value from a dict that likely holds features."""
    for k in _FEATURE_KEYS:
        if k in d:
            return d[k]
    # fallback to first value
    if len(d) > 0:
        return next(iter(d.values()))
    raise ValueError("Encountered empty dict while loading features.")


def _to_2d_float32(arr: np.ndarray) -> np.ndarray:
    """Normalize numeric ndarray to shape (T, D) with float32 dtype."""
    if arr.ndim == 1:
        arr = arr[None, :]  # (1, D)
    elif arr.ndim >= 3:
        T = arr.shape[0]
        arr = arr.reshape(T, -1)
    return arr.astype("float32", copy=False)


def _coerce_sequence(seq: List[Any]) -> np.ndarray:
    """
    Coerce a list of items (arrays or dicts) into a stacked (T, D) float32 array.
    Handles lists of ndarrays, lists of vectors, or lists of dicts with feature keys.
    """
    if len(seq) == 0:
        raise ValueError("Empty sequence encountered while stacking features.")

    # If list of dicts, extract a consistent key across all items
    if isinstance(seq[0], dict):
        # choose key using preference list, fallback to first key in first dict
        first = seq[0]
        key = None
        for k in _FEATURE_KEYS:
            if k in first:
                key = k
                break
        if key is None:
            if len(first) == 0:
                raise ValueError("Empty dict encountered inside sequence.")
            key = next(iter(first.keys()))
        stacked = [np.asarray(item[key]).squeeze() for item in seq]
        return _to_2d_float32(np.stack(stacked, axis=0))

    # Otherwise treat as list of arrays / numbers
    stacked = [np.asarray(item).squeeze() for item in seq]
    return _to_2d_float32(np.stack(stacked, axis=0))


def _load_npy_features(path: str) -> np.ndarray:
    """
    Robust loader for RIP feature .npy files.

    Handles:
      - 0-D object arrays (single pickled object) -> unwrap via .item()
      - dict -> extract feature value via preferred keys, or first value
      - list/tuple -> stack to (T, D), supports list of dicts/arrays
      - object ndarray -> convert to list and process as above
      - numeric ndarray -> normalize to (T, D) / flatten higher dims

    Always returns float32 array of shape (T, D).
    """
    obj = np.load(path, allow_pickle=True)

    # Unwrap 0-D object arrays
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        obj = obj.item()

    # Dict case
    if isinstance(obj, dict):
        val = _extract_from_dict(obj)
        val = np.asarray(val, dtype=object)  # may still be object/sequence
        if isinstance(val, np.ndarray) and val.dtype == object:
            seq = val.tolist()
            return _coerce_sequence(seq)
        val = np.asarray(val)
        return _to_2d_float32(val)

    # List/tuple case (of arrays or dicts)
    if isinstance(obj, (list, tuple)):
        return _coerce_sequence(list(obj))

    # Object ndarray case (list-like or dict-like elements)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        seq = obj.tolist()
        return _coerce_sequence(seq)

    # Numeric ndarray case
    arr = np.asarray(obj)
    return _to_2d_float32(arr)


class FeatureSequenceDataset(Dataset):
    """
    Loads pre-extracted feature .npy files laid out as:
      data/features/resnet50/<split>/<view>/<class_name>/*.npy

    For test split (unlabeled), it accepts flat .npy files under the view folder.

    Args:
        split_root: e.g., "data/features/resnet50/train"
        view: typically "frontal_view"
        max_files_per_class: if set, caps files per class (useful for quick runs)
    """

    def __init__(
        self,
        split_root: str,
        view: str = "frontal_view",
        max_files_per_class: Optional[int] = None,
    ):
        self.samples: List[Tuple[str, int]] = []
        view_dir = os.path.join(split_root, view)

        # Labeled (train/val): expect subfolders per class
        if any(os.path.isdir(os.path.join(view_dir, c)) for c in CLASS_NAMES):
            for ci, cname in enumerate(CLASS_NAMES):
                cdir = os.path.join(view_dir, cname)
                if not os.path.isdir(cdir):
                    continue
                files = sorted(glob.glob(os.path.join(cdir, "*.npy")))
                if max_files_per_class is not None:
                    files = files[: max_files_per_class]
                for f in files:
                    self.samples.append((f, ci))
            self.labeled = True
        else:
            # Unlabeled (test): collect all .npy in the view folder
            files = sorted(glob.glob(os.path.join(view_dir, "*.npy")))
            if max_files_per_class is not None:
                files = files[: max_files_per_class * len(CLASS_NAMES)]
            for f in files:
                self.samples.append((f, -1))
            self.labeled = False

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy feature files found under {view_dir}")

        # Infer (T, D) from a sample
        x0 = _load_npy_features(self.samples[0][0])
        self.seq_len = x0.shape[0]
        self.feat_dim = x0.shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = _load_npy_features(path)  # (T, D)
        x = torch.from_numpy(arr)       # float32 tensor (T, D)
        if self.labeled:
            y = torch.tensor(label, dtype=torch.long)
            return x, y
        else:
            return x, path