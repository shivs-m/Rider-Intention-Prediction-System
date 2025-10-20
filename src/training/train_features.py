# src/training/train_features.py
import os, yaml, argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from features.dataset import FeatureSequenceDataset, CLASS_NAMES
from features.lstm_classifier import LSTMFeatureClassifier


# ----------------------------- Utility Functions -----------------------------
def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resample_to_len(x: torch.Tensor, T_out: int) -> torch.Tensor:
    """Pad or truncate a sequence to T_out frames."""
    T_in, D = x.shape
    if T_in == T_out:
        return x
    if T_in > T_out:
        idx = np.linspace(0, T_in - 1, num=T_out).round().astype(np.int64)
        return x[idx]
    out = torch.zeros(T_out, D, dtype=x.dtype)
    out[:T_in] = x
    return out


def collate_fixed_len(batch, T_out: int):
    """Ensures all sequences in a batch are of equal temporal length."""
    xs, ys = zip(*batch)
    xs = [_resample_to_len(x, T_out) for x in xs]
    x = torch.stack(xs, dim=0)             # (B, T_out, D)
    y = torch.tensor(ys, dtype=torch.long) # (B,)
    return x, y


def compute_class_weights(train_root: str, view: str) -> torch.Tensor:
    """Computes inverse-frequency class weights for CrossEntropyLoss."""
    counts = Counter()
    for i, cname in enumerate(CLASS_NAMES):
        cdir = os.path.join(train_root, view, cname)
        if os.path.isdir(cdir):
            n = sum(1 for f in os.listdir(cdir) if f.endswith(".npy"))
            counts[i] = n
    total = sum(counts.values()) if counts else 1
    weights = []
    for i in range(len(CLASS_NAMES)):
        ci = counts.get(i, 1)
        weights.append(total / (ci * len(CLASS_NAMES)))
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------- Training Script -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    ap.add_argument("--max_per_class", type=int, default=10, help="Cap files per class; -1 = use all")
    ap.add_argument("--epochs", type=int, default=5, help="Training epochs")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--lr", type=float, default=None, help="Learning rate override")
    ap.add_argument("--clip_len", type=int, default=64, help="Fixed sequence length (frames)")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay (L2)")
    ap.add_argument("--plateau_patience", type=int, default=3, help="Scheduler patience (epochs)")
    ap.add_argument("--plateau_factor", type=float, default=0.5, help="LR reduce factor on plateau")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for scheduler")
    ap.add_argument("--no_class_weights", action="store_true", help="Disable class-weighted loss")
    ap.add_argument("--confmat", action="store_true", help="Show confusion matrix each epoch")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg["train"]["seed"]))

    # ----------------------------- Dataset -----------------------------
    cap = None if args.max_per_class == -1 else args.max_per_class
    train_ds = FeatureSequenceDataset(
        split_root=os.path.join(cfg["data"]["root"], "train"),
        view=cfg["data"]["view"],
        max_files_per_class=cap,
    )
    val_ds = FeatureSequenceDataset(
        split_root=os.path.join(cfg["data"]["root"], "val"),
        view=cfg["data"]["view"],
        max_files_per_class=cap,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fixed_len(b, args.clip_len),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fixed_len(b, args.clip_len),
    )

    # ----------------------------- Model -----------------------------
    model = LSTMFeatureClassifier(
        input_dim=train_ds.feat_dim,
        num_classes=len(CLASS_NAMES),
        hidden=int(cfg["model"]["lstm_hidden"]),
        layers=int(cfg["model"]["lstm_layers"]),
        bi=bool(cfg["model"]["bidirectional"]),
    )

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Device:", device)
    model.to(device)

    # ----------------------------- Optimizer & Scheduler -----------------------------
    cfg_lr = float(cfg["train"]["lr"]) if args.lr is None else float(args.lr)
    opt = optim.Adam(model.parameters(), lr=cfg_lr, weight_decay=args.weight_decay)

    # Remove "verbose" because some PyTorch MPS builds don't support it
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=args.min_lr
    )

    # ----------------------------- Loss -----------------------------
    if args.no_class_weights:
        crit = nn.CrossEntropyLoss()
    else:
        class_weights = compute_class_weights(
            train_root=os.path.join(cfg["data"]["root"], "train"),
            view=cfg["data"]["view"],
        ).to(device)
        crit = nn.CrossEntropyLoss(weight=class_weights)

    # ----------------------------- Training Loop -----------------------------
    best_f1 = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("dist", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # ----------------------------- Validation -----------------------------
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                pred = model(x).argmax(1).cpu()
                y_pred.extend(pred.tolist())
                y_true.extend(y.tolist())

        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
        avg_loss = total_loss / max(1, len(train_dl))

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | acc={acc:.3f} | macroF1={f1:.3f} | "
              f"train={len(train_ds)} | val={len(val_ds)} | lr={opt.param_groups[0]['lr']:.2e}")

        if args.confmat and y_true:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
            print("Confusion Matrix:\n", cm)

        sched.step(f1)

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "checkpoints/feat_lstm_best.pt")

    # ----------------------------- Export -----------------------------
    import shutil
    if os.path.exists("checkpoints/feat_lstm_best.pt"):
        shutil.copy("checkpoints/feat_lstm_best.pt", "dist/model.pt")
        print("✅ Saved dist/model.pt (best feature LSTM model)")


if __name__ == "__main__":
    main()