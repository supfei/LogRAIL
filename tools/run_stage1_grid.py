#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 grid: num_layers × lr, val θ sweep, select + θ±0.02 stability.

Example:
  python tools/run_stage1_grid.py \\
    --out_dir ~/lograil/runs/.../artifacts \\
    --npz_dir LogFormer/npz \\
    --ckpt_dir LogFormer/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_LOGFORMER = _REPO / "LogFormer"

LAYERS = [1, 2]
LRS = [1e-5, 3e-5, 1e-4]
F1_TIE = 0.005
STAB_DROP = 0.01  # unstable if F1 drops > this vs best within θ±0.02
EXTRA_SEEDS = [456, 789]


def _lr_tag(lr: float) -> str:
    """Match train_transformer.py suffix formatting: f'{lr}'."""
    return f"{lr}"


def _ckpt_name(log_name: str, mode: str, layers: int, dropout: float, lr: float, seed: int) -> str:
    return f"best_{log_name}_{mode}_{layers}L_do{dropout}_{_lr_tag(lr)}_seed{seed}.pt"


def _run(cmd: List[str], log_path: Path, cwd: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[>] {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit(f"[x] command failed ({p.returncode}): {' '.join(cmd)}\n--- tail ---\n{text[-2000:]}")
    return text


def _parse_train_argmax_f1(log_text: str) -> Optional[float]:
    m = re.search(r"best_val_argmax_f1=([0-9.]+)", log_text)
    return float(m.group(1)) if m else None


def _best_from_sweep(sweep_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(sweep_csv)
    if df.empty:
        raise SystemExit(f"[x] empty sweep: {sweep_csv}")
    i = int(df["f1"].idxmax())
    row = df.loc[i]
    return {
        "best_theta": float(row["threshold"]),
        "best_f1": float(row["f1"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "TN": int(row["TN"]),
        "FP": int(row["FP"]),
        "FN": int(row["FN"]),
        "TP": int(row["TP"]),
    }


def _stability(sweep_csv: Path, theta: float) -> Dict[str, Any]:
    df = pd.read_csv(sweep_csv)
    best_f1 = float(df["f1"].max())
    lo, hi = theta - 0.02, theta + 0.02
    band = df[(df["threshold"] >= lo - 1e-9) & (df["threshold"] <= hi + 1e-9)]
    if band.empty:
        return {
            "stable": False,
            "reason": "no thresholds in θ±0.02",
            "best_f1": best_f1,
            "min_f1_in_band": None,
            "max_drop": None,
            "band_rows": 0,
        }
    min_f1 = float(band["f1"].min())
    max_drop = best_f1 - min_f1
    return {
        "stable": bool(max_drop <= STAB_DROP),
        "reason": None if max_drop <= STAB_DROP else f"F1 drop {max_drop:.4f} > {STAB_DROP}",
        "best_f1": best_f1,
        "min_f1_in_band": min_f1,
        "max_drop": max_drop,
        "band_rows": int(len(band)),
        "band_thetas": [float(x) for x in band["threshold"].tolist()],
    }


def train_one(
    *,
    layers: int,
    lr: float,
    seed: int,
    out_dir: Path,
    npz_dir: Path,
    ckpt_dir: Path,
    log_name: str,
    mode: str,
    dropout: float,
    window_size: int,
    epochs: int,
    patience: int,
    batch: int,
    py: str,
) -> Dict[str, Any]:
    tag = f"L{layers}_lr{_lr_tag(lr)}_seed{seed}"
    train_log = out_dir / f"train_{tag}.log"
    ckpt = ckpt_dir / _ckpt_name(log_name, mode, layers, dropout, lr, seed)
    cmd = [
        py,
        str(_LOGFORMER / "train_transformer.py"),
        "--log_name",
        log_name,
        "--mode",
        mode,
        "--num_layers",
        str(layers),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--dropout",
        str(dropout),
        "--window_size",
        str(window_size),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch",
        str(batch),
        "--npz_dir",
        str(npz_dir),
        "--ckpt_dir",
        str(ckpt_dir),
    ]
    log_text = _run(cmd, train_log, _LOGFORMER)
    if not ckpt.exists():
        raise SystemExit(f"[x] expected ckpt missing: {ckpt}")
    sweep_csv = out_dir / f"sweep_{tag}.csv"
    sweep_log = out_dir / f"sweep_{tag}.log"
    scmd = [
        py,
        str(_LOGFORMER / "sweep_threshold.py"),
        "--ckpt",
        str(ckpt),
        "--npz_dir",
        str(npz_dir),
        "--split",
        "val",
        "--mode",
        mode,
        "--num_layers",
        str(layers),
        "--dropout",
        str(dropout),
        "--window_size",
        str(window_size),
        "--thr_start",
        "0.05",
        "--thr_end",
        "0.95",
        "--thr_step",
        "0.01",
        "--out_csv",
        str(sweep_csv),
    ]
    _run(scmd, sweep_log, _LOGFORMER)
    best = _best_from_sweep(sweep_csv)
    row = {
        "num_layers": layers,
        "lr": lr,
        "seed": seed,
        "ckpt": str(ckpt),
        "sweep_csv": str(sweep_csv),
        "train_log": str(train_log),
        "train_argmax_f1": _parse_train_argmax_f1(log_text),
        **best,
    }
    print(
        f"[✓] {tag}: θ={best['best_theta']:.2f} F1={best['best_f1']:.4f} "
        f"P={best['precision']:.4f} R={best['recall']:.4f}",
        flush=True,
    )
    return row


def select_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    best_f1 = float(df["best_f1"].max())
    tied = df[df["best_f1"] >= best_f1 - F1_TIE].copy()
    tied = tied.sort_values(
        ["num_layers", "lr", "best_f1", "seed"],
        ascending=[True, True, False, True],
    )
    return tied.iloc[0].to_dict()


def write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    cols = [
        "num_layers",
        "lr",
        "seed",
        "best_theta",
        "best_f1",
        "precision",
        "recall",
        "train_argmax_f1",
        "TN",
        "FP",
        "FN",
        "TP",
        "ckpt",
        "sweep_csv",
        "train_log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1 L×lr grid + val θ select")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--npz_dir", default=str(_LOGFORMER / "npz"))
    ap.add_argument("--ckpt_dir", default=str(_LOGFORMER / "checkpoints"))
    ap.add_argument("--log_name", default="android_time")
    ap.add_argument("--mode", default="classifier")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123, help="Primary training seed for the 6-grid")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--skip_train_if_ckpt",
        action="store_true",
        help="If ckpt exists, only re-sweep (debug); default always retrain",
    )
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Absolute paths: train/sweep are launched with cwd=LogFormer/.
    npz_dir = Path(args.npz_dir).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    py = args.python
    if not (npz_dir / "train.npz").exists():
        raise SystemExit(f"[x] train.npz not found under {npz_dir}")

    rows: List[Dict[str, Any]] = []
    for L in LAYERS:
        for lr in LRS:
            tag = f"L{L}_lr{_lr_tag(lr)}_seed{args.seed}"
            ckpt = ckpt_dir / _ckpt_name(
                args.log_name, args.mode, L, args.dropout, lr, args.seed
            )
            if args.skip_train_if_ckpt and ckpt.exists():
                sweep_csv = out_dir / f"sweep_{tag}.csv"
                if not sweep_csv.exists():
                    scmd = [
                        py,
                        str(_LOGFORMER / "sweep_threshold.py"),
                        "--ckpt",
                        str(ckpt),
                        "--npz_dir",
                        str(npz_dir),
                        "--split",
                        "val",
                        "--mode",
                        args.mode,
                        "--num_layers",
                        str(L),
                        "--dropout",
                        str(args.dropout),
                        "--window_size",
                        str(args.window_size),
                        "--out_csv",
                        str(sweep_csv),
                    ]
                    _run(scmd, out_dir / f"sweep_{tag}.log", _LOGFORMER)
                best = _best_from_sweep(sweep_csv)
                rows.append(
                    {
                        "num_layers": L,
                        "lr": lr,
                        "seed": args.seed,
                        "ckpt": str(ckpt),
                        "sweep_csv": str(sweep_csv),
                        "train_log": "",
                        "train_argmax_f1": None,
                        **best,
                    }
                )
                continue
            rows.append(
                train_one(
                    layers=L,
                    lr=lr,
                    seed=args.seed,
                    out_dir=out_dir,
                    npz_dir=npz_dir,
                    ckpt_dir=ckpt_dir,
                    log_name=args.log_name,
                    mode=args.mode,
                    dropout=args.dropout,
                    window_size=args.window_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch=args.batch,
                    py=py,
                )
            )

    summary_path = out_dir / "grid_summary.csv"
    write_summary(rows, summary_path)
    print(f"[✓] wrote {summary_path}", flush=True)

    pick = select_best(rows)
    stab = _stability(Path(pick["sweep_csv"]), float(pick["best_theta"]))
    notes = [
        f"Among configs with F1 within {F1_TIE} of best, prefer smaller num_layers then smaller lr.",
        f"Stability θ±0.02: stable={stab['stable']} max_drop={stab.get('max_drop')}",
    ]

    # If unstable, retrain winning (L,lr) with extra seeds and re-select among them.
    seed_rows = [r for r in rows if r["num_layers"] == pick["num_layers"] and abs(r["lr"] - pick["lr"]) < 1e-15]
    if not stab["stable"]:
        notes.append(f"Unstable under seed={args.seed}: {stab.get('reason')}; training extra seeds {EXTRA_SEEDS}.")
        for s in EXTRA_SEEDS:
            seed_rows.append(
                train_one(
                    layers=int(pick["num_layers"]),
                    lr=float(pick["lr"]),
                    seed=s,
                    out_dir=out_dir,
                    npz_dir=npz_dir,
                    ckpt_dir=ckpt_dir,
                    log_name=args.log_name,
                    mode=args.mode,
                    dropout=args.dropout,
                    window_size=args.window_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch=args.batch,
                    py=py,
                )
            )
        # re-select among seeds for this (L,lr) only
        pick = select_best(seed_rows)
        stab = _stability(Path(pick["sweep_csv"]), float(pick["best_theta"]))
        notes.append(
            f"After seed search, selected seed={int(pick['seed'])} "
            f"θ={float(pick['best_theta']):.2f} F1={float(pick['best_f1']):.4f} "
            f"stable={stab['stable']}"
        )
        # refresh full summary including extras
        all_rows = {(r["num_layers"], r["lr"], r["seed"]): r for r in rows}
        for r in seed_rows:
            all_rows[(r["num_layers"], r["lr"], r["seed"])] = r
        write_summary(list(all_rows.values()), summary_path)

    selected = {
        "num_layers": int(pick["num_layers"]),
        "lr": float(pick["lr"]),
        "seed": int(pick["seed"]),
        "dropout": float(args.dropout),
        "mode": args.mode,
        "window_size": int(args.window_size),
        "log_name": args.log_name,
        "ckpt": pick["ckpt"],
        "theta": float(pick["best_theta"]),
        "val_metrics": {
            "precision": float(pick["precision"]),
            "recall": float(pick["recall"]),
            "f1": float(pick["best_f1"]),
            "TN": int(pick["TN"]),
            "FP": int(pick["FP"]),
            "FN": int(pick["FN"]),
            "TP": int(pick["TP"]),
            "train_argmax_f1": pick.get("train_argmax_f1"),
        },
        "stability": stab,
        "f1_tie_tol": F1_TIE,
        "stability_drop_tol": STAB_DROP,
        "selection_notes": notes,
        "grid_summary_csv": str(summary_path),
        "npz_dir": str(npz_dir),
        "no_test_eval": True,
    }
    sel_path = out_dir / "selected_stage1.json"
    sel_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] wrote {sel_path}", flush=True)
    print(
        f"[✓] SELECTED L={selected['num_layers']} lr={selected['lr']} seed={selected['seed']} "
        f"θ={selected['theta']:.2f} F1={selected['val_metrics']['f1']:.4f} "
        f"stable={stab['stable']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
