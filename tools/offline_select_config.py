#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Stage-2 config selection from δ=0.30 wide-run LLM labels.

Premise (experiment plan §1 item 9 + §7):
  - Greedy decoding => LLM label for a window is fixed given (mode, k, evidence).
  - Narrower δ or budget B only changes which windows *apply* that label;
    others fall back to Stage-1 (θ).
  - Different k need separate wide runs (evidence differs).

Example:
  python tools/offline_select_config.py \\
    --mode precision \\
    --pred_csv output/logformer_preds_val.csv \\
    --gt_csv output/gt_val.csv \\
    --theta 0.53 \\
    --wide_run 5:path/to/rag_normal.csv \\
    --wide_raw 5:path/to/rag_normal_raw.json \\
    --out_dir /tmp/offline_precision
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import eval_final as evalmod  # noqa: E402

DEFAULT_DELTAS = [0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_BUDGETS = [0.05, 0.10, 0.20, 1.0]
F1_TIE = 0.005  # 0.5 percentage points
CONSTRAINT = 0.90


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_wide_spec(items: Sequence[str]) -> Dict[int, Path]:
    """Parse repeated 'k:path' into {k: Path}."""
    out: Dict[int, Path] = {}
    for raw in items:
        if ":" not in raw:
            raise SystemExit(f"[x] expected k:path, got: {raw}")
        k_s, path_s = raw.split(":", 1)
        k = int(k_s)
        p = Path(os.path.expanduser(path_s))
        if not p.exists():
            raise SystemExit(f"[x] wide run file not found: {p}")
        out[k] = p
    return out


def _load_stage1(pred_csv: Path, theta: float) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    if "anomaly_pred" not in df.columns and "is_anomaly_pred" in df.columns:
        df = df.rename(columns={"is_anomaly_pred": "anomaly_pred"})
    need = {"window_id", "prob"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[x] pred_csv needs columns {need}, got {list(df.columns)}")
    df = df[["window_id", "prob"] + (["anomaly_pred"] if "anomaly_pred" in df.columns else [])].copy()
    df["window_id"] = df["window_id"].astype(int)
    df["prob"] = df["prob"].astype(float)
    df["stage1"] = (df["prob"] >= float(theta)).astype(int)
    return df


def _load_gt(gt_csv: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_csv)
    if "window_id" not in gt.columns or "label" not in gt.columns:
        raise SystemExit("[x] gt_csv needs window_id,label")
    gt = gt[["window_id", "label"]].copy()
    gt["window_id"] = gt["window_id"].astype(int)
    gt["label"] = gt["label"].astype(int)
    return gt


def _load_called_ids(
    k: int,
    rag_csv: Path,
    raw_map: Dict[int, Path],
    called_map: Dict[int, Path],
    flips_map: Dict[int, Path],
) -> Set[int]:
    if k in called_map:
        data = json.loads(called_map[k].read_text(encoding="utf-8"))
        if isinstance(data, dict) and "window_ids" in data:
            return {int(x) for x in data["window_ids"]}
        if isinstance(data, list):
            return {int(x) for x in data}
        raise SystemExit(f"[x] bad called_json for k={k}")

    if k in raw_map:
        raw = json.loads(raw_map[k].read_text(encoding="utf-8"))
        ids = set()
        for item in raw:
            if isinstance(item, dict) and "window_id" in item:
                # Abnormal raw may mark llm_called
                if item.get("llm_called") is False:
                    continue
                ids.add(int(item["window_id"]))
        if ids:
            return ids

    if k in flips_map:
        flips = pd.read_csv(flips_map[k])
        if "window_id" in flips.columns:
            # flips alone under-count (non-flip LLM calls missing); still better than nothing
            flip_ids = {int(x) for x in flips["window_id"].tolist()}
        else:
            flip_ids = set()
    else:
        flip_ids = set()

    # Default: all rows in rag out_csv are treated as having a reusable LLM label
    # (correct when wide run only wrote LLM-called / candidate rows, or full val
    #  where non-called rows equal stage1 — applying llm label == stage1 is fine).
    rag = pd.read_csv(rag_csv)
    all_ids = {int(x) for x in rag["window_id"].tolist()}
    if flip_ids and not raw_map and k not in called_map:
        # Prefer union: rag rows that appear in raw would be ideal; without raw,
        # use all rag rows (smoke cand20 case).
        return all_ids
    return all_ids


def _load_llm_labels(rag_csv: Path, called: Set[int]) -> Dict[int, int]:
    rag = pd.read_csv(rag_csv)
    if "anomaly" not in rag.columns:
        raise SystemExit(f"[x] rag csv missing anomaly column: {rag_csv}")
    out: Dict[int, int] = {}
    for _, row in rag.iterrows():
        wid = int(row["window_id"])
        if wid in called:
            out[wid] = int(row["anomaly"])
    return out


def _eligible_mask(
    mode: str,
    stage1: pd.Series,
    prob: pd.Series,
    theta: float,
    delta: float,
) -> pd.Series:
    lo_p = float(theta)
    hi_p = min(1.0, float(theta) + float(delta))
    lo_r = max(0.0, float(theta) - float(delta))
    hi_r = float(theta)
    if mode == "precision":
        return (stage1 == 1) & (prob >= lo_p) & (prob <= hi_p)
    if mode == "recall":
        return (stage1 == 0) & (prob >= lo_r) & (prob < hi_r)
    raise ValueError(mode)


def _apply_budget(
    eligible_idx: List[int],
    probs: pd.Series,
    theta: float,
    budget: float,
    m_total: int,
) -> List[int]:
    if budget >= 1.0 - 1e-12:
        return eligible_idx
    if budget <= 0:
        return []
    cap = int(math.floor(float(budget) * int(m_total)))
    if cap <= 0:
        return []
    ranked = sorted(eligible_idx, key=lambda i: abs(float(probs.iloc[i]) - float(theta)))
    return ranked[:cap]


def _simulate(
    base: pd.DataFrame,
    llm: Dict[int, int],
    mode: str,
    theta: float,
    delta: float,
    budget: float,
) -> Tuple[pd.Series, Dict[str, int]]:
    """Return final predictions aligned to base index + stats."""
    elig = _eligible_mask(mode, base["stage1"], base["prob"], theta, delta)
    elig_idx = [i for i, ok in enumerate(elig.tolist()) if ok]
    kept_idx = _apply_budget(elig_idx, base["prob"], theta, budget, len(base))
    kept_wids = {int(base.iloc[i]["window_id"]) for i in kept_idx}

    finals = []
    n_llm = 0
    n_missing_llm = 0
    for _, row in base.iterrows():
        wid = int(row["window_id"])
        s1 = int(row["stage1"])
        if wid in kept_wids:
            if wid in llm:
                finals.append(int(llm[wid]))
                n_llm += 1
            else:
                finals.append(s1)
                n_missing_llm += 1
        else:
            finals.append(s1)
    stats = {
        "n_cand": len(elig_idx),
        "n_after_budget": len(kept_idx),
        "n_llm_applied": n_llm,
        "n_missing_llm": n_missing_llm,
    }
    return pd.Series(finals, index=base.index, dtype=int), stats


def _flip_stats(y_true: pd.Series, stage1: pd.Series, final: pd.Series) -> Dict[str, int]:
    flip = final != stage1
    benefit = int((flip & (final == y_true)).sum())
    harm = int((flip & (final != y_true)).sum())
    return {
        "flip_count": int(flip.sum()),
        "flip_benefit": benefit,
        "flip_harm": harm,
        "net_gain": benefit - harm,
    }


def _band_endpoints(mode: str, theta: float, delta: float) -> Dict[str, float]:
    if mode == "precision":
        return {
            "prob_lo": float(theta),
            "prob_hi": float(min(1.0, theta + delta)),
        }
    return {
        "prob_lo": float(max(0.0, theta - delta)),
        "prob_hi": float(theta),
    }


def run_grid(
    mode: str,
    base: pd.DataFrame,
    gt: pd.DataFrame,
    llm_by_k: Dict[int, Dict[int, int]],
    theta: float,
    deltas: List[float],
    budgets: List[float],
) -> pd.DataFrame:
    merged = base.merge(gt, on="window_id", how="inner")
    if len(merged) == 0:
        raise SystemExit("[x] no overlap between pred_csv and gt_csv")
    coverage = {
        k: len(set(llm_by_k[k]) & set(merged["window_id"])) for k in llm_by_k
    }

    rows = []
    for k, llm in sorted(llm_by_k.items()):
        for delta in deltas:
            for budget in budgets:
                final, stats = _simulate(merged, llm, mode, theta, delta, budget)
                m = evalmod._metrics(merged["label"], final)
                flips = _flip_stats(merged["label"], merged["stage1"], final)
                band = _band_endpoints(mode, theta, delta)
                # stage1 baseline metrics once reusable
                rows.append(
                    {
                        "mode": mode,
                        "k": int(k),
                        "delta": float(delta),
                        "budget": float(budget),
                        "theta": float(theta),
                        "prob_lo": band["prob_lo"],
                        "prob_hi": band["prob_hi"],
                        "llm_label_coverage": int(coverage[k]),
                        "n_eval": int(len(merged)),
                        **stats,
                        **m,
                        **flips,
                        "constraint_precision_ok": int(m["precision"] >= CONSTRAINT),
                        "constraint_recall_ok": int(m["recall"] >= CONSTRAINT),
                    }
                )
    return pd.DataFrame(rows)


def _stage1_metrics(base: pd.DataFrame, gt: pd.DataFrame) -> Dict[str, float]:
    merged = base.merge(gt, on="window_id", how="inner")
    return evalmod._metrics(merged["label"], merged["stage1"])


# Selection policies (default remains plan §7).
POLICY_SECTION7 = "section7"  # tie → smaller k, smaller δ
POLICY_MAX_F1 = "max_f1"  # within F1-tie band, prefer higher F1 first
POLICY_NET_GAIN_FIRST = "net_gain_first"  # require net_gain>0 at B=1, then F1
ALL_POLICIES = (POLICY_SECTION7, POLICY_MAX_F1, POLICY_NET_GAIN_FIRST)


def _constraint_pool(
    unlim: pd.DataFrame, mode: str
) -> Tuple[pd.DataFrame, str, bool, List[str]]:
    notes: List[str] = []
    if mode == "precision":
        constrained = unlim[unlim["precision"] >= CONSTRAINT]
        metric_name = "precision"
    else:
        constrained = unlim[unlim["recall"] >= CONSTRAINT]
        metric_name = "recall"
    constraint_met = len(constrained) > 0
    pool = constrained if constraint_met else unlim
    if not constraint_met:
        notes.append(
            f"No (δ,k) met {metric_name}>={CONSTRAINT:.2f} at B=1; fell back to max F1."
        )
    return pool, metric_name, constraint_met, notes


def _pick_budget(grid: pd.DataFrame, k_sel: int, delta_sel: float, f1_unlim: float) -> pd.Series:
    sub = grid[(grid["k"] == k_sel) & (grid["delta"] == delta_sel)].copy()
    ok_b = sub[sub["f1"] >= f1_unlim - F1_TIE].copy()
    if ok_b.empty:
        ok_b = sub.copy()
    ok_b = ok_b.sort_values(["budget", "f1"], ascending=[True, False])
    return ok_b.iloc[0]


def _pack_selected(
    mode: str,
    policy: str,
    pick_b: Mapping[str, Any],
    k_sel: int,
    delta_sel: float,
    metric_name: str,
    constraint_met: bool,
    notes: List[str],
    *,
    selected: bool = True,
    unavailable_reason: Optional[str] = None,
) -> Dict[str, Any]:
    budget_sel = float(pick_b["budget"]) if selected else None
    net_gain = int(pick_b.get("net_gain", 0)) if selected else None
    band = (
        _band_endpoints(mode, float(pick_b["theta"]), delta_sel)
        if selected
        else {"prob_lo": None, "prob_hi": None}
    )
    out: Dict[str, Any] = {
        "policy": policy,
        "selected": bool(selected),
        "mode": mode,
        "theta": float(pick_b["theta"]) if selected else None,
        "delta": delta_sel if selected else None,
        "k": k_sel if selected else None,
        "budget": budget_sel,
        "budget_unlimited": (
            bool(budget_sel is not None and budget_sel >= 1.0 - 1e-12) if selected else None
        ),
        "prob_lo": band["prob_lo"],
        "prob_hi": band["prob_hi"],
        "constraint_metric": metric_name,
        "constraint_threshold": CONSTRAINT,
        "constraint_met": bool(constraint_met),
        "f1_tie_tol": F1_TIE,
        "net_gain_warning": bool(selected and net_gain is not None and net_gain <= 0),
        "selection_notes": notes,
        "unavailable_reason": unavailable_reason,
    }
    if not selected:
        out["metrics"] = None
        out["cli_hints"] = None
        return out

    out["metrics"] = {
        "precision": float(pick_b["precision"]),
        "recall": float(pick_b["recall"]),
        "f1": float(pick_b["f1"]),
        "accuracy": float(pick_b["accuracy"]),
        "n": int(pick_b["n"]),
        "tp": int(pick_b["tp"]),
        "fp": int(pick_b["fp"]),
        "fn": int(pick_b["fn"]),
        "tn": int(pick_b["tn"]),
        "n_cand": int(pick_b["n_cand"]),
        "n_after_budget": int(pick_b["n_after_budget"]),
        "n_llm_applied": int(pick_b["n_llm_applied"]),
        "n_missing_llm": int(pick_b["n_missing_llm"]),
        "flip_count": int(pick_b["flip_count"]),
        "flip_benefit": int(pick_b["flip_benefit"]),
        "flip_harm": int(pick_b["flip_harm"]),
        "net_gain": int(net_gain),
        "llm_label_coverage": int(pick_b["llm_label_coverage"]),
        "n_eval": int(pick_b["n_eval"]),
    }
    out["cli_hints"] = {
        "precision": {
            "call_policy": "pred1_boundary",
            "base_threshold": float(pick_b["theta"]),
            "prob_lo": band["prob_lo"],
            "prob_hi_b": band["prob_hi"],
            "evidence_k": k_sel,
            "budget": None if budget_sel >= 1.0 - 1e-12 else budget_sel,
        },
        "recall": {
            "call_policy": "boundary",
            "prob0_min": band["prob_lo"],
            "prob0_max": band["prob_hi"],
            "evidence_k": k_sel,
            "top_k": 1,
            "budget": None if budget_sel >= 1.0 - 1e-12 else budget_sel,
        },
    }[mode]
    return out


def select_config(
    grid: pd.DataFrame, mode: str, policy: str = POLICY_SECTION7
) -> Dict[str, Any]:
    """Select (δ,k,B) under one policy. Default is experiment-plan §7."""
    if policy not in ALL_POLICIES:
        raise ValueError(f"unknown policy {policy}; expected one of {ALL_POLICIES}")

    unlim = grid[grid["budget"] >= 1.0 - 1e-12].copy()
    if unlim.empty:
        raise SystemExit("[x] grid has no unlimited-budget (B=1.0) rows")

    pool, metric_name, constraint_met, notes = _constraint_pool(unlim, mode)
    notes = list(notes)

    if policy == POLICY_NET_GAIN_FIRST:
        pos = pool[pool["net_gain"] > 0].copy()
        if pos.empty:
            notes.append(
                "net_gain_first: no (δ,k) with net_gain>0 at B=1 under constraint pool; "
                "no config selected (do not freeze for test)."
            )
            # Still return a stub for comparison tables.
            dummy = pool.sort_values("f1", ascending=False).iloc[0].to_dict()
            return _pack_selected(
                mode,
                policy,
                dummy,
                int(dummy["k"]),
                float(dummy["delta"]),
                metric_name,
                constraint_met,
                notes,
                selected=False,
                unavailable_reason="no_positive_net_gain_at_B1",
            )
        pool = pos
        notes.append("net_gain_first: restricted to net_gain>0 at B=1 before F1 ranking.")

    best_f1 = float(pool["f1"].max())
    tied = pool[pool["f1"] >= best_f1 - F1_TIE].copy()

    if policy == POLICY_SECTION7:
        # Plan §7: among near-ties, prefer smaller k then smaller δ.
        tied = tied.sort_values(["k", "delta", "f1"], ascending=[True, True, False])
        notes.append(
            f"section7: among F1 within {F1_TIE} of best, chose smallest k then smallest δ."
        )
    elif policy == POLICY_MAX_F1:
        # Prefer higher F1 inside the near-tie band; break remaining ties by smaller k, δ.
        tied = tied.sort_values(["f1", "k", "delta"], ascending=[False, True, True])
        notes.append(
            f"max_f1: among F1 within {F1_TIE} of best, prefer higher F1, then smaller k/δ."
        )
    else:  # POLICY_NET_GAIN_FIRST — same F1 ranking as section7 after net_gain filter
        tied = tied.sort_values(
            ["f1", "k", "delta"], ascending=[False, True, True]
        )
        notes.append(
            f"net_gain_first: among net_gain>0 and F1 within {F1_TIE} of best, "
            "prefer higher F1 then smaller k/δ."
        )

    pick_dk = tied.iloc[0]
    k_sel = int(pick_dk["k"])
    delta_sel = float(pick_dk["delta"])
    f1_unlim = float(pick_dk["f1"])

    pick_b = _pick_budget(grid, k_sel, delta_sel, f1_unlim)
    # For net_gain_first, also require final budgeted row net_gain>0 when possible.
    if policy == POLICY_NET_GAIN_FIRST:
        sub = grid[(grid["k"] == k_sel) & (grid["delta"] == delta_sel)].copy()
        ok_b = sub[
            (sub["f1"] >= f1_unlim - F1_TIE) & (sub["net_gain"] > 0)
        ].copy()
        if ok_b.empty:
            notes.append(
                "net_gain_first: no budget keeps net_gain>0 with F1 drop tolerance; "
                "no config selected."
            )
            return _pack_selected(
                mode,
                policy,
                pick_dk.to_dict(),
                k_sel,
                delta_sel,
                metric_name,
                constraint_met,
                notes,
                selected=False,
                unavailable_reason="no_positive_net_gain_under_budget",
            )
        pick_b = ok_b.sort_values(["budget", "f1"], ascending=[True, False]).iloc[0]
        notes.append(
            f"net_gain_first: among B with net_gain>0 and F1 drop <= {F1_TIE}, "
            f"chose smallest B={float(pick_b['budget'])}."
        )
    else:
        notes.append(
            f"Among B with F1 drop <= {F1_TIE} vs B=1, chose smallest B={float(pick_b['budget'])}."
        )

    selected_row = pick_b.to_dict()
    net_gain = int(selected_row.get("net_gain", 0))
    if net_gain <= 0:
        notes.append("net_gain_warning: net_gain <= 0 (plan fallback condition).")

    return _pack_selected(
        mode,
        policy,
        selected_row,
        k_sel,
        delta_sel,
        metric_name,
        constraint_met,
        notes,
        selected=True,
    )


def select_all_policies(grid: pd.DataFrame, mode: str) -> Dict[str, Any]:
    """Run §7 + two trial policies; §7 remains the primary freeze candidate."""
    by_policy = {p: select_config(grid, mode, policy=p) for p in ALL_POLICIES}
    summary_rows = []
    for p, cfg in by_policy.items():
        row = {
            "policy": p,
            "selected": cfg.get("selected"),
            "delta": cfg.get("delta"),
            "k": cfg.get("k"),
            "budget": cfg.get("budget"),
            "constraint_met": cfg.get("constraint_met"),
            "net_gain_warning": cfg.get("net_gain_warning"),
            "unavailable_reason": cfg.get("unavailable_reason"),
        }
        m = cfg.get("metrics") or {}
        for key in ("precision", "recall", "f1", "net_gain", "n_llm_applied"):
            row[key] = m.get(key)
        summary_rows.append(row)
    return {
        "primary_policy": POLICY_SECTION7,
        "policies": by_policy,
        "comparison": summary_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline select δ/k/B from δ=0.30 wide-run LLM labels"
    )
    ap.add_argument("--mode", choices=["precision", "recall"], required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument(
        "--wide_run",
        action="append",
        default=[],
        help="k:rag_out.csv (repeatable)",
    )
    ap.add_argument(
        "--wide_raw",
        action="append",
        default=[],
        help="k:raw.json listing called window_ids (preferred)",
    )
    ap.add_argument(
        "--wide_flips",
        action="append",
        default=[],
        help="k:flips.csv (optional hint)",
    )
    ap.add_argument(
        "--called_json",
        action="append",
        default=[],
        help="k:json with list or {window_ids:[...]}",
    )
    ap.add_argument(
        "--deltas",
        default=",".join(str(x) for x in DEFAULT_DELTAS),
    )
    ap.add_argument(
        "--budgets",
        default=",".join(str(x) for x in DEFAULT_BUDGETS),
        help="use 1.0 for unlimited",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if not args.wide_run:
        raise SystemExit("[x] provide at least one --wide_run k:path")
    wide = _parse_wide_spec(args.wide_run)
    raw_map = _parse_wide_spec(args.wide_raw) if args.wide_raw else {}
    flips_map = _parse_wide_spec(args.wide_flips) if args.wide_flips else {}
    called_map = _parse_wide_spec(args.called_json) if args.called_json else {}

    deltas = _parse_float_list(args.deltas)
    budgets = _parse_float_list(args.budgets)
    if 1.0 not in budgets and not any(b >= 1.0 - 1e-12 for b in budgets):
        budgets.append(1.0)

    base = _load_stage1(Path(args.pred_csv), args.theta)
    gt = _load_gt(Path(args.gt_csv))
    s1m = _stage1_metrics(base, gt)

    llm_by_k: Dict[int, Dict[int, int]] = {}
    called_stats = {}
    for k, rag_path in sorted(wide.items()):
        called = _load_called_ids(k, rag_path, raw_map, called_map, flips_map)
        llm_by_k[k] = _load_llm_labels(rag_path, called)
        called_stats[k] = {
            "n_called": len(called),
            "n_llm_labels": len(llm_by_k[k]),
            "rag_csv": str(rag_path),
        }

    grid = run_grid(args.mode, base, gt, llm_by_k, args.theta, deltas, budgets)
    multi = select_all_policies(grid, args.mode)
    selected = multi["policies"][POLICY_SECTION7]
    selected["stage1_metrics"] = s1m
    selected["wide_runs"] = called_stats
    selected["policy_variants"] = {
        p: multi["policies"][p] for p in ALL_POLICIES if p != POLICY_SECTION7
    }
    # Coverage vs δ=0.30 band candidates (not vs all val rows).
    # Formal wide runs only label band windows; warn if labels < that band size.
    n_eval = int(grid["n_eval"].iloc[0])
    max_cov = int(grid["llm_label_coverage"].max())
    band30 = grid[grid["delta"] >= 0.30 - 1e-12]
    n_band30 = int(band30["n_cand"].max()) if len(band30) else 0
    coverage_meta = {
        "llm_label_coverage": max_cov,
        "n_eval": n_eval,
        "n_cand_delta030": n_band30,
    }
    for cfg in multi["policies"].values():
        cfg.update(coverage_meta)
        cfg["stage1_metrics"] = s1m
        cfg["wide_runs"] = called_stats
    if n_band30 > 0 and max_cov < n_band30:
        warn = (
            f"LLM labels cover {max_cov}/{n_band30} δ=0.30 band candidates "
            f"(eval n={n_eval}); wide run incomplete — do not treat as formal freeze."
        )
        for cfg in multi["policies"].values():
            cfg["coverage_warning"] = warn
            cfg.setdefault("selection_notes", []).append(warn)
        selected["coverage_warning"] = warn

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "grid_metrics.csv"
    grid.to_csv(grid_path, index=False)
    cfg_path = out_dir / "selected_config.json"
    cfg_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compare_path = out_dir / "selection_compare.json"
    compare_path.write_text(
        json.dumps(
            {
                "primary_policy": POLICY_SECTION7,
                "comparison": multi["comparison"],
                "policies": multi["policies"],
                "stage1_metrics": s1m,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for p in (POLICY_MAX_F1, POLICY_NET_GAIN_FIRST):
        p_path = out_dir / f"selected_config_{p}.json"
        p_path.write_text(
            json.dumps(multi["policies"][p], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    readme = out_dir / "README_offline.txt"
    readme.write_text(
        "Offline selection from greedy δ=0.30 wide-run labels.\n"
        "Do not reuse labels across different evidence_k.\n"
        "Use same theta and split as the wide run. Formal freeze requires full-val coverage.\n"
        "\n"
        "Policies:\n"
        f"  - {POLICY_SECTION7} (primary / selected_config.json): plan §7; "
        "F1 near-tie -> smaller k then smaller δ.\n"
        f"  - {POLICY_MAX_F1}: F1 near-tie -> higher F1 first.\n"
        f"  - {POLICY_NET_GAIN_FIRST}: require net_gain>0 then F1; may be unavailable.\n"
        "See selection_compare.json for side-by-side results.\n",
        encoding="utf-8",
    )

    print(f"[✓] wrote {grid_path}  rows={len(grid)}")
    print(f"[✓] wrote {cfg_path}  (primary={POLICY_SECTION7})")
    print(f"[✓] wrote {compare_path}")
    for row in multi["comparison"]:
        if row["selected"]:
            print(
                f"[✓] policy={row['policy']}: δ={row['delta']} k={row['k']} "
                f"B={row['budget']} F1={row['f1']:.4f} net_gain={row['net_gain']}"
            )
        else:
            print(
                f"[w] policy={row['policy']}: unavailable ({row['unavailable_reason']})"
            )
    if selected.get("coverage_warning"):
        print(f"[w] {selected['coverage_warning']}")


if __name__ == "__main__":
    main()
