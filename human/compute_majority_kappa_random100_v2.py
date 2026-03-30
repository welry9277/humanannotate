"""
Compute majority-vote vs gold Cohen's kappa (binary) for Random 100 v2.

Majority vote uses 4 specified annotator files:
  - human/human_labels_jeongju.json
  - human/human_labels_bogyung.json
  - human/human_labels_minwoo.json
  - human/human_labels_dongwook.json

Gold:
  streamlit_hazard_correct_labeler/data/random_100_samples_v2_with_hazard_correct.json

Metrics:
  1) All-annotators voting accuracy + Cohen's kappa (only keys where all 4 have labels)
  2) >=2-votes accuracy + Cohen's kappa (only keys where at least 2 annotators have labels)

Run:
  python human/compute_majority_kappa_random100_v2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Key = Tuple[str, int]  # (normalized source_jsonl_path, idx)


def _norm_src(p: str) -> str:
    return str(p).replace("\\", "/").strip()


def load_gold_map(gold_path: Path) -> Dict[Key, bool]:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Gold JSON should be a list.")

    gold_map: Dict[Key, bool] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        src = row.get("source_jsonl_path")
        idx = row.get("idx")
        hz = row.get("hazard_correct")
        if src is None or idx is None or hz is None:
            continue
        gold_map[(_norm_src(src), int(idx))] = bool(hz)
    return gold_map


def load_human_label_map(human_json_path: Path) -> Dict[Key, bool]:
    data = json.loads(human_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Human JSON should be a list: {human_json_path}")

    m: Dict[Key, bool] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        src = row.get("source_jsonl_path")
        idx = row.get("idx")
        hz = row.get("human_hazard_correct")
        if src is None or idx is None or hz is None:
            continue
        m[(_norm_src(src), int(idx))] = bool(hz)
    return m


def majority_vote(votes: List[bool]) -> bool:
    # Binary majority: pred True if True_votes >= 2 (for odd counts this is unambiguous).
    # For even counts, this makes ties => False.
    true_votes = sum(1 for v in votes if v)
    return true_votes > (len(votes) - true_votes)


def cohens_kappa_binary(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths must match")
    n = len(y_true)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "kappa": 0.0}

    correct = sum(1 for a, b in zip(y_true, y_pred) if bool(a) == bool(b))
    accuracy = correct / n

    pred_pos = sum(1 for v in y_pred if v) / n
    true_pos = sum(1 for v in y_true if v) / n
    pred_neg = 1.0 - pred_pos
    true_neg = 1.0 - true_pos

    p_o = accuracy
    p_e = pred_pos * true_pos + pred_neg * true_neg

    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) != 0 else 0.0
    return {
        "n": n,
        "accuracy": accuracy,
        "kappa": kappa,
        "pred_pos_rate": pred_pos,
        "true_pos_rate": true_pos,
        "p_o": p_o,
        "p_e": p_e,
    }


def evaluate_majority(
    gold_map: Dict[Key, bool],
    annotator_maps: List[Dict[Key, bool]],
    mode: str,
) -> Dict[str, Any]:
    """
    mode:
      - "all3": only keys where all annotators have labels
      - "ge2": only keys where at least 2 annotators have labels
    """
    if mode not in {"all3", "ge2"}:
        raise ValueError("Invalid mode")

    y_true: List[bool] = []
    y_pred: List[bool] = []

    annotator_count = len(annotator_maps)

    for k, gold_hz in gold_map.items():
        available_votes: List[bool] = []
        for m in annotator_maps:
            if k in m:
                available_votes.append(m[k])

        if mode == "all3":
            if len(available_votes) != annotator_count:
                continue
        else:
            if len(available_votes) < 2:
                continue

        # Majority vote among available votes.
        # If only 2 are available and they disagree, pred becomes False due to majority rule.
        pred = majority_vote(available_votes)

        y_true.append(bool(gold_hz))
        y_pred.append(bool(pred))

    return cohens_kappa_binary(y_true, y_pred)


def evaluate_single_annotator(
    gold_map: Dict[Key, bool],
    annotator_map: Dict[Key, bool],
) -> Dict[str, Any]:
    """Compute annotator vs gold metrics on matched keys only."""
    y_true: List[bool] = []
    y_pred: List[bool] = []

    for k, gold_hz in gold_map.items():
        if k not in annotator_map:
            continue
        y_true.append(bool(gold_hz))
        y_pred.append(bool(annotator_map[k]))

    metrics = cohens_kappa_binary(y_true, y_pred)
    metrics["gold_total"] = len(gold_map)
    metrics["missing_keys"] = len(gold_map) - metrics["n"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to random_100_samples_v2_with_hazard_correct.json",
    )
    parser.add_argument(
        "--human_jeongju",
        type=str,
        default=None,
        help="Path to human_labels_jeongju.json",
    )
    parser.add_argument(
        "--human_bogyung",
        type=str,
        default=None,
        help="Path to human_labels_bogyung.json",
    )
    parser.add_argument(
        "--human_minwoo",
        type=str,
        default=None,
        help="Path to human_labels_minwoo.json",
    )
    parser.add_argument(
        "--human_dongwook",
        type=str,
        default=None,
        help="Path to human_labels_dongwook.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    gold_path = (
        Path(args.gold)
        if args.gold
        else repo_root
        / "streamlit_hazard_correct_labeler"
        / "data"
        / "random_100_samples_v2_with_hazard_correct.json"
    )

    human_jeongju = Path(args.human_jeongju) if args.human_jeongju else repo_root / "human" / "human_labels_jeongju.json"
    human_bogyung = (
        Path(args.human_bogyung)
        if args.human_bogyung
        else repo_root / "human" / "human_labels_bogyung.json"
    )
    human_minwoo = (
        Path(args.human_minwoo)
        if args.human_minwoo
        else repo_root / "human" / "human_labels_minwoo.json"
    )
    human_dongwook = (
        Path(args.human_dongwook)
        if args.human_dongwook
        else repo_root / "human" / "human_labels_dongwook.json"
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = repo_root / "human" / "majority_kappa_random100_v2_results.json"

    gold_map = load_gold_map(gold_path)
    jeongju_map = load_human_label_map(human_jeongju)
    bogyung_map = load_human_label_map(human_bogyung)
    minwoo_map = load_human_label_map(human_minwoo)
    dongwook_map = load_human_label_map(human_dongwook)
    annotator_maps: List[Dict[Key, bool]] = [jeongju_map, bogyung_map, minwoo_map, dongwook_map]

    result = {
        "gold_path": str(gold_path),
        "human_paths": {
            "jeongju": str(human_jeongju),
            "bogyung": str(human_bogyung),
            "minwoo": str(human_minwoo),
            "dongwook": str(human_dongwook),
        },
        "majority_rule": {
            "n_annotators": 4,
            "pred_rule": "binary majority; ties => False (>=2 True votes wins)",
        },
        "metrics": {
            "all3": evaluate_majority(gold_map, annotator_maps, mode="all3"),
            "ge2": evaluate_majority(gold_map, annotator_maps, mode="ge2"),
            "per_annotator_vs_groundtruth": {
                "jeongju": evaluate_single_annotator(gold_map, jeongju_map),
                "bogyung": evaluate_single_annotator(gold_map, bogyung_map),
                "minwoo": evaluate_single_annotator(gold_map, minwoo_map),
                "dongwook": evaluate_single_annotator(gold_map, dongwook_map),
            },
        },
        "notes": [
            "all3 uses only keys present in all annotators.",
            "ge2 uses only keys where at least 2 annotators have labels.",
        ],
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote majority kappa results: {out_path}")


if __name__ == "__main__":
    main()

