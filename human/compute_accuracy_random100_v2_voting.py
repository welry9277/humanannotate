"""
Compute per-annotator accuracy and 3-way voting accuracy (v2).

Gold:
  streamlit_hazard_correct_labeler/data/random_100_samples_v2_with_hazard_correct.json
Humans:
  human/human_labels_*.json (field: human_hazard_correct, annotator_id)
Voting:
  For each gold key (source_jsonl_path, idx), take available annotator labels.
  If tie among available votes -> predict False.
  Accuracy is computed on keys with at least 2 votes (>=2 annotators).

Outputs a JSON summary and per-key details.

Run from repo root:
  python human/compute_accuracy_random100_v2_voting.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Key = Tuple[str, int]  # (normalized source_jsonl_path, idx)


def _norm_src(p: str) -> str:
    return str(p).replace("\\", "/").strip()


def load_gold(gold_path: Path) -> Tuple[Dict[Key, bool], Dict[Key, Dict[str, Any]]]:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Gold JSON should be a list.")

    gold_map: Dict[Key, bool] = {}
    meta_map: Dict[Key, Dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        src = row.get("source_jsonl_path")
        idx = row.get("idx")
        hz = row.get("hazard_correct")
        if src is None or idx is None or hz is None:
            continue
        key: Key = (_norm_src(src), int(idx))
        gold_map[key] = bool(hz)
        meta_map[key] = row
    return gold_map, meta_map


def load_human_file(path: Path) -> Tuple[str, Dict[Key, bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Human JSON should be a list: {path}")

    items = [x for x in data if isinstance(x, dict)]
    annotator_id: Optional[str] = None
    for x in items:
        if isinstance(x.get("annotator_id"), str):
            annotator_id = x["annotator_id"]
            break
    if annotator_id is None:
        # Fallback to filename
        annotator_id = path.stem.replace("human_labels_", "")

    label_map: Dict[Key, bool] = {}
    for x in items:
        src = x.get("source_jsonl_path")
        idx = x.get("idx")
        hz = x.get("human_hazard_correct")
        if src is None or idx is None or hz is None:
            continue
        label_map[(_norm_src(src), int(idx))] = bool(hz)
    return annotator_id, label_map


def majority_vote(labels: List[bool]) -> Tuple[bool, bool, int, int]:
    """
    Returns:
      (pred, tie, true_votes, false_votes)
    Tie rule:
      if tie -> predict False
    """
    true_votes = sum(1 for x in labels if x)
    false_votes = len(labels) - true_votes
    tie = true_votes == false_votes
    pred = true_votes > false_votes
    return pred, tie, true_votes, false_votes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Gold JSON path",
    )
    parser.add_argument(
        "--human_dir",
        type=str,
        default=None,
        help="Directory containing human_labels_*.json",
    )
    parser.add_argument(
        "--human_pattern",
        type=str,
        default="human_labels_*.json",
        help="Glob pattern for human label files",
    )
    parser.add_argument(
        "--min_votes",
        type=int,
        default=2,
        help="Minimum available annotator votes required to include a key in voting accuracy.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path for summary + per-key details.",
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
    human_dir = Path(args.human_dir) if args.human_dir else repo_root / "human"

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = human_dir / "accuracy_random100_v2_voting_results.json"

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold not found: {gold_path}")
    if not human_dir.exists():
        raise FileNotFoundError(f"Human dir not found: {human_dir}")

    gold_map, gold_meta = load_gold(gold_path)
    if not gold_map:
        raise ValueError("Gold map is empty.")

    human_files = sorted(human_dir.glob(args.human_pattern))
    if not human_files:
        raise FileNotFoundError(f"No human label files matched: {human_dir} / {args.human_pattern}")

    annotator_maps: Dict[str, Dict[Key, bool]] = {}
    for hf in human_files:
        annotator_id, label_map = load_human_file(hf)
        annotator_maps[annotator_id] = label_map

    per_annotator: Dict[str, Any] = {}
    for annotator_id, label_map in annotator_maps.items():
        keys = set(label_map.keys())
        matched_keys = [k for k in gold_map.keys() if k in keys]
        missing_label_keys = [k for k in gold_map.keys() if k not in keys]

        correct = 0
        for k in matched_keys:
            if bool(label_map[k]) == bool(gold_map[k]):
                correct += 1

        acc = (correct / len(matched_keys)) if matched_keys else 0.0
        per_annotator[annotator_id] = {
            "matched": len(matched_keys),
            "gold_total": len(gold_map),
            "correct": correct,
            "accuracy": acc,
            "missing_gold_keys": [
                {"source_jsonl_path": gold_meta[k].get("source_jsonl_path"), "idx": gold_meta[k].get("idx")}
                for k in missing_label_keys[:50]
            ],
            "human_only_keys": 0,  # computed as labels with no matching gold; not needed for this report
        }

    # Voting over gold keys.
    voting_total = 0
    voting_correct = 0
    insufficient_votes = 0
    all_three_votes = 0
    two_votes = 0
    one_vote = 0
    tie_count = 0

    per_key_details: List[Dict[str, Any]] = []

    annotator_ids_sorted = sorted(annotator_maps.keys())
    for k, gold_hz in gold_map.items():
        true_labels: Dict[str, bool] = {}
        labels_available: List[bool] = []

        for aid in annotator_ids_sorted:
            if k in annotator_maps[aid]:
                v = bool(annotator_maps[aid][k])
                true_labels[aid] = v
                labels_available.append(v)

        n_avail = len(labels_available)
        if n_avail < args.min_votes:
            insufficient_votes += 1
            pred: Optional[bool] = None
            tie = False
            tv = 0
            fv = 0
        else:
            pred, tie, tv, fv = majority_vote(labels_available)
            voting_total += 1
            if pred == gold_hz:
                voting_correct += 1
            if tie:
                tie_count += 1
            if n_avail == 3:
                all_three_votes += 1
            elif n_avail == 2:
                two_votes += 1
            else:
                one_vote += 1

        per_key_details.append(
            {
                "source_jsonl_path": gold_meta[k].get("source_jsonl_path"),
                "idx": gold_meta[k].get("idx"),
                "gold_hazard_correct": gold_hz,
                "annotator_votes": {aid: true_labels.get(aid) for aid in annotator_ids_sorted},
                "n_votes_available": n_avail,
                "pred_voting": pred,
                "tie": tie,
                "true_votes": tv,
                "false_votes": fv,
            }
        )

    voting_accuracy = (voting_correct / voting_total) if voting_total else 0.0

    result = {
        "gold_path": str(gold_path),
        "human_dir": str(human_dir),
        "human_files": [str(p) for p in human_files],
        "annotators": annotator_ids_sorted,
        "min_votes_for_voting_accuracy": args.min_votes,
        "per_annotator": per_annotator,
        "voting": {
            "included_keys": voting_total,
            "correct": voting_correct,
            "accuracy": voting_accuracy,
            "insufficient_votes": insufficient_votes,
            "n_all_three_votes": all_three_votes,
            "n_two_votes": two_votes,
            "n_one_vote": one_vote,
            "n_tie_cases": tie_count,
        },
        "per_key_details": per_key_details,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote voting+accuracy results: {out_path}")
    print(
        "Voting accuracy:",
        voting_correct,
        "/",
        voting_total,
        "=", f"{voting_accuracy:.4f}",
    )


if __name__ == "__main__":
    main()

