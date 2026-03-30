"""
Rebuild `data/random_100_samples.json` with ~50/50 hazard_correct from aligned JSONL.

Sampling tries to **minimize repeated `groundtruth_hazard` text** within each class:
always extend from the groundtruth string that has been picked the fewest times so far
(among rows still available). True/false each need 50 rows; the true pool only has
~32 distinct groundtruth strings, so some GT repetition on the true side is unavoidable.

Run from repo root:
  python streamlit_hazard_correct_labeler/build_balanced_random100.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

WORKSPACE = Path(__file__).resolve().parents[1]
SAMPLING_ROOT = WORKSPACE / "Sampling_aligned_triplets_v2"
OUT_PATH = WORKSPACE / "streamlit_hazard_correct_labeler" / "data" / "random_100_samples_v2.json"

SEED = 42
TARGET_TOTAL = 100
TARGET_PER_CLASS = 50


def _is_usable_row(obj: Dict[str, Any]) -> bool:
    if "idx" not in obj or "groundtruth_hazard" not in obj:
        return False
    v = obj.get("response_hazard")
    if v is None:
        return False
    if isinstance(v, str) and v.strip().lower() == "none":
        return False
    return True


def collect_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    true_rows: List[Dict[str, Any]] = []
    false_rows: List[Dict[str, Any]] = []
    for path in sorted(SAMPLING_ROOT.rglob("*.jsonl")):
        rel = path.relative_to(WORKSPACE)
        rel_s = str(rel).replace("\\", "/")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not _is_usable_row(obj):
                    continue
                row = {
                    "source_jsonl_path": rel_s,
                    "idx": int(obj["idx"]),
                    "groundtruth_hazard": str(obj["groundtruth_hazard"]),
                    "response_hazard": obj.get("response_hazard"),
                }
                if bool(obj.get("hazard_correct")):
                    true_rows.append(row)
                else:
                    false_rows.append(row)
    return true_rows, false_rows


def sample_min_gt_overlap(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Pick n rows; prefer spreading across distinct `groundtruth_hazard` strings."""
    by_gt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_gt[str(r["groundtruth_hazard"])].append(r)
    for gt in by_gt:
        rng.shuffle(by_gt[gt])

    ptr: Dict[str, int] = {gt: 0 for gt in by_gt}
    use_count: Dict[str, int] = defaultdict(int)
    out: List[Dict[str, Any]] = []

    while len(out) < n:
        candidates = [gt for gt in by_gt if ptr[gt] < len(by_gt[gt])]
        if not candidates:
            break
        candidates.sort(key=lambda gt: (use_count[gt], rng.random()))
        gt = candidates[0]
        out.append(by_gt[gt][ptr[gt]])
        ptr[gt] += 1
        use_count[gt] += 1

    if len(out) < n:
        raise RuntimeError(f"Could only pick {len(out)}/{n} rows (pool exhausted).")
    return out


def _gt_overlap_stats(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    from collections import Counter

    c = Counter(str(r["groundtruth_hazard"]) for r in rows)
    return len(c), max(c.values())


def main() -> None:
    true_rows, false_rows = collect_rows()
    n_t, n_f = len(true_rows), len(false_rows)
    if n_t < TARGET_PER_CLASS or n_f < TARGET_PER_CLASS:
        raise SystemExit(
            f"Need at least {TARGET_PER_CLASS} true and {TARGET_PER_CLASS} false; got {n_t} true, {n_f} false."
        )

    rng = random.Random(SEED)
    picked_t = sample_min_gt_overlap(true_rows, TARGET_PER_CLASS, rng)
    picked_f = sample_min_gt_overlap(false_rows, TARGET_PER_CLASS, rng)
    picked = picked_t + picked_f
    picked.sort(key=lambda r: (r["source_jsonl_path"], r["idx"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)
        f.write("\n")

    u_all, mx_all = _gt_overlap_stats(picked)
    u_t, mx_t = _gt_overlap_stats(picked_t)
    u_f, mx_f = _gt_overlap_stats(picked_f)

    print(f"Wrote {OUT_PATH.relative_to(WORKSPACE)}")
    print(f"Pool: true={n_t} false={n_f} | sample: true={TARGET_PER_CLASS} false={TARGET_PER_CLASS} seed={SEED}")
    print(
        f"Unique groundtruth_hazard: all={u_all} (max repeats {mx_all}); "
        f"true-half={u_t} (max {mx_t}); false-half={u_f} (max {mx_f})"
    )


if __name__ == "__main__":
    main()
