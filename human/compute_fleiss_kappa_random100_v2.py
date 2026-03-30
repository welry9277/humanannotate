"""
Compute Fleiss' kappa for 4 human annotators (binary labels).

Annotators (default):
  - human/human_labels_jeongju.json
  - human/human_labels_bogyung.json
  - human/human_labels_minwoo.json
  - human/human_labels_dongwook.json

Uses (source_jsonl_path, idx) as item key.
By default, computes Fleiss' kappa on items where all 4 annotators have labels.

Run:
  python human/compute_fleiss_kappa_random100_v2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


Key = Tuple[str, int]


def _norm_src(p: str) -> str:
    return str(p).replace("\\", "/").strip()


def load_human_label_map(path: Path) -> Dict[Key, bool]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Human JSON should be a list: {path}")
    out: Dict[Key, bool] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        src = row.get("source_jsonl_path")
        idx = row.get("idx")
        hz = row.get("human_hazard_correct")
        if src is None or idx is None or hz is None:
            continue
        out[(_norm_src(src), int(idx))] = bool(hz)
    return out


def fleiss_kappa_binary(vote_rows: List[Tuple[int, int]], n_raters: int) -> Dict[str, Any]:
    """
    vote_rows: list of (n_false, n_true) counts per item.
    n_raters: fixed number of raters per item.
    """
    n_items = len(vote_rows)
    if n_items == 0:
        return {"n_items": 0, "n_raters": n_raters, "kappa": 0.0}

    # Category proportions p_j
    total_false = sum(r[0] for r in vote_rows)
    total_true = sum(r[1] for r in vote_rows)
    total_votes = n_items * n_raters
    p_false = total_false / total_votes
    p_true = total_true / total_votes

    # Per-item agreement P_i
    # P_i = (1/(n(n-1))) * sum_j n_ij(n_ij-1)
    denom = n_raters * (n_raters - 1)
    p_i_vals: List[float] = []
    for n_false, n_true in vote_rows:
        p_i = ((n_false * (n_false - 1)) + (n_true * (n_true - 1))) / denom
        p_i_vals.append(p_i)

    p_bar = sum(p_i_vals) / n_items
    p_e = (p_false * p_false) + (p_true * p_true)
    kappa = (p_bar - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0.0

    return {
        "n_items": n_items,
        "n_raters": n_raters,
        "kappa": kappa,
        "p_bar": p_bar,
        "p_e": p_e,
        "category_proportions": {"false": p_false, "true": p_true},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_jeongju", type=str, default=None)
    parser.add_argument("--human_bogyung", type=str, default=None)
    parser.add_argument("--human_minwoo", type=str, default=None)
    parser.add_argument("--human_dongwook", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    human_jeongju = Path(args.human_jeongju) if args.human_jeongju else repo_root / "human" / "human_labels_jeongju.json"
    human_bogyung = Path(args.human_bogyung) if args.human_bogyung else repo_root / "human" / "human_labels_bogyung.json"
    human_minwoo = Path(args.human_minwoo) if args.human_minwoo else repo_root / "human" / "human_labels_minwoo.json"
    human_dongwook = Path(args.human_dongwook) if args.human_dongwook else repo_root / "human" / "human_labels_dongwook.json"
    out_path = Path(args.out) if args.out else repo_root / "human" / "fleiss_kappa_random100_v2_results.json"

    m_jeongju = load_human_label_map(human_jeongju)
    m_bogyung = load_human_label_map(human_bogyung)
    m_minwoo = load_human_label_map(human_minwoo)
    m_dongwook = load_human_label_map(human_dongwook)

    common_keys = sorted(
        set(m_jeongju.keys())
        & set(m_bogyung.keys())
        & set(m_minwoo.keys())
        & set(m_dongwook.keys())
    )
    vote_rows: List[Tuple[int, int]] = []
    for k in common_keys:
        votes = [m_jeongju[k], m_bogyung[k], m_minwoo[k], m_dongwook[k]]
        n_true = sum(1 for v in votes if v)
        n_false = 4 - n_true
        vote_rows.append((n_false, n_true))

    fleiss = fleiss_kappa_binary(vote_rows, n_raters=4)
    result = {
        "human_paths": {
            "jeongju": str(human_jeongju),
            "bogyung": str(human_bogyung),
            "minwoo": str(human_minwoo),
            "dongwook": str(human_dongwook),
        },
        "common_item_count_all4": len(common_keys),
        "fleiss_kappa_all4_items": fleiss,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Fleiss kappa results: {out_path}")
    print(f"Fleiss' kappa (all4 common items): {fleiss['kappa']:.6f}")


if __name__ == "__main__":
    main()

