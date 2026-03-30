"""
Read `data/random_100_samples.json` and write `data/random_100_samples_with_hazard_correct.json`
with `hazard_correct` copied from the aligned JSONL row (Sampling_aligned_triplets).

Run from repo root:
  python streamlit_hazard_correct_labeler/merge_hazard_correct_into_random100.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

WORKSPACE = Path(__file__).resolve().parents[1]
IN_PATH = WORKSPACE / "streamlit_hazard_correct_labeler" / "data" / "random_100_samples.json"
OUT_PATH = WORKSPACE / "streamlit_hazard_correct_labeler" / "data" / "random_100_samples_with_hazard_correct.json"


def main() -> None:
    rows: List[Dict[str, Any]] = json.loads(IN_PATH.read_text(encoding="utf-8"))
    cache: Dict[str, Dict[int, bool]] = {}

    out: List[Dict[str, Any]] = []
    for r in rows:
        rel = str(r["source_jsonl_path"]).replace("\\", "/")
        idx = int(r["idx"])
        if rel not in cache:
            path = WORKSPACE / rel
            m: Dict[int, bool] = {}
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "idx" in obj and "hazard_correct" in obj:
                        m[int(obj["idx"])] = bool(obj["hazard_correct"])
            cache[rel] = m
        gold = cache[rel].get(idx)
        item = dict(r)
        item["hazard_correct"] = gold
        out.append(item)

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = sum(1 for x in out if x["hazard_correct"] is None)
    print(f"Wrote {OUT_PATH.relative_to(WORKSPACE)} rows={len(out)} missing_gold={missing}")


if __name__ == "__main__":
    main()
