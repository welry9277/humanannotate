import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_aligned_jsonl_files(root: Path) -> List[Path]:
    aligned_root = root / "Sampling_aligned_triplets"
    if not aligned_root.exists():
        return []
    return sorted(aligned_root.rglob("*.jsonl"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_has_keys(row: Dict[str, Any], keys: List[str]) -> bool:
    for k in keys:
        if k not in row:
            return False
    return True


def _sanitize_annotator_id(annotator_id: str) -> str:
    # Keep filesystem-safe, stable identifiers.
    return annotator_id.strip().replace(os.sep, "_").replace(":", "_") or "anonymous"


def src_labels_path(root: Path, source_jsonl_path: Path, annotator_id: str) -> Path:
    # Keep human labels per source file and per annotator (to avoid collisions).
    rel = source_jsonl_path.relative_to(root)
    safe = str(rel).replace(os.sep, "__").replace(":", "")
    annotator_safe = _sanitize_annotator_id(annotator_id)
    return (
        root
        / "streamlit_hazard_correct_labeler"
        / "human_labels"
        / annotator_safe
        / f"{safe}__labels.jsonl"
    )


def load_human_labels(labels_path: Path) -> Dict[int, Dict[str, Any]]:
    if not labels_path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("idx")
            if isinstance(idx, int):
                out[idx] = obj
    return out


def write_human_labels(labels_path: Path, labels_by_idx: Dict[int, Dict[str, Any]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    # Rewrite whole file to avoid duplicates when re-labeling the same idx.
    sorted_items = [labels_by_idx[i] for i in sorted(labels_by_idx.keys())]
    with labels_path.open("w", encoding="utf-8") as f:
        for obj in sorted_items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def update_source_hazard_correct(source_jsonl_path: Path, idx: int, new_value: bool) -> None:
    # Rewrite source file to update exactly one row.
    rows = load_jsonl(source_jsonl_path)
    changed = False
    for r in rows:
        if r.get("idx") == idx:
            r["hazard_correct"] = new_value
            changed = True
            break
    if not changed:
        raise KeyError(f"idx={idx} not found in {source_jsonl_path}")

    with source_jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def update_source_hazard_correct_batch(source_jsonl_path: Path, updates: Dict[int, bool]) -> None:
    """Rewrite one aligned jsonl, updating hazard_correct for a set of idx values."""
    if not updates:
        return
    rows = load_jsonl(source_jsonl_path)
    found_any = False
    for r in rows:
        idx = r.get("idx")
        if idx in updates:
            r["hazard_correct"] = bool(updates[idx])
            found_any = True
    if not found_any:
        raise KeyError(f"No matching idx found in {source_jsonl_path} for updates.")
    with source_jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass
class Item:
    idx: int
    groundtruth_hazard: str
    hazard_correct: bool
    response_hazard: Optional[str] = None


def to_item(row: Dict[str, Any]) -> Item:
    response_hazard = row.get("response_hazard")
    return Item(
        idx=int(row["idx"]),
        groundtruth_hazard=str(row["groundtruth_hazard"]),
        hazard_correct=bool(row["hazard_correct"]),
        response_hazard=str(response_hazard) if response_hazard is not None else None,
    )


def main() -> None:
    st.set_page_config(page_title="Hazard Correct Labeler", layout="wide")

    default_root = Path(os.environ.get("WORKSPACE_ROOT", "/home/dongwook/EMBGuard_outputs/web")).resolve()
    try:
        workspace_root = st.secrets.get("WORKSPACE_ROOT", None)
    except StreamlitSecretNotFoundError:
        workspace_root = None
    root = Path(workspace_root).resolve() if workspace_root else default_root

    st.title("Hazard Correct Human Labeler")
    st.caption("`Sampling_aligned_triplets`의 모든 샘플을 한 화면에서 보고 true/false를 선택합니다.")

    annotator_id = st.text_input("annotator_id", value=os.environ.get("ANNOTATOR_ID", "annotator_1"))
    note = st.text_input("Optional note (applied to all saved rows)", value="")
    overwrite_source = st.checkbox(
        "Overwrite source JSONL (hazard_correct)  [optional]",
        value=False,
        help="기본값은 꺼져 있습니다. 켜면 원본 aligned jsonl의 hazard_correct를 직접 수정합니다.",
    )

    files = find_aligned_jsonl_files(root)
    if not files:
        st.error(f"No jsonl files found under `{root / 'Sampling_aligned_triplets'}`")
        return

    required = ["idx", "groundtruth_hazard", "hazard_correct"]

    # Load all rows once.
    all_items: List[Tuple[Path, Item]] = []  # (source_jsonl_path, item)
    invalid_counts = 0
    for source_jsonl_path in files:
        rows = load_jsonl(source_jsonl_path)
        valid_rows = [r for r in rows if ensure_has_keys(r, required)]
        invalid_counts += len(rows) - len(valid_rows)
        if not valid_rows:
            continue
        for r in valid_rows:
            all_items.append((source_jsonl_path, to_item(r)))

    if not all_items:
        st.error("No valid rows found (required keys: idx, groundtruth_hazard, hazard_correct).")
        return

    st.info(f"Loaded {len(all_items)} rows. Invalid/skipped rows: {invalid_counts}.")

    # Load existing human labels per source file (per annotator).
    annotator_safe = _sanitize_annotator_id(annotator_id)
    labels_by_source_path: Dict[Path, Dict[int, Dict[str, Any]]] = {}
    labels_paths_by_source_path: Dict[Path, Path] = {}
    for source_jsonl_path in sorted(files):
        labels_path = src_labels_path(root, source_jsonl_path, annotator_safe)
        labels_paths_by_source_path[source_jsonl_path] = labels_path
        labels_by_source_path[source_jsonl_path] = load_human_labels(labels_path)

    # Render all samples.
    st.markdown("## Review (all samples)")
    choice_by_item_key: Dict[str, bool] = {}

    # Deterministic ordering so keys remain stable across reruns.
    all_items_sorted = sorted(all_items, key=lambda x: (str(x[0]), x[1].idx))
    for pos, (source_jsonl_path, item) in enumerate(all_items_sorted):
        saved_for_item = labels_by_source_path.get(source_jsonl_path, {}).get(item.idx)
        default_choice = saved_for_item.get("human_hazard_correct", False) if saved_for_item else False
        default_choice = bool(default_choice)

        # Keys are internal; idx/hazard_correct are not displayed.
        key = f"label::{annotator_safe}::{pos}"

        with st.expander(f"Sample {pos + 1}", expanded=False):
            st.text_area(
                "groundtruth_hazard",
                value=item.groundtruth_hazard,
                height=140,
                disabled=True,
                key=f"gt::{annotator_safe}::{pos}",
            )
            if item.response_hazard is not None:
                st.text_area(
                    "response_hazard",
                    value=item.response_hazard,
                    height=110,
                    disabled=True,
                    key=f"resp::{annotator_safe}::{pos}",
                )
            selected = st.radio(
                "Select (true/false)",
                options=[True, False],
                index=0 if default_choice else 1,
                format_func=lambda b: "true" if b else "false",
                horizontal=True,
                key=key,
            )
            choice_by_item_key[key] = bool(selected)

    st.divider()
    if st.button("Save all", type="primary"):
        # 1) Save human labels (source별로 한 번만 write)
        pos_to_item: Dict[int, Tuple[Path, Item]] = {i: it for i, it in enumerate(all_items_sorted)}
        for pos, (source_jsonl_path, item) in pos_to_item.items():
            key = f"label::{annotator_safe}::{pos}"
            selected = bool(st.session_state.get(key, False))
            labels_by_source_path[source_jsonl_path][item.idx] = {
                "source_jsonl_path": str(source_jsonl_path),
                "idx": item.idx,
                "groundtruth_hazard": item.groundtruth_hazard,
                "human_hazard_correct": selected,
                "annotator_id": annotator_safe,
                "note": note,
                "saved_at_utc": _now_utc_iso(),
            }

        for source_jsonl_path in sorted(labels_paths_by_source_path.keys()):
            write_human_labels(
                labels_paths_by_source_path[source_jsonl_path],
                labels_by_source_path[source_jsonl_path],
            )

        # 2) Optional overwrite source jsonl (aligned data)
        if overwrite_source:
            updates_by_source: Dict[Path, Dict[int, bool]] = {}
            for pos, (source_jsonl_path, item) in pos_to_item.items():
                updates_by_source.setdefault(source_jsonl_path, {})[item.idx] = bool(
                    st.session_state.get(f"label::{annotator_safe}::{pos}", False)
                )
            for source_jsonl_path, updates in updates_by_source.items():
                update_source_hazard_correct_batch(source_jsonl_path, updates)

        st.success("Saved.")

if __name__ == "__main__":
    main()

