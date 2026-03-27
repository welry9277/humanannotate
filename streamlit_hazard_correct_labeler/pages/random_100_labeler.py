import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from app import (
    _now_utc_iso,
    _sanitize_annotator_id,
    _source_key,
    _supabase_config,
    load_human_labels,
    load_human_labels_from_db,
    src_labels_path,
    write_human_labels,
    write_human_labels_to_db,
)


def _resolve_root() -> Path:
    inferred_default_root = Path(__file__).resolve().parents[2]
    default_root = Path(os.environ.get("WORKSPACE_ROOT", str(inferred_default_root))).resolve()
    try:
        workspace_root = st.secrets.get("WORKSPACE_ROOT", None)
    except StreamlitSecretNotFoundError:
        workspace_root = None
    return Path(workspace_root).resolve() if workspace_root else default_root


def _load_random_subset(root: Path) -> List[Dict[str, Any]]:
    subset_path = root / "streamlit_hazard_correct_labeler" / "data" / "random_100_samples.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    with subset_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("Subset JSON should be a list.")
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(row)
    return out


def main() -> None:
    st.set_page_config(page_title="Random 100 Labeler", layout="wide")
    root = _resolve_root()
    db_cfg = _supabase_config()

    st.title("Hazard Correct Labeler (Random 100)")
    st.caption("고정 랜덤 100개 샘플만 라벨링합니다. (`data/random_100_samples.json` 기반)")

    annotator_id = st.text_input("annotator_id", value=os.environ.get("ANNOTATOR_ID", "annotator_1"))
    note = st.text_input("Optional note (applied to all saved rows)", value="")
    annotator_safe = _sanitize_annotator_id(annotator_id)

    try:
        subset_rows = _load_random_subset(root)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        return

    subset_items: List[Tuple[Path, Dict[str, Any]]] = []
    for row in subset_rows:
        rel = str(row.get("source_jsonl_path", "")).strip()
        idx = row.get("idx")
        gt = row.get("groundtruth_hazard")
        if not rel or not isinstance(idx, int) or gt is None:
            continue
        subset_items.append((root / rel, row))

    if not subset_items:
        st.error("No valid rows in random_100_samples.json")
        return

    labels_by_source_path: Dict[Path, Dict[int, Dict[str, Any]]] = {}
    labels_paths_by_source_path: Dict[Path, Path] = {}
    source_files = sorted({src for src, _ in subset_items}, key=lambda p: str(p))
    for source_jsonl_path in source_files:
        labels_by_source_path[source_jsonl_path] = {}
        labels_paths_by_source_path[source_jsonl_path] = src_labels_path(root, source_jsonl_path, annotator_safe)

    if db_cfg:
        try:
            labels_by_source_path = load_human_labels_from_db(db_cfg, annotator_safe, source_files, root)
            st.info("Storage backend: Supabase DB")
        except (error.URLError, error.HTTPError, TimeoutError, ValueError) as e:
            st.warning(f"DB load failed; fallback to file storage. ({e})")
            for source_jsonl_path in source_files:
                labels_by_source_path[source_jsonl_path] = load_human_labels(
                    labels_paths_by_source_path[source_jsonl_path]
                )
    else:
        st.info("Storage backend: local files (`human_labels`)")
        for source_jsonl_path in source_files:
            labels_by_source_path[source_jsonl_path] = load_human_labels(
                labels_paths_by_source_path[source_jsonl_path]
            )

    st.info(f"Loaded random subset: {len(subset_items)} rows.")

    all_items_sorted = sorted(subset_items, key=lambda x: (str(x[0]), int(x[1]["idx"])))

    for pos, (source_jsonl_path, item) in enumerate(all_items_sorted):
        saved_for_item = labels_by_source_path.get(source_jsonl_path, {}).get(int(item["idx"]))
        default_choice = bool(saved_for_item.get("human_hazard_correct", False)) if saved_for_item else False
        key = f"random100::{annotator_safe}::{pos}"

        with st.expander(f"Sample {pos + 1}", expanded=False):
            st.caption(f"source: {source_jsonl_path.relative_to(root)} / idx: {item['idx']}")
            st.text_area(
                "groundtruth_hazard",
                value=str(item["groundtruth_hazard"]),
                height=130,
                disabled=True,
                key=f"random100_gt::{annotator_safe}::{pos}",
            )
            response_hazard = item.get("response_hazard")
            if response_hazard is not None:
                st.text_area(
                    "response_hazard",
                    value=str(response_hazard),
                    height=100,
                    disabled=True,
                    key=f"random100_resp::{annotator_safe}::{pos}",
                )
            st.radio(
                "Select (true/false)",
                options=[True, False],
                index=0 if default_choice else 1,
                format_func=lambda b: "true" if b else "false",
                horizontal=True,
                key=key,
            )

    if st.button("Save all (random 100)", type="primary"):
        for pos, (source_jsonl_path, item) in enumerate(all_items_sorted):
            selected = bool(st.session_state.get(f"random100::{annotator_safe}::{pos}", False))
            labels_by_source_path[source_jsonl_path][int(item["idx"])] = {
                "source_jsonl_path": _source_key(root, source_jsonl_path),
                "idx": int(item["idx"]),
                "groundtruth_hazard": str(item["groundtruth_hazard"]),
                "human_hazard_correct": selected,
                "annotator_id": annotator_safe,
                "note": note,
                "saved_at_utc": _now_utc_iso(),
            }

        if db_cfg:
            try:
                rows_to_upsert: List[Dict[str, Any]] = []
                for source_jsonl_path in sorted(labels_by_source_path.keys(), key=lambda p: str(p)):
                    rows_to_upsert.extend(
                        [labels_by_source_path[source_jsonl_path][i] for i in sorted(labels_by_source_path[source_jsonl_path].keys())]
                    )
                write_human_labels_to_db(db_cfg, rows_to_upsert)
            except (error.URLError, error.HTTPError, TimeoutError, ValueError) as e:
                st.error(f"DB save failed: {e}")
                return
        else:
            for source_jsonl_path in sorted(labels_paths_by_source_path.keys(), key=lambda p: str(p)):
                write_human_labels(
                    labels_paths_by_source_path[source_jsonl_path],
                    labels_by_source_path[source_jsonl_path],
                )

        st.success("Saved random 100 labels.")


if __name__ == "__main__":
    main()
