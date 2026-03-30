import json
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_aligned_jsonl_files(root: Path) -> List[Path]:
    aligned_root = root / "Sampling_aligned_triplets_v2"
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


def load_random_subset_json(root: Path) -> List[Dict[str, Any]]:
    subset_path = root / "streamlit_hazard_correct_labeler" / "data" / "random_100_samples_v2.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    with subset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Subset JSON should be a list.")
    return [row for row in data if isinstance(row, dict)]


def ensure_has_keys(row: Dict[str, Any], keys: List[str]) -> bool:
    for k in keys:
        if k not in row:
            return False
    return True


def _sanitize_annotator_id(annotator_id: str) -> str:
    # Keep filesystem-safe, stable identifiers.
    cleaned = annotator_id.strip().replace(os.sep, "_").replace(":", "_") or "anonymous"
    # Keep directory name within safe limits (Windows component length).
    return cleaned[:80]


def src_labels_path(root: Path, source_jsonl_path: Path, annotator_id: str) -> Path:
    # Keep human labels per source file and per annotator (to avoid collisions).
    rel = source_jsonl_path.relative_to(root)
    annotator_safe = _sanitize_annotator_id(annotator_id)

    # Windows path length limits can be hit if we embed the entire `rel` into the filename.
    # Use a stable short hash so the filename stays within safe limits.
    rel_str = str(rel)
    rel_hash = hashlib.sha1(rel_str.encode("utf-8")).hexdigest()[:12]

    # Backward-compat: if the old naming scheme already exists on disk,
    # prefer it so we don't lose previously saved labels.
    try:
        legacy_safe = rel_str.replace(os.sep, "__").replace(":", "")
        legacy_path = (
            root
            / "streamlit_hazard_correct_labeler"
            / "human_labels"
            / annotator_safe
            / f"{legacy_safe}__labels.jsonl"
        )
        if legacy_path.exists():
            return legacy_path
    except OSError:
        # Path might be too long for Windows; fall back to hashed filename.
        pass

    # Keep a small readable prefix (last component), but truncate to be safe.
    name_prefix = rel.name
    # Remove characters that can be invalid on Windows filenames.
    name_prefix = (
        name_prefix.replace(":", "")
        .replace("\\", "_")
        .replace("/", "_")
        .replace("?", "")
        .replace("*", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
    )
    name_prefix = name_prefix[:60] if len(name_prefix) > 60 else name_prefix

    filename = f"{name_prefix}__{rel_hash}__labels.jsonl"
    return (
        root
        / "streamlit_hazard_correct_labeler"
        / "human_labels"
        / annotator_safe
        / filename
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


def _source_key(root: Path, source_jsonl_path: Path) -> str:
    return str(source_jsonl_path.relative_to(root))


def _get_secret_or_env(key: str) -> Optional[str]:
    try:
        value = st.secrets.get(key, None)
    except StreamlitSecretNotFoundError:
        value = None
    if value:
        return str(value)
    env_value = os.environ.get(key)
    return str(env_value) if env_value else None


def _supabase_config() -> Optional[Dict[str, str]]:
    url = _get_secret_or_env("SUPABASE_URL")
    key = _get_secret_or_env("SUPABASE_KEY")
    if not url or not key:
        return None
    table = _get_secret_or_env("SUPABASE_TABLE") or "human_labels"
    return {"url": url.rstrip("/"), "key": key, "table": table}


def _supabase_request_json(
    method: str,
    cfg: Dict[str, str],
    query_params: Optional[Dict[str, str]] = None,
    payload: Optional[List[Dict[str, Any]]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Any:
    base = f"{cfg['url']}/rest/v1/{cfg['table']}"
    if query_params:
        base = f"{base}?{parse.urlencode(query_params)}"
    headers = {
        "apikey": cfg["key"],
        "Authorization": f"Bearer {cfg['key']}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(base, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))


def load_human_labels_from_db(
    cfg: Dict[str, str], annotator_id: str, files: List[Path], root: Path
) -> Dict[Path, Dict[int, Dict[str, Any]]]:
    by_source: Dict[Path, Dict[int, Dict[str, Any]]] = {p: {} for p in files}
    source_by_key: Dict[str, Path] = {_source_key(root, p): p for p in files}
    rows = _supabase_request_json(
        method="GET",
        cfg=cfg,
        query_params={
            "select": "source_jsonl_path,idx,groundtruth_hazard,human_hazard_correct,annotator_id,note,saved_at_utc",
            "annotator_id": f"eq.{annotator_id}",
            "limit": "50000",
        },
        extra_headers={"Accept": "application/json"},
    )
    if not isinstance(rows, list):
        return by_source
    for obj in rows:
        source_key = str(obj.get("source_jsonl_path", ""))
        source_path = source_by_key.get(source_key)
        idx = obj.get("idx")
        if source_path is None or not isinstance(idx, int):
            continue
        by_source[source_path][idx] = obj
    return by_source


def write_human_labels_to_db(cfg: Dict[str, str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    _supabase_request_json(
        method="POST",
        cfg=cfg,
        query_params={"on_conflict": "source_jsonl_path,idx,annotator_id"},
        payload=rows,
        extra_headers={
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
    )


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

    # Workspace root:
    # - Prefer `WORKSPACE_ROOT` env var if provided
    # - Otherwise infer from this file location (…/web/streamlit_hazard_correct_labeler/app.py -> …/web)
    inferred_default_root = Path(__file__).resolve().parents[1]
    default_root = Path(os.environ.get("WORKSPACE_ROOT", str(inferred_default_root))).resolve()
    try:
        workspace_root = st.secrets.get("WORKSPACE_ROOT", None)
    except StreamlitSecretNotFoundError:
        workspace_root = None
    root = Path(workspace_root).resolve() if workspace_root else default_root
    db_cfg = _supabase_config()

    st.title("Hazard Correct Human Labeler")
    st.caption("`Sampling_aligned_triplets`의 모든 샘플을 한 화면에서 보고 true/false를 선택합니다.")
    st.caption("`SUPABASE_URL`/`SUPABASE_KEY`가 설정되면 DB에 영구 저장됩니다.")

    annotator_id = st.text_input("annotator_id", value=os.environ.get("ANNOTATOR_ID", "annotator_1"))
    note = st.text_input("Optional note (applied to all saved rows)", value="")
    use_random_subset = st.checkbox(
        "Use random 100 subset (`data/random_100_samples.json`)",
        value=False,
        help="켜면 전체 대신 고정 랜덤 100개만 라벨링합니다.",
    )
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

    # Load rows once (all rows or fixed random 100 subset).
    all_items: List[Tuple[Path, Item]] = []  # (source_jsonl_path, item)
    invalid_counts = 0
    if use_random_subset:
        try:
            subset_rows = load_random_subset_json(root)
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            return

        for row in subset_rows:
            rel = str(row.get("source_jsonl_path", "")).strip()
            if not rel:
                invalid_counts += 1
                continue
            source_jsonl_path = (root / rel).resolve()
            if source_jsonl_path not in files:
                invalid_counts += 1
                continue

            subset_item = {
                "idx": row.get("idx"),
                "groundtruth_hazard": row.get("groundtruth_hazard"),
                "hazard_correct": False,
                "response_hazard": row.get("response_hazard"),
            }
            if not ensure_has_keys(subset_item, required):
                invalid_counts += 1
                continue
            try:
                all_items.append((source_jsonl_path, to_item(subset_item)))
            except (TypeError, ValueError):
                invalid_counts += 1
    else:
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

    mode_text = "random subset (100)" if use_random_subset else "full dataset"
    st.info(f"Loaded {len(all_items)} rows ({mode_text}). Invalid/skipped rows: {invalid_counts}.")

    # Load existing human labels per source file (per annotator).
    annotator_safe = _sanitize_annotator_id(annotator_id)
    labels_by_source_path: Dict[Path, Dict[int, Dict[str, Any]]] = {}
    labels_paths_by_source_path: Dict[Path, Path] = {}
    for source_jsonl_path in sorted(files):
        labels_by_source_path[source_jsonl_path] = {}
        labels_paths_by_source_path[source_jsonl_path] = src_labels_path(root, source_jsonl_path, annotator_safe)

    if db_cfg:
        try:
            labels_by_source_path = load_human_labels_from_db(db_cfg, annotator_safe, sorted(files), root)
            st.info("Storage backend: Supabase DB")
        except (error.URLError, error.HTTPError, TimeoutError, ValueError) as e:
            st.warning(f"DB load failed; fallback to file storage. ({e})")
            for source_jsonl_path in sorted(files):
                labels_by_source_path[source_jsonl_path] = load_human_labels(
                    labels_paths_by_source_path[source_jsonl_path]
                )
    else:
        st.info("Storage backend: local files (`human_labels`) ")
        for source_jsonl_path in sorted(files):
            labels_by_source_path[source_jsonl_path] = load_human_labels(
                labels_paths_by_source_path[source_jsonl_path]
            )

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
                "source_jsonl_path": _source_key(root, source_jsonl_path),
                "idx": item.idx,
                "groundtruth_hazard": item.groundtruth_hazard,
                "human_hazard_correct": selected,
                "annotator_id": annotator_safe,
                "note": note,
                "saved_at_utc": _now_utc_iso(),
            }

        if db_cfg:
            try:
                rows_to_upsert: List[Dict[str, Any]] = []
                for source_jsonl_path in sorted(labels_by_source_path.keys()):
                    rows_to_upsert.extend(
                        [labels_by_source_path[source_jsonl_path][i] for i in sorted(labels_by_source_path[source_jsonl_path].keys())]
                    )
                write_human_labels_to_db(db_cfg, rows_to_upsert)
            except (error.URLError, error.HTTPError, TimeoutError, ValueError) as e:
                st.error(f"DB save failed: {e}")
                return
        else:
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

