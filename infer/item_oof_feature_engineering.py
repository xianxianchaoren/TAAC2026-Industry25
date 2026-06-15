import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq


ITEM_OOF_FEATURE_NAMES: List[str] = [
    "item_oof_show_cnt_1h_log1p",
    "item_oof_show_cnt_3h_log1p",
    "item_oof_show_cnt_6h_log1p",
    "item_oof_show_cnt_24h_log1p",
    "item_oof_pos_cnt_1h_log1p",
    "item_oof_pos_cnt_3h_log1p",
    "item_oof_pos_cnt_6h_log1p",
    "item_oof_pos_cnt_24h_log1p",
    "item_oof_smoothed_cvr_1h",
    "item_oof_smoothed_cvr_6h",
    "item_oof_smoothed_cvr_24h",
    "item_oof_show_trend_1h_6h",
    "item_oof_show_trend_3h_24h",
    "item_oof_cvr_trend_1h_6h",
    "item_oof_hours_since_last_show_log1p",
    "item_oof_hours_since_last_pos_log1p",
]

ITEM_OOF_FEATURE_FIDS: List[int] = [
    11000 + i for i in range(1, len(ITEM_OOF_FEATURE_NAMES) + 1)
]

_HOUR_SECONDS = 3600
_DEFAULT_WINDOWS: Tuple[int, ...] = (1, 3, 6, 24)


def _safe_log1p(x: float) -> float:
    return float(math.log1p(max(x, 0.0)))


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _smooth_rate(pos_cnt: float, show_cnt: float, prior: float, alpha: float) -> float:
    return float((pos_cnt + alpha * prior) / (show_cnt + alpha))


def row_group_key(file_path: str, row_group_idx: int) -> str:
    return f"{os.path.abspath(file_path)}::{int(row_group_idx)}"


def list_row_groups(parquet_path: str) -> List[Tuple[str, int, int]]:
    if os.path.isdir(parquet_path):
        import glob

        parquet_files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found under {parquet_path}")
    else:
        parquet_files = [parquet_path]

    row_groups: List[Tuple[str, int, int]] = []
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            row_groups.append((file_path, rg_idx, pf.metadata.row_group(rg_idx).num_rows))
    return row_groups


def split_train_valid_row_groups(
    row_groups: Sequence[Tuple[str, int, int]],
    valid_ratio: float,
    train_ratio: float,
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    total_rgs = len(row_groups)
    if total_rgs == 0:
        return [], []

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
    return list(row_groups[:n_train_rgs]), list(row_groups[n_train_rgs:])


def make_contiguous_folds(
    row_groups: Sequence[Tuple[str, int, int]],
    num_folds: int,
) -> Dict[int, List[Tuple[str, int, int]]]:
    if not row_groups:
        return {}
    num_folds = max(2, min(int(num_folds), len(row_groups)))
    boundaries = np.linspace(0, len(row_groups), num_folds + 1, dtype=int)
    folds: Dict[int, List[Tuple[str, int, int]]] = {}
    for fold_id in range(num_folds):
        start = int(boundaries[fold_id])
        end = int(boundaries[fold_id + 1])
        if start < end:
            folds[fold_id] = list(row_groups[start:end])
    return folds


@dataclass
class ItemOOFFeatureTable:
    feature_names: Sequence[str]
    feature_fids: Sequence[int]
    vectors_by_item: Dict[int, np.ndarray]


def save_item_oof_feature_table(table: ItemOOFFeatureTable, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    item_ids = np.array(sorted(table.vectors_by_item.keys()), dtype=np.int64)
    if len(item_ids) == 0:
        vectors = np.zeros((0, len(table.feature_names)), dtype=np.float32)
    else:
        vectors = np.stack([table.vectors_by_item[int(item_id)] for item_id in item_ids]).astype(np.float32)

    np.savez_compressed(
        file_path,
        item_ids=item_ids,
        vectors=vectors,
        feature_fids=np.asarray(table.feature_fids, dtype=np.int64),
    )
    meta_path = os.path.splitext(file_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": list(table.feature_names),
                "feature_fids": list(map(int, table.feature_fids)),
            },
            f,
            indent=2,
        )


def load_item_oof_feature_table(file_path: str) -> ItemOOFFeatureTable:
    npz = np.load(file_path, allow_pickle=False)
    item_ids = npz["item_ids"].astype(np.int64)
    vectors = npz["vectors"].astype(np.float32)
    feature_fids = npz["feature_fids"].astype(np.int64).tolist()

    meta_path = os.path.splitext(file_path)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feature_names = meta["feature_names"]
    else:
        feature_names = ITEM_OOF_FEATURE_NAMES[: vectors.shape[1]]

    vectors_by_item = {
        int(item_id): vectors[i]
        for i, item_id in enumerate(item_ids.tolist())
    }
    return ItemOOFFeatureTable(
        feature_names=feature_names,
        feature_fids=feature_fids,
        vectors_by_item=vectors_by_item,
    )


@dataclass
class _ItemTemporalStats:
    last_show_ts: Optional[int] = None
    last_pos_ts: Optional[int] = None


def _group_row_groups(
    row_groups: Sequence[Tuple[str, int, int]]
) -> Mapping[str, List[int]]:
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for file_path, row_group_idx, _ in row_groups:
        grouped[file_path].append(int(row_group_idx))
    return grouped


class HourlyItemOOFDenseFeatureBuilder:
    def __init__(
        self,
        row_groups: Sequence[Tuple[str, int, int]],
        hour_windows: Sequence[int] = _DEFAULT_WINDOWS,
        batch_size: int = 65536,
        positive_label_value: int = 2,
        smoothing_alpha: float = 100.0,
    ) -> None:
        self._row_groups = list(row_groups)
        self._hour_windows = tuple(sorted(set(int(h) for h in hour_windows if int(h) > 0)))
        self._batch_size = int(batch_size)
        self._positive_label_value = int(positive_label_value)
        self._smoothing_alpha = float(smoothing_alpha)
        if self._hour_windows != _DEFAULT_WINDOWS:
            raise ValueError(
                f"Current ITEM_OOF feature layout assumes hour_windows={_DEFAULT_WINDOWS}, "
                f"got {self._hour_windows}"
            )

    def build(self) -> ItemOOFFeatureTable:
        show_hour_counts: MutableMapping[int, Counter] = defaultdict(Counter)
        pos_hour_counts: MutableMapping[int, Counter] = defaultdict(Counter)
        item_stats: MutableMapping[int, _ItemTemporalStats] = {}
        global_show_cnt = 0
        global_pos_cnt = 0
        max_ts: Optional[int] = None

        grouped_row_groups = _group_row_groups(self._row_groups)
        for file_path, rg_indices in grouped_row_groups.items():
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(
                columns=["item_id", "timestamp", "label_type"],
                batch_size=self._batch_size,
                row_groups=rg_indices,
            ):
                item_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
                timestamps = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
                label_types = batch.column(2).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                positives = (label_types == self._positive_label_value).astype(np.int64)

                if len(item_ids) == 0:
                    continue

                global_show_cnt += int(len(item_ids))
                global_pos_cnt += int(positives.sum())
                batch_max_ts = int(timestamps.max())
                max_ts = batch_max_ts if max_ts is None else max(max_ts, batch_max_ts)

                hour_buckets = timestamps // _HOUR_SECONDS

                show_pairs = np.column_stack((item_ids, hour_buckets))
                uniq_show_pairs, show_pair_counts = np.unique(show_pairs, axis=0, return_counts=True)
                for pair, count in zip(uniq_show_pairs, show_pair_counts):
                    item_id = int(pair[0])
                    hour_bucket = int(pair[1])
                    show_hour_counts[item_id][hour_bucket] += int(count)

                pos_item_ids = item_ids[positives > 0]
                pos_hour_buckets = hour_buckets[positives > 0]
                if len(pos_item_ids) > 0:
                    pos_pairs = np.column_stack((pos_item_ids, pos_hour_buckets))
                    uniq_pos_pairs, pos_pair_counts = np.unique(pos_pairs, axis=0, return_counts=True)
                    for pair, count in zip(uniq_pos_pairs, pos_pair_counts):
                        item_id = int(pair[0])
                        hour_bucket = int(pair[1])
                        pos_hour_counts[item_id][hour_bucket] += int(count)

                sort_idx = np.argsort(item_ids, kind="stable")
                s_item_ids = item_ids[sort_idx]
                s_timestamps = timestamps[sort_idx]
                s_positives = positives[sort_idx]
                uniq_items, starts = np.unique(s_item_ids, return_index=True)
                ends = np.append(starts[1:], len(s_item_ids))
                max_item_ts = np.maximum.reduceat(s_timestamps, starts)
                for item_id, item_last_ts, start, end in zip(uniq_items, max_item_ts, starts, ends):
                    item_id_i = int(item_id)
                    stat = item_stats.get(item_id_i)
                    if stat is None:
                        stat = _ItemTemporalStats()
                        item_stats[item_id_i] = stat
                    cur_last_show = int(item_last_ts)
                    stat.last_show_ts = cur_last_show if stat.last_show_ts is None else max(stat.last_show_ts, cur_last_show)
                    pos_slice = s_positives[start:end]
                    if pos_slice.any():
                        pos_last_ts = int(s_timestamps[start:end][pos_slice > 0].max())
                        stat.last_pos_ts = pos_last_ts if stat.last_pos_ts is None else max(stat.last_pos_ts, pos_last_ts)

        if max_ts is None:
            return ItemOOFFeatureTable(
                feature_names=ITEM_OOF_FEATURE_NAMES,
                feature_fids=ITEM_OOF_FEATURE_FIDS,
                vectors_by_item={},
            )

        global_prior = _safe_ratio(global_pos_cnt, global_show_cnt)
        max_hour_bucket = max_ts // _HOUR_SECONDS
        all_item_ids = set(show_hour_counts.keys()) | set(pos_hour_counts.keys()) | set(item_stats.keys())
        vectors_by_item: Dict[int, np.ndarray] = {}
        for item_id in all_item_ids:
            show_counter = show_hour_counts.get(item_id, Counter())
            pos_counter = pos_hour_counts.get(item_id, Counter())
            stat = item_stats.get(item_id, _ItemTemporalStats())

            show_by_window: Dict[int, float] = {}
            pos_by_window: Dict[int, float] = {}
            cvr_by_window: Dict[int, float] = {}
            for window_hours in self._hour_windows:
                min_hour = max_hour_bucket - (window_hours - 1)
                show_cnt = float(
                    sum(count for hour_bucket, count in show_counter.items() if hour_bucket >= min_hour)
                )
                pos_cnt = float(
                    sum(count for hour_bucket, count in pos_counter.items() if hour_bucket >= min_hour)
                )
                show_by_window[window_hours] = show_cnt
                pos_by_window[window_hours] = pos_cnt
                cvr_by_window[window_hours] = _smooth_rate(
                    pos_cnt,
                    show_cnt,
                    prior=global_prior,
                    alpha=self._smoothing_alpha,
                )

            hours_since_last_show = (
                (max_ts - int(stat.last_show_ts)) / float(_HOUR_SECONDS)
                if stat.last_show_ts is not None else 0.0
            )
            hours_since_last_pos = (
                (max_ts - int(stat.last_pos_ts)) / float(_HOUR_SECONDS)
                if stat.last_pos_ts is not None else 0.0
            )

            vector = np.array(
                [
                    _safe_log1p(show_by_window[1]),
                    _safe_log1p(show_by_window[3]),
                    _safe_log1p(show_by_window[6]),
                    _safe_log1p(show_by_window[24]),
                    _safe_log1p(pos_by_window[1]),
                    _safe_log1p(pos_by_window[3]),
                    _safe_log1p(pos_by_window[6]),
                    _safe_log1p(pos_by_window[24]),
                    cvr_by_window[1],
                    cvr_by_window[6],
                    cvr_by_window[24],
                    _safe_ratio(show_by_window[1], show_by_window[6]),
                    _safe_ratio(show_by_window[3], show_by_window[24]),
                    _safe_ratio(cvr_by_window[1], cvr_by_window[6]),
                    _safe_log1p(hours_since_last_show),
                    _safe_log1p(hours_since_last_pos),
                ],
                dtype=np.float32,
            )
            vectors_by_item[int(item_id)] = vector

        return ItemOOFFeatureTable(
            feature_names=ITEM_OOF_FEATURE_NAMES,
            feature_fids=ITEM_OOF_FEATURE_FIDS,
            vectors_by_item=vectors_by_item,
        )


def build_item_oof_artifacts(
    train_row_groups: Sequence[Tuple[str, int, int]],
    artifact_dir: str,
    num_folds: int,
    hour_windows: Sequence[int] = _DEFAULT_WINDOWS,
    batch_size: int = 65536,
    positive_label_value: int = 2,
    smoothing_alpha: float = 100.0,
) -> Tuple[str, Dict[int, str], Dict[str, int]]:
    os.makedirs(artifact_dir, exist_ok=True)
    full_table = HourlyItemOOFDenseFeatureBuilder(
        row_groups=train_row_groups,
        hour_windows=hour_windows,
        batch_size=batch_size,
        positive_label_value=positive_label_value,
        smoothing_alpha=smoothing_alpha,
    ).build()
    full_path = os.path.join(artifact_dir, "item_oof_full_table.npz")
    save_item_oof_feature_table(full_table, full_path)

    folds = make_contiguous_folds(train_row_groups, num_folds=num_folds)
    fold_paths: Dict[int, str] = {}
    fold_by_rg: Dict[str, int] = {}
    for fold_id, fold_rows in folds.items():
        fit_rows = [rg for rg in train_row_groups if rg not in fold_rows]
        if not fit_rows:
            fit_rows = list(train_row_groups)
        table = HourlyItemOOFDenseFeatureBuilder(
            row_groups=fit_rows,
            hour_windows=hour_windows,
            batch_size=batch_size,
            positive_label_value=positive_label_value,
            smoothing_alpha=smoothing_alpha,
        ).build()
        fold_path = os.path.join(artifact_dir, f"item_oof_fold{fold_id}.npz")
        save_item_oof_feature_table(table, fold_path)
        fold_paths[fold_id] = fold_path
        for file_path, rg_idx, _ in fold_rows:
            fold_by_rg[row_group_key(file_path, rg_idx)] = fold_id

    return full_path, fold_paths, fold_by_rg
