"""Streaming extraction of time-safe spatial weather features from NetCDF."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from .data import TIME_COL


EXPECTED_CHANNELS = ("ghi", "sp", "t2m", "tcc", "tp", "u100", "v100")
FILE_PATTERN = re.compile(r"^(\d{8})\.nc$")
WEATHER_ISSUE_COL = "weather_issue_time_utc"
WEATHER_SOURCE_COL = "weather_source_file"


@dataclass(frozen=True)
class WeatherAudit:
    files_processed: int
    rows_hourly: int
    first_valid_time: str
    last_valid_time: str
    feature_count: int
    schema_errors: int
    negative_tp_differences_clipped: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _decode_channels(values: np.ndarray) -> tuple[str, ...]:
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def _initial_time_from_units(units: str) -> pd.Timestamp:
    marker = "since "
    if marker not in units:
        raise ValueError(f"Unsupported NetCDF time units: {units}")
    return pd.Timestamp(units.split(marker, 1)[1])


def _spatial_features(
    data: np.ndarray,
    channels: tuple[str, ...],
    tile_rows: int,
    tile_cols: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Aggregate a (lead, channel, lat, lon) tensor into hourly features."""

    arrays = {channel: data[:, index].astype(np.float32, copy=False) for index, channel in enumerate(channels)}
    tp_diff, negative_count = cumulative_to_increment(arrays["tp"])
    arrays["tp_increment"] = tp_diff
    arrays["wind_speed_100"] = np.hypot(arrays["u100"], arrays["v100"])

    output: dict[str, np.ndarray] = {}
    lat_groups = np.array_split(np.arange(data.shape[2]), tile_rows)
    lon_groups = np.array_split(np.arange(data.shape[3]), tile_cols)
    for channel, values in arrays.items():
        flat = values.reshape(values.shape[0], -1)
        output[f"wx_{channel}__mean"] = np.nanmean(flat, axis=1)
        output[f"wx_{channel}__std"] = np.nanstd(flat, axis=1)
        output[f"wx_{channel}__p10"] = np.nanpercentile(flat, 10, axis=1)
        output[f"wx_{channel}__p90"] = np.nanpercentile(flat, 90, axis=1)
        for row_index, lat_index in enumerate(lat_groups):
            for col_index, lon_index in enumerate(lon_groups):
                tile = values[:, lat_index][:, :, lon_index]
                output[f"wx_{channel}__tile_{row_index}_{col_index}_mean"] = np.nanmean(tile, axis=(1, 2))
    return output, negative_count


def cumulative_to_increment(
    cumulative: np.ndarray,
    *,
    tolerance: float = 1e-10,
    negative_policy: str = "raise",
) -> tuple[np.ndarray, int]:
    """Convert a per-file cumulative lead sequence into lead increments.

    The first lead is differenced against zero. Negative resets are never
    silently converted with ``abs``; callers must either fail closed or choose
    the explicitly audited ``clip`` policy.
    """

    values = np.asarray(cumulative)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("Cumulative weather input must have a non-empty lead axis")
    increment = np.diff(values, axis=0, prepend=np.zeros_like(values[:1]))
    negative_mask = increment < -abs(tolerance)
    negative_count = int(negative_mask.sum())
    if negative_count:
        if negative_policy == "raise":
            raise ValueError(f"Cumulative field contains {negative_count} negative lead differences")
        if negative_policy != "clip":
            raise ValueError(f"Unknown negative_policy: {negative_policy}")
        increment = np.where(negative_mask, 0, increment)
    increment = np.where(np.abs(increment) <= abs(tolerance), 0, increment)
    return increment, negative_count


def extract_weather_file(
    path: str | Path,
    *,
    tile_rows: int = 3,
    tile_cols: int = 3,
) -> tuple[pd.DataFrame, int]:
    source = Path(path)
    match = FILE_PATTERN.match(source.name)
    if not match:
        raise ValueError(f"Unexpected NetCDF filename: {source.name}")
    file_date = pd.to_datetime(match.group(1), format="%Y%m%d")

    with Dataset(source, mode="r") as dataset:
        required = {"data", "time", "lead_time", "channel", "lat", "lon"}
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ValueError(f"{source.name} is missing NetCDF variables: {missing}")
        channels = _decode_channels(np.asarray(dataset.variables["channel"][:]))
        if channels != EXPECTED_CHANNELS:
            raise ValueError(f"{source.name} channels {channels} do not match {EXPECTED_CHANNELS}")
        leads = np.asarray(dataset.variables["lead_time"][:], dtype=int)
        if not np.array_equal(leads, np.arange(24)):
            raise ValueError(f"{source.name} has invalid lead_time coordinates: {leads.tolist()}")
        raw = np.asarray(dataset.variables["data"][0], dtype=np.float32)
        if raw.shape != (24, 7, 104, 225):
            raise ValueError(f"{source.name} has unexpected data shape {raw.shape}")
        initial_utc = _initial_time_from_units(str(dataset.variables["time"].units))

    if initial_utc.normalize() != file_date or initial_utc.hour != 16:
        raise ValueError(f"{source.name} initial UTC time {initial_utc} does not match release-date contract")
    valid_local = initial_utc + pd.to_timedelta(leads, unit="h") + pd.Timedelta(hours=8)
    if not (valid_local.normalize() == file_date + pd.Timedelta(days=1)).all():
        raise ValueError(f"{source.name} does not map entirely to local D+1")

    features, negative_count = _spatial_features(raw, channels, tile_rows, tile_cols)
    frame = pd.DataFrame(
        {
            TIME_COL: valid_local,
            WEATHER_ISSUE_COL: pd.Timestamp(initial_utc).tz_localize("UTC"),
            WEATHER_SOURCE_COL: source.name,
            **features,
        }
    )
    return frame, negative_count


def extract_weather_directory(
    weather_dir: str | Path,
    *,
    tile_rows: int = 3,
    tile_cols: int = 3,
    files: Iterable[Path] | None = None,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, WeatherAudit]:
    root = Path(weather_dir)
    selected = sorted(files if files is not None else root.glob("*.nc"))
    if not selected:
        raise FileNotFoundError(f"No NetCDF files found under {root}")
    frames: list[pd.DataFrame] = []
    negative_total = 0
    for index, path in enumerate(selected, start=1):
        frame, negative_count = extract_weather_file(path, tile_rows=tile_rows, tile_cols=tile_cols)
        frames.append(frame)
        negative_total += negative_count
        if progress_every and (index % progress_every == 0 or index == len(selected)):
            print(f"weather extraction: {index}/{len(selected)} files", flush=True)
    hourly = pd.concat(frames, ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    if hourly[TIME_COL].duplicated().any():
        raise ValueError("Duplicate valid weather timestamps after concatenation")
    feature_cols = [column for column in hourly if column != TIME_COL]
    audit = WeatherAudit(
        files_processed=len(selected),
        rows_hourly=len(hourly),
        first_valid_time=str(hourly[TIME_COL].iloc[0]),
        last_valid_time=str(hourly[TIME_COL].iloc[-1]),
        feature_count=len(feature_cols),
        schema_errors=0,
        negative_tp_differences_clipped=negative_total,
    )
    return hourly, audit


def expand_hourly_to_quarters(hourly: pd.DataFrame) -> pd.DataFrame:
    """Repeat each released hourly forecast across its four quarter-hours."""

    frames: list[pd.DataFrame] = []
    for minutes in (0, 15, 30, 45):
        part = hourly.copy()
        part[TIME_COL] = part[TIME_COL] + pd.Timedelta(minutes=minutes)
        frames.append(part)
    quarters = pd.concat(frames, ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    if quarters[TIME_COL].duplicated().any():
        raise ValueError("Duplicate timestamps while expanding hourly weather")
    return quarters
