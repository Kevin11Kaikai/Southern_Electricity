"""Causal and forecast-availability-safe feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import FORECAST_COLS, TIME_COL, assert_model_features_are_inference_safe


SLUGS = {
    "系统负荷预测值": "system_load",
    "风光总加预测值": "renewables",
    "联络线预测值": "intertie",
    "风电预测值": "wind",
    "光伏预测值": "solar",
    "水电预测值": "hydro",
    "非市场化机组预测值": "nonmarket",
}


@dataclass(frozen=True)
class FeatureSets:
    baseline: tuple[str, ...]
    contextual: tuple[str, ...]
    temporal: tuple[str, ...]
    full: tuple[str, ...]


def _calendar_features(times: pd.Series) -> pd.DataFrame:
    minutes = times.dt.hour * 60 + times.dt.minute
    day_fraction = minutes / (24 * 60)
    weekday = times.dt.dayofweek
    day_of_year = times.dt.dayofyear
    return pd.DataFrame(
        {
            "hour": times.dt.hour.astype(np.int16),
            "minute": times.dt.minute.astype(np.int16),
            "dayofweek": weekday.astype(np.int16),
            "month": times.dt.month.astype(np.int16),
            "is_weekend": (weekday >= 5).astype(np.int8),
            "tod_sin": np.sin(2 * np.pi * day_fraction),
            "tod_cos": np.cos(2 * np.pi * day_fraction),
            "dow_sin": np.sin(2 * np.pi * weekday / 7),
            "dow_cos": np.cos(2 * np.pi * weekday / 7),
            "year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        },
        index=times.index,
    )


def _interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    load = frame["系统负荷预测值"].astype(float)
    renewables = frame["风光总加预测值"].astype(float)
    wind = frame["风电预测值"].astype(float)
    solar = frame["光伏预测值"].astype(float)
    intertie = frame["联络线预测值"].astype(float)
    return pd.DataFrame(
        {
            "net_load": load - renewables,
            "load_to_renewables": load / (renewables.abs() + eps),
            "renewables_share": renewables / (load.abs() + eps),
            "solar_share": solar / (load.abs() + eps),
            "wind_share": wind / (load.abs() + eps),
            "intertie_share": intertie / (load.abs() + eps),
            "renewables_identity_gap": renewables - wind - solar,
        },
        index=frame.index,
    )


def _day_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    day_key = frame[TIME_COL].dt.normalize()
    output: dict[str, pd.Series] = {}
    for column in FORECAST_COLS:
        slug = SLUGS[column]
        grouped = frame[column].groupby(day_key)
        day_mean = grouped.transform("mean")
        day_std = grouped.transform("std").replace(0, np.nan)
        day_min = grouped.transform("min")
        day_max = grouped.transform("max")
        output[f"{slug}__day_centered"] = frame[column] - day_mean
        output[f"{slug}__day_zscore"] = (frame[column] - day_mean) / day_std
        output[f"{slug}__day_range_position"] = (frame[column] - day_min) / (day_max - day_min + 1e-6)
    return pd.DataFrame(output, index=frame.index)


def _known_day_block_features(frame: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Features for a full D+1 forecast that is available before dispatch.

    These are intentionally *forward-looking in valid time* but not leaked:
    the competition provides all 96 next-day boundary forecasts together.
    Names contain ``known_fwd`` so this availability assumption remains
    visible in manifests and reviews.
    """

    day_key = frame[TIME_COL].dt.normalize()
    output: dict[str, pd.Series] = {}
    for column in FORECAST_COLS:
        slug = SLUGS[column]
        grouped = frame[column].groupby(day_key, sort=False)
        output[f"{slug}__known_fwd8_mean"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=window).mean().shift(-(window - 1))
        )
        output[f"{slug}__known_fwd8_min"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=window).min().shift(-(window - 1))
        )
        output[f"{slug}__known_fwd8_max"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=window).max().shift(-(window - 1))
        )
    return pd.DataFrame(output, index=frame.index)


def _timestamp_safe_temporal_features(
    frame: pd.DataFrame,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    ordered = frame[[TIME_COL, *FORECAST_COLS]].sort_values(TIME_COL).copy()
    if ordered[TIME_COL].duplicated().any():
        raise ValueError("Temporal feature construction requires unique timestamps")
    original_times = pd.DatetimeIndex(ordered[TIME_COL])
    regular_index = pd.date_range(original_times.min(), original_times.max(), freq="15min")
    regular = ordered.set_index(TIME_COL)[FORECAST_COLS].reindex(regular_index)
    output: dict[str, pd.Series] = {}
    for column in FORECAST_COLS:
        slug = SLUGS[column]
        history = regular[column].shift(1)
        for lag in lags:
            output[f"{slug}__lag_{int(lag)}"] = regular[column].shift(int(lag))
        for window in rolling_windows:
            min_periods = max(2, min(int(window), int(window) // 4))
            rolled = history.rolling(int(window), min_periods=min_periods)
            output[f"{slug}__roll_{int(window)}_mean"] = rolled.mean()
            output[f"{slug}__roll_{int(window)}_std"] = rolled.std()
            output[f"{slug}__roll_{int(window)}_min"] = rolled.min()
            output[f"{slug}__roll_{int(window)}_max"] = rolled.max()
    temporal = pd.DataFrame(output, index=regular_index).loc[original_times]
    temporal.index = ordered.index
    return temporal.sort_index()


def build_features(
    forecast_timeline: pd.DataFrame,
    *,
    lags: Iterable[int] = (1, 2, 4, 8, 24, 48, 96, 192, 672),
    rolling_windows: Iterable[int] = (16, 96, 672),
    weather: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FeatureSets]:
    frame = forecast_timeline.copy()
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    assert_model_features_are_inference_safe(FORECAST_COLS)

    calendar = _calendar_features(frame[TIME_COL])
    interactions = _interaction_features(frame)
    day_context = _day_context_features(frame)
    block_context = _known_day_block_features(frame)
    temporal = _timestamp_safe_temporal_features(frame, lags=lags, rolling_windows=rolling_windows)

    feature_frame = pd.concat(
        [frame.reset_index(drop=True), calendar, interactions, day_context, block_context, temporal],
        axis=1,
    )
    baseline = tuple(FORECAST_COLS + ["hour", "minute", "dayofweek", "month"])
    contextual = tuple(
        [
            *baseline,
            *[c for c in calendar.columns if c not in baseline],
            *interactions.columns,
            *day_context.columns,
            *block_context.columns,
        ]
    )
    temporal_columns = tuple(temporal.columns)
    temporal_set = tuple([*contextual, *temporal_columns])

    weather_columns: list[str] = []
    if weather is not None:
        weather_frame = weather.copy()
        weather_frame[TIME_COL] = pd.to_datetime(weather_frame[TIME_COL], errors="raise")
        weather_columns = [
            column
            for column in weather_frame.columns
            if column != TIME_COL and pd.api.types.is_numeric_dtype(weather_frame[column])
        ]
        weather_frame = weather_frame[[TIME_COL, *weather_columns]]
        feature_frame = feature_frame.merge(weather_frame, on=TIME_COL, how="left", validate="one_to_one")
        feature_frame["weather_missing"] = feature_frame[weather_columns].isna().all(axis=1).astype(np.int8)
        weather_columns.append("weather_missing")

    all_model_features = [*temporal_set, *weather_columns]
    assert_model_features_are_inference_safe(all_model_features)
    for column in all_model_features:
        if pd.api.types.is_float_dtype(feature_frame[column]):
            feature_frame[column] = feature_frame[column].astype(np.float32)
    sets = FeatureSets(
        baseline=baseline,
        contextual=contextual,
        temporal=temporal_set,
        full=tuple(all_model_features),
    )
    return feature_frame, sets


def feature_causality_probe(
    frame: pd.DataFrame,
    *,
    lags: Iterable[int] = (1, 96),
    rolling_windows: Iterable[int] = (16,),
) -> bool:
    """Verify that appending future rows cannot change past temporal features."""

    cutoff = max(10, len(frame) // 2)
    past = frame.iloc[:cutoff].copy()
    past_features = _timestamp_safe_temporal_features(past, lags, rolling_windows)
    full_features = _timestamp_safe_temporal_features(frame, lags, rolling_windows).iloc[:cutoff]
    return past_features.equals(full_features)
