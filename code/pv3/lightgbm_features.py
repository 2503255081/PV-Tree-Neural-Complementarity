"""Pure-NumPy construction of the study's direct LightGBM input features.

Callers supply issue-time power/weather arrays and explicit validity masks.
Missing numeric inputs remain NaN for LightGBM; coverage indicators are retained."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


CONTEXT_LENGTH = 96
PREDICTION_LENGTH = 16
RECENT_5MIN_LENGTH = 24

DEFAULT_TIME_FEATURE_NAMES = (
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "solar_elevation_sin",
    "solar_azimuth_sin",
    "solar_azimuth_cos",
    "clear_sky_ghi_haurwitz_per_1000",
    "is_daylight",
    "sunrise_transition",
    "sunset_transition",
)

DEFAULT_NWP_FEATURE_NAMES = (
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "dew_point_2m_c",
    "cloud_cover_pct",
    "cloud_cover_low_pct",
    "cloud_cover_mid_pct",
    "cloud_cover_high_pct",
    "shortwave_radiation_instant_w_m2",
    "direct_radiation_instant_w_m2",
    "diffuse_radiation_instant_w_m2",
    "direct_normal_irradiance_instant_w_m2",
    "precipitation_hourly_sum_mm",
    "shortwave_radiation_hourly_mean_w_m2",
    "wind_u_10m_ms",
    "wind_v_10m_ms",
)

DEFAULT_NWP_METADATA_FEATURE_NAMES = (
    "run_age_days",
    "feature_lead_days",
    "run_hour_sin",
    "run_hour_cos",
    "run_time_valid",
    "older_run_fallback",
    "shortwave_clear_sky_ratio",
    "shortwave_clear_sky_ratio_valid",
    "shortwave_delta_per_1000",
    "cloud_cover_delta_per_100",
    "nwp_delta_valid",
)

_PAST_POINT_MINUTES = (15, 30, 60, 120, 240)
_PAST_ROLLING_MINUTES = (60, 240, 1440)
_RECENT_POINT_MINUTES = (5, 15, 30, 60, 120)
_RECENT_DELTA_MINUTES = (5, 15, 30, 60)
_RECENT_ROLLING_MINUTES = (15, 30, 60, 120)
_ROLLING_STAT_NAMES = (
    "mean",
    "std",
    "minimum",
    "maximum",
    "range",
    "coverage",
    "observed_fraction",
)


@dataclass(frozen=True)
class LightGBMFeatureMatrices:
    """Shared and per-horizon feature blocks for one batch.

    ``shared`` has shape ``[B, F_shared]`` and ``horizon`` has shape
    ``[B, 16, F_horizon]``.  A direct model consumes the concatenation returned
    by :meth:`for_horizon`.
    """

    shared: np.ndarray
    horizon: np.ndarray
    shared_feature_names: tuple[str, ...]
    horizon_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.shared.ndim != 2:
            raise ValueError("shared features must have shape [B, F_shared]")
        if self.horizon.ndim != 3:
            raise ValueError("horizon features must have shape [B, H, F_horizon]")
        if self.shared.shape[0] != self.horizon.shape[0]:
            raise ValueError("shared and horizon feature batch sizes differ")
        if self.horizon.shape[1] != PREDICTION_LENGTH:
            raise ValueError(
                f"horizon features must contain {PREDICTION_LENGTH} horizons"
            )
        if self.shared.shape[1] != len(self.shared_feature_names):
            raise ValueError("shared feature names do not match the matrix")
        if self.horizon.shape[2] != len(self.horizon_feature_names):
            raise ValueError("horizon feature names do not match the matrix")
        if np.isinf(self.shared).any() or np.isinf(self.horizon).any():
            raise ValueError("LightGBM features may contain NaN but not infinity")

    @property
    def batch_size(self) -> int:
        return int(self.shared.shape[0])

    @property
    def combined_feature_names(self) -> tuple[str, ...]:
        return self.shared_feature_names + self.horizon_feature_names

    def for_horizon(self, horizon: int) -> np.ndarray:
        """Return a contiguous ``float32 [B, F_total]`` direct-model matrix."""

        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
            raise TypeError("horizon must be an integer")
        horizon = int(horizon)
        if not 0 <= horizon < PREDICTION_LENGTH:
            raise IndexError(f"horizon must be in [0, {PREDICTION_LENGTH})")
        return np.ascontiguousarray(
            np.concatenate((self.shared, self.horizon[:, horizon, :]), axis=1),
            dtype=np.float32,
        )


def shared_feature_names(
    *,
    time_feature_names: Sequence[str] = DEFAULT_TIME_FEATURE_NAMES,
    nwp_feature_names: Sequence[str] = DEFAULT_NWP_FEATURE_NAMES,
    nwp_metadata_feature_names: Sequence[str] = DEFAULT_NWP_METADATA_FEATURE_NAMES,
) -> tuple[str, ...]:
    """Return the ordered shared-feature contract."""

    time_names = _validated_names(time_feature_names, "time_feature_names")
    nwp_names = _validated_names(nwp_feature_names, "nwp_feature_names")
    metadata_names = _validated_names(
        nwp_metadata_feature_names, "nwp_metadata_feature_names"
    )
    names = [
        "station_index",
        "station_latitude_degrees",
        "station_longitude_degrees",
        "station_capacity_mw",
        "station_log1p_capacity_mw",
    ]
    names.extend(f"latest_past_{name}" for name in time_names)
    for minutes in _PAST_POINT_MINUTES:
        prefix = f"past_power_t_minus_{minutes}m"
        names.extend((prefix, f"{prefix}_valid", f"{prefix}_observed"))
    for minutes in _PAST_ROLLING_MINUTES:
        names.extend(
            f"past_power_last_{minutes}m_{stat}" for stat in _ROLLING_STAT_NAMES
        )
    for minutes in _RECENT_POINT_MINUTES:
        prefix = f"recent_power_t_minus_{minutes}m"
        names.extend((prefix, f"{prefix}_valid", f"{prefix}_observed"))
    for minutes in _RECENT_DELTA_MINUTES:
        prefix = f"recent_power_delta_{minutes}m"
        names.extend((prefix, f"{prefix}_valid"))
    for minutes in _RECENT_ROLLING_MINUTES:
        names.extend(
            f"recent_power_last_{minutes}m_{stat}" for stat in _ROLLING_STAT_NAMES
        )
    for name in nwp_names:
        names.extend((f"latest_past_nwp_{name}", f"latest_past_nwp_{name}_valid"))
    names.extend(f"latest_past_nwp_meta_{name}" for name in metadata_names)
    return tuple(names)


def horizon_feature_names(
    *,
    time_feature_names: Sequence[str] = DEFAULT_TIME_FEATURE_NAMES,
    nwp_feature_names: Sequence[str] = DEFAULT_NWP_FEATURE_NAMES,
    nwp_metadata_feature_names: Sequence[str] = DEFAULT_NWP_METADATA_FEATURE_NAMES,
) -> tuple[str, ...]:
    """Return the ordered feature contract used by every direct horizon."""

    time_names = _validated_names(time_feature_names, "time_feature_names")
    nwp_names = _validated_names(nwp_feature_names, "nwp_feature_names")
    metadata_names = _validated_names(
        nwp_metadata_feature_names, "nwp_metadata_feature_names"
    )
    names = [
        "power_same_time_24h",
        "power_same_time_24h_valid",
        "power_same_time_24h_observed",
        "power_same_time_48h",
        "power_same_time_48h_valid",
    ]
    names.extend(f"future_{name}" for name in time_names)
    for name in nwp_names:
        names.extend((f"future_nwp_{name}", f"future_nwp_{name}_valid"))
    for name in nwp_names:
        names.extend(
            (f"future_minus_latest_nwp_{name}", f"future_minus_latest_nwp_{name}_valid")
        )
    names.extend(f"future_nwp_meta_{name}" for name in metadata_names)
    return tuple(names)


def build_lightgbm_features(
    batch: Mapping,
    *,
    causal_48h_power,
    causal_48h_mask=None,
    time_feature_names: Sequence[str] = DEFAULT_TIME_FEATURE_NAMES,
    nwp_feature_names: Sequence[str] = DEFAULT_NWP_FEATURE_NAMES,
    nwp_metadata_feature_names: Sequence[str] = DEFAULT_NWP_METADATA_FEATURE_NAMES,
) -> LightGBMFeatureMatrices:
    """Build leakage-safe V14 feature blocks from a causal Dataset batch.

    ``causal_48h_power`` must contain one capacity-normalized value per sample
    and horizon, with shape ``[B, 16]`` (a trailing singleton is accepted).  It
    is deliberately external because the standard 96-step Dataset context only
    reaches 24 hours.  The caller is responsible for resolving those older
    values using the same forecast-origin availability rules as the Dataset.
    """

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    time_names = _validated_names(time_feature_names, "time_feature_names")
    nwp_names = _validated_names(nwp_feature_names, "nwp_feature_names")
    metadata_names = _validated_names(
        nwp_metadata_feature_names, "nwp_metadata_feature_names"
    )

    past_power = _scalar_sequence(batch, "past_power", CONTEXT_LENGTH)
    batch_size = past_power.shape[0]
    past_complete = _bool_scalar_sequence(
        batch, "past_complete_mask", batch_size, CONTEXT_LENGTH
    )
    past_source = _bool_scalar_sequence(
        batch, "past_source_span_mask", batch_size, CONTEXT_LENGTH
    )
    past_observed = _bool_scalar_sequence(
        batch, "past_observed_mask", batch_size, CONTEXT_LENGTH
    )
    past_valid = past_complete & past_source & np.isfinite(past_power)

    recent_power = _scalar_sequence(
        batch, "recent_5min_power", RECENT_5MIN_LENGTH, batch_size=batch_size
    )
    recent_complete = _bool_scalar_sequence(
        batch, "recent_5min_complete_mask", batch_size, RECENT_5MIN_LENGTH
    )
    recent_source = _bool_scalar_sequence(
        batch, "recent_5min_source_span_mask", batch_size, RECENT_5MIN_LENGTH
    )
    recent_observed = _bool_scalar_sequence(
        batch, "recent_5min_observed_mask", batch_size, RECENT_5MIN_LENGTH
    )
    recent_valid = recent_complete & recent_source & np.isfinite(recent_power)

    past_features = _feature_sequence(
        batch,
        "past_features",
        batch_size,
        CONTEXT_LENGTH,
        len(time_names),
        require_finite=True,
    )
    future_features = _feature_sequence(
        batch,
        "future_features",
        batch_size,
        PREDICTION_LENGTH,
        len(time_names),
        require_finite=True,
    )
    past_nwp = _feature_sequence(
        batch,
        "past_nwp",
        batch_size,
        CONTEXT_LENGTH,
        len(nwp_names),
    )
    past_nwp_mask = _bool_feature_sequence(
        batch, "past_nwp_mask", past_nwp.shape
    )
    future_nwp = _feature_sequence(
        batch,
        "future_nwp",
        batch_size,
        PREDICTION_LENGTH,
        len(nwp_names),
    )
    future_nwp_mask = _bool_feature_sequence(
        batch, "future_nwp_mask", future_nwp.shape
    )
    past_nwp_metadata = _feature_sequence(
        batch,
        "past_nwp_metadata",
        batch_size,
        CONTEXT_LENGTH,
        len(metadata_names),
    )
    future_nwp_metadata = _feature_sequence(
        batch,
        "future_nwp_metadata",
        batch_size,
        PREDICTION_LENGTH,
        len(metadata_names),
    )

    station_index = _vector(batch, "station_index", batch_size)
    if np.any(~np.isfinite(station_index)) or np.any(station_index < 0):
        raise ValueError("station_index must contain finite non-negative values")
    coordinates = _matrix(batch, "station_coordinates", (batch_size, 2))
    if not np.isfinite(coordinates).all():
        raise ValueError("station_coordinates must be finite")
    capacity_w = _vector(batch, "power_scale_w", batch_size)
    if np.any(~np.isfinite(capacity_w)) or np.any(capacity_w <= 0):
        raise ValueError("power_scale_w must contain finite positive capacities")

    causal_48h = _external_scalar_horizons(
        causal_48h_power, "causal_48h_power", batch_size
    )
    if causal_48h_mask is None:
        causal_48h_valid = np.isfinite(causal_48h)
    else:
        causal_48h_valid = _external_bool_horizons(
            causal_48h_mask, "causal_48h_mask", batch_size
        ) & np.isfinite(causal_48h)

    shared_columns = [
        station_index.astype(np.float32, copy=False),
        coordinates[:, 0],
        coordinates[:, 1],
        capacity_w / np.float32(1_000_000.0),
        np.log1p(capacity_w / np.float32(1_000_000.0)),
    ]
    shared_columns.extend(past_features[:, -1, index] for index in range(len(time_names)))

    for minutes in _PAST_POINT_MINUTES:
        index = -(minutes // 15)
        valid = past_valid[:, index]
        shared_columns.extend(
            (
                _masked(past_power[:, index], valid),
                valid.astype(np.float32),
                (valid & past_observed[:, index]).astype(np.float32),
            )
        )
    for minutes in _PAST_ROLLING_MINUTES:
        length = minutes // 15
        shared_columns.extend(
            _rolling_statistics(
                past_power[:, -length:],
                past_valid[:, -length:],
                past_observed[:, -length:],
            )
        )

    for minutes in _RECENT_POINT_MINUTES:
        index = -(minutes // 5)
        valid = recent_valid[:, index]
        shared_columns.extend(
            (
                _masked(recent_power[:, index], valid),
                valid.astype(np.float32),
                (valid & recent_observed[:, index]).astype(np.float32),
            )
        )
    recent_last = recent_power[:, -1]
    recent_last_valid = recent_valid[:, -1]
    for minutes in _RECENT_DELTA_MINUTES:
        lag = minutes // 5
        previous = recent_power[:, -1 - lag]
        valid = recent_last_valid & recent_valid[:, -1 - lag]
        shared_columns.extend(
            (_masked(recent_last - previous, valid), valid.astype(np.float32))
        )
    for minutes in _RECENT_ROLLING_MINUTES:
        length = minutes // 5
        shared_columns.extend(
            _rolling_statistics(
                recent_power[:, -length:],
                recent_valid[:, -length:],
                recent_observed[:, -length:],
            )
        )

    latest_past_nwp_valid = past_nwp_mask[:, -1, :] & np.isfinite(
        past_nwp[:, -1, :]
    )
    latest_past_nwp = _masked(past_nwp[:, -1, :], latest_past_nwp_valid)
    for index in range(len(nwp_names)):
        shared_columns.extend(
            (
                latest_past_nwp[:, index],
                latest_past_nwp_valid[:, index].astype(np.float32),
            )
        )
    latest_metadata = _metadata_with_nan(
        past_nwp_metadata[:, -1, :], metadata_names
    )
    shared_columns.extend(
        latest_metadata[:, index] for index in range(len(metadata_names))
    )

    shared = _column_stack(shared_columns, "shared")

    horizon_blocks = []
    for horizon in range(PREDICTION_LENGTH):
        lag_24_valid = past_valid[:, horizon]
        future_valid = future_nwp_mask[:, horizon, :] & np.isfinite(
            future_nwp[:, horizon, :]
        )
        future_values = _masked(future_nwp[:, horizon, :], future_valid)
        nwp_delta_valid = future_valid & latest_past_nwp_valid
        nwp_delta = _masked(
            future_nwp[:, horizon, :] - past_nwp[:, -1, :],
            nwp_delta_valid,
        )
        future_metadata = _metadata_with_nan(
            future_nwp_metadata[:, horizon, :], metadata_names
        )
        columns = [
            _masked(past_power[:, horizon], lag_24_valid),
            lag_24_valid.astype(np.float32),
            (lag_24_valid & past_observed[:, horizon]).astype(np.float32),
            _masked(causal_48h[:, horizon], causal_48h_valid[:, horizon]),
            causal_48h_valid[:, horizon].astype(np.float32),
        ]
        columns.extend(
            future_features[:, horizon, index] for index in range(len(time_names))
        )
        for index in range(len(nwp_names)):
            columns.extend(
                (
                    future_values[:, index],
                    future_valid[:, index].astype(np.float32),
                )
            )
        for index in range(len(nwp_names)):
            columns.extend(
                (
                    nwp_delta[:, index],
                    nwp_delta_valid[:, index].astype(np.float32),
                )
            )
        columns.extend(
            future_metadata[:, index] for index in range(len(metadata_names))
        )
        horizon_blocks.append(_column_stack(columns, f"horizon {horizon}"))
    horizon_matrix = np.stack(horizon_blocks, axis=1).astype(
        np.float32, copy=False
    )

    return LightGBMFeatureMatrices(
        shared=shared,
        horizon=horizon_matrix,
        shared_feature_names=shared_feature_names(
            time_feature_names=time_names,
            nwp_feature_names=nwp_names,
            nwp_metadata_feature_names=metadata_names,
        ),
        horizon_feature_names=horizon_feature_names(
            time_feature_names=time_names,
            nwp_feature_names=nwp_names,
            nwp_metadata_feature_names=metadata_names,
        ),
    )


def extract_capacity_normalized_targets(batch: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return ``target`` as ``[B, 16]`` with invalid labels replaced by NaN."""

    target = _scalar_sequence(batch, "target", PREDICTION_LENGTH)
    batch_size = target.shape[0]
    mask = _bool_scalar_sequence(
        batch, "target_observed_mask", batch_size, PREDICTION_LENGTH
    ) & np.isfinite(target)
    return _masked(target, mask), mask


def _validated_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of names")
    names = tuple(values)
    if not names or any(not isinstance(value, str) or not value for value in names):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names")
    return names


def _numpy(value, label: str) -> np.ndarray:
    # Accept CPU tensors without importing torch; all feature computation below
    # remains NumPy-only.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} cannot be converted to a NumPy array") from error


def _required(batch: Mapping, key: str):
    if key not in batch:
        raise KeyError(f"batch is missing required field {key!r}")
    return batch[key]


def _scalar_sequence(
    batch: Mapping,
    key: str,
    length: int,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    values = _numpy(_required(batch, key), key)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != length:
        raise ValueError(f"{key} must have shape [B, {length}] or [B, {length}, 1]")
    if batch_size is not None and values.shape[0] != batch_size:
        raise ValueError(f"{key} batch size differs from past_power")
    return values.astype(np.float32, copy=False)


def _bool_scalar_sequence(
    batch: Mapping, key: str, batch_size: int, length: int
) -> np.ndarray:
    values = _numpy(_required(batch, key), key)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != (batch_size, length):
        raise ValueError(f"{key} must have shape [{batch_size}, {length}, 1]")
    return values.astype(np.bool_, copy=False)


def _feature_sequence(
    batch: Mapping,
    key: str,
    batch_size: int,
    length: int,
    feature_count: int,
    *,
    require_finite: bool = False,
) -> np.ndarray:
    values = _numpy(_required(batch, key), key)
    expected = (batch_size, length, feature_count)
    if values.shape != expected:
        raise ValueError(f"{key} must have shape {expected}")
    values = values.astype(np.float32, copy=False)
    if require_finite and not np.isfinite(values).all():
        raise ValueError(f"{key} must contain only finite values")
    if np.isinf(values).any():
        raise ValueError(f"{key} cannot contain infinity")
    return values


def _bool_feature_sequence(batch: Mapping, key: str, shape: tuple[int, ...]) -> np.ndarray:
    values = _numpy(_required(batch, key), key)
    if values.shape != shape:
        raise ValueError(f"{key} must have shape {shape}")
    return values.astype(np.bool_, copy=False)


def _vector(batch: Mapping, key: str, batch_size: int) -> np.ndarray:
    values = _numpy(_required(batch, key), key)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (batch_size,):
        raise ValueError(f"{key} must have shape [{batch_size}]")
    return values.astype(np.float32, copy=False)


def _matrix(batch: Mapping, key: str, shape: tuple[int, int]) -> np.ndarray:
    values = _numpy(_required(batch, key), key)
    if values.shape != shape:
        raise ValueError(f"{key} must have shape {shape}")
    return values.astype(np.float32, copy=False)


def _external_scalar_horizons(value, label: str, batch_size: int) -> np.ndarray:
    values = _numpy(value, label)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    expected = (batch_size, PREDICTION_LENGTH)
    if values.shape != expected:
        raise ValueError(f"{label} must have shape {expected}")
    values = values.astype(np.float32, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"{label} cannot contain infinity")
    return values


def _external_bool_horizons(value, label: str, batch_size: int) -> np.ndarray:
    values = _numpy(value, label)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    expected = (batch_size, PREDICTION_LENGTH)
    if values.shape != expected:
        raise ValueError(f"{label} must have shape {expected}")
    return values.astype(np.bool_, copy=False)


def _masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_) & np.isfinite(values)
    return np.where(valid, values, np.float32(np.nan)).astype(
        np.float32, copy=False
    )


def _rolling_statistics(
    values: np.ndarray, valid: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, ...]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_) & np.isfinite(values)
    observed = np.asarray(observed, dtype=np.bool_) & valid
    count = valid.sum(axis=1)
    safe_count = np.maximum(count, 1).astype(np.float32)
    filled = np.where(valid, values, np.float32(0.0))
    mean = filled.sum(axis=1, dtype=np.float32) / safe_count
    centered = np.where(valid, values - mean[:, None], np.float32(0.0))
    std = np.sqrt(
        np.square(centered).sum(axis=1, dtype=np.float32) / safe_count
    ).astype(np.float32, copy=False)
    minimum = np.where(valid, values, np.float32(np.inf)).min(axis=1)
    maximum = np.where(valid, values, np.float32(-np.inf)).max(axis=1)
    has_value = count > 0
    mean = _masked(mean, has_value)
    std = _masked(std, has_value)
    minimum = _masked(minimum, has_value)
    maximum = _masked(maximum, has_value)
    value_range = _masked(maximum - minimum, has_value)
    denominator = np.float32(values.shape[1])
    coverage = count.astype(np.float32) / denominator
    observed_fraction = observed.sum(axis=1).astype(np.float32) / denominator
    return (
        mean,
        std,
        minimum,
        maximum,
        value_range,
        coverage,
        observed_fraction,
    )


def _metadata_with_nan(
    values: np.ndarray, feature_names: tuple[str, ...]
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if "run_time_valid" not in feature_names:
        raise ValueError("NWP metadata must contain run_time_valid")
    valid_index = feature_names.index("run_time_valid")
    row_valid = np.isfinite(values[:, valid_index]) & (
        values[:, valid_index] > 0.5
    )
    result = np.where(row_valid[:, None], values, np.float32(np.nan)).astype(
        np.float32, copy=False
    )
    # Validity/status flags stay explicit even when continuous metadata are NaN.
    for index, name in enumerate(feature_names):
        if name == "older_run_fallback" or name.endswith("_valid"):
            result[:, index] = np.where(
                np.isfinite(values[:, index]), values[:, index], np.float32(0.0)
            )
    return result


def _column_stack(columns: Sequence[np.ndarray], label: str) -> np.ndarray:
    if not columns:
        raise ValueError(f"{label} feature block is empty")
    batch_sizes = {np.asarray(column).shape for column in columns}
    if len(batch_sizes) != 1 or len(next(iter(batch_sizes))) != 1:
        raise ValueError(f"{label} feature columns must all have shape [B]")
    result = np.column_stack(columns).astype(np.float32, copy=False)
    if np.isinf(result).any():
        raise ValueError(f"{label} features contain infinity")
    return result


__all__ = [
    "CONTEXT_LENGTH",
    "DEFAULT_NWP_FEATURE_NAMES",
    "DEFAULT_NWP_METADATA_FEATURE_NAMES",
    "DEFAULT_TIME_FEATURE_NAMES",
    "LightGBMFeatureMatrices",
    "PREDICTION_LENGTH",
    "RECENT_5MIN_LENGTH",
    "build_lightgbm_features",
    "extract_capacity_normalized_targets",
    "horizon_feature_names",
    "shared_feature_names",
]
