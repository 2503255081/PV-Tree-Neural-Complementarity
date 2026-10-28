"""Local N-HiTS variant for direct photovoltaic power forecasts.

Historical and future weather covariates condition multirate pooled blocks.
Returns power normalized by fixed site capacity, shape [batch, 16, 1].
This is the local implementation used in the study, not the authors' official N-HiTS package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_positive_int_sequence(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


class _NHITSxBlock(nn.Module):
    """One pooled backcast/forecast block with exogenous conditioning."""

    def __init__(
        self,
        *,
        context_length: int,
        prediction_length: int,
        history_exog_dim: int,
        future_exog_dim: int,
        static_dim: int,
        pool_kernel: int,
        frequency_downsample: int,
        hidden_size: int,
        mlp_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.pool_kernel = int(pool_kernel)
        self.frequency_downsample = int(frequency_downsample)

        pooled_history_length = math.ceil(self.context_length / self.pool_kernel)
        pooled_future_length = math.ceil(
            self.prediction_length / self.pool_kernel
        )
        self.backcast_coefficient_count = max(
            1,
            math.ceil(self.context_length / self.frequency_downsample),
        )
        self.forecast_coefficient_count = max(
            1,
            math.ceil(self.prediction_length / self.frequency_downsample),
        )
        input_size = (
            pooled_history_length
            + pooled_history_length * int(history_exog_dim)
            + pooled_future_length * int(future_exog_dim)
            + int(static_dim)
        )

        layers: list[nn.Module] = []
        current_size = input_size
        for _ in range(int(mlp_layers)):
            layers.extend(
                (
                    nn.Linear(current_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            current_size = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.theta = nn.Linear(
            current_size,
            self.backcast_coefficient_count + self.forecast_coefficient_count,
        )

    def _pool(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B, L, C]. ceil_mode retains the most recent partial bin.
        return F.avg_pool1d(
            values.transpose(1, 2),
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
            ceil_mode=True,
        ).transpose(1, 2)

    @staticmethod
    def _interpolate(coefficients: torch.Tensor, output_length: int) -> torch.Tensor:
        if coefficients.shape[1] == output_length:
            return coefficients
        return F.interpolate(
            coefficients[:, None, :],
            size=output_length,
            mode="linear",
            align_corners=False,
        )[:, 0, :]

    def forward(
        self,
        residual: torch.Tensor,
        history_exog: torch.Tensor,
        future_exog: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_residual = self._pool(residual[:, :, None]).flatten(1)
        pooled_history = self._pool(history_exog).flatten(1)
        pooled_future = self._pool(future_exog).flatten(1)
        hidden = self.mlp(
            torch.cat(
                (pooled_residual, pooled_history, pooled_future, static),
                dim=-1,
            )
        )
        theta = self.theta(hidden)
        backcast_coefficients, forecast_coefficients = torch.split(
            theta,
            (self.backcast_coefficient_count, self.forecast_coefficient_count),
            dim=-1,
        )
        backcast = self._interpolate(
            backcast_coefficients,
            self.context_length,
        )
        forecast = self._interpolate(
            forecast_coefficients,
            self.prediction_length,
        )
        return backcast, forecast


class PVNHITSx(nn.Module):
    """Multi-station N-HiTSx-style model using causal weather covariates."""

    def __init__(
        self,
        *,
        num_stations: int,
        context_length: int = 96,
        prediction_length: int = 16,
        past_feature_dim: int = 7,
        future_feature_dim: int = 6,
        nwp_feature_dim: int = 15,
        use_nwp: bool = True,
        use_historical_nwp: bool = True,
        hidden_size: int = 512,
        mlp_layers: int = 2,
        blocks_per_stack: int = 1,
        pool_kernels: Sequence[int] = (8, 4, 1),
        frequency_downsamples: Sequence[int] = (4, 2, 1),
        history_exog_size: int = 16,
        future_exog_size: int = 16,
        station_embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_stations", num_stations),
            ("context_length", context_length),
            ("prediction_length", prediction_length),
            ("past_feature_dim", past_feature_dim),
            ("future_feature_dim", future_feature_dim),
            ("nwp_feature_dim", nwp_feature_dim),
            ("hidden_size", hidden_size),
            ("mlp_layers", mlp_layers),
            ("blocks_per_stack", blocks_per_stack),
            ("history_exog_size", history_exog_size),
            ("future_exog_size", future_exog_size),
            ("station_embedding_dim", station_embedding_dim),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        pool_kernels = _validate_positive_int_sequence(
            "pool_kernels", pool_kernels
        )
        frequency_downsamples = _validate_positive_int_sequence(
            "frequency_downsamples", frequency_downsamples
        )
        if len(pool_kernels) != len(frequency_downsamples):
            raise ValueError(
                "pool_kernels and frequency_downsamples must have equal length"
            )
        if use_historical_nwp and not use_nwp:
            raise ValueError("historical NWP requires use_nwp=True")

        self.num_stations = int(num_stations)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.past_feature_dim = int(past_feature_dim)
        self.future_feature_dim = int(future_feature_dim)
        self.nwp_feature_dim = int(nwp_feature_dim)
        self.use_nwp = bool(use_nwp)
        self.use_historical_nwp = bool(use_historical_nwp)
        self.hidden_size = int(hidden_size)
        self.mlp_layers = int(mlp_layers)
        self.blocks_per_stack = int(blocks_per_stack)
        self.pool_kernels = pool_kernels
        self.frequency_downsamples = frequency_downsamples
        self.history_exog_size = int(history_exog_size)
        self.future_exog_size = int(future_exog_size)
        self.station_embedding_dim = int(station_embedding_dim)
        self.dropout = float(dropout)

        # Power availability flags are exogenous inputs, not replacement labels.
        raw_history_exog_dim = self.past_feature_dim + 2
        if self.use_historical_nwp:
            raw_history_exog_dim += 2 * self.nwp_feature_dim
        raw_future_exog_dim = self.future_feature_dim
        if self.use_nwp:
            raw_future_exog_dim += 2 * self.nwp_feature_dim

        self.history_exog_projection = nn.Sequential(
            nn.Linear(raw_history_exog_dim, self.history_exog_size),
            nn.LayerNorm(self.history_exog_size),
            nn.GELU(),
        )
        self.future_exog_projection = nn.Sequential(
            nn.Linear(raw_future_exog_dim, self.future_exog_size),
            nn.LayerNorm(self.future_exog_size),
            nn.GELU(),
        )
        self.station_embedding = nn.Embedding(
            self.num_stations,
            self.station_embedding_dim,
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, self.station_embedding_dim),
            nn.GELU(),
        )
        static_dim = 2 * self.station_embedding_dim

        blocks: list[nn.Module] = []
        for pool_kernel, frequency_downsample in zip(
            self.pool_kernels,
            self.frequency_downsamples,
        ):
            for _ in range(self.blocks_per_stack):
                blocks.append(
                    _NHITSxBlock(
                        context_length=self.context_length,
                        prediction_length=self.prediction_length,
                        history_exog_dim=self.history_exog_size,
                        future_exog_dim=self.future_exog_size,
                        static_dim=static_dim,
                        pool_kernel=pool_kernel,
                        frequency_downsample=frequency_downsample,
                        hidden_size=self.hidden_size,
                        mlp_layers=self.mlp_layers,
                        dropout=self.dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.station_embedding.weight, mean=0.0, std=0.02)
        # Start each block as a small correction around the last observation.
        for block in self.blocks:
            nn.init.xavier_uniform_(block.theta.weight, gain=0.01)
            nn.init.zeros_(block.theta.bias)

    @staticmethod
    def _required(batch: Mapping, name: str) -> torch.Tensor:
        value = batch.get(name)
        if not torch.is_tensor(value):
            raise KeyError(f"N-HiTSx batch requires Tensor {name!r}")
        return value

    @staticmethod
    def _validate_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")

    @staticmethod
    def _last_valid(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            values.shape[1], device=values.device, dtype=torch.long
        )[None, :]
        chosen = torch.where(valid, positions, positions.new_full((), -1)).max(dim=1).values
        if (chosen < 0).any().item():
            raise ValueError("Every sample needs at least one valid historical power value")
        return values.gather(1, chosen[:, None])[:, 0]

    def forward(self, batch: Mapping) -> torch.Tensor:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        device = self.station_embedding.weight.device
        past_power = self._required(batch, "past_power").to(device=device, dtype=torch.float32)
        batch_size = past_power.shape[0]
        self._validate_shape(
            "past_power", past_power, (batch_size, self.context_length, 1)
        )
        past_features = self._required(batch, "past_features").to(
            device=device, dtype=torch.float32
        )
        future_features = self._required(batch, "future_features").to(
            device=device, dtype=torch.float32
        )
        observed = self._required(batch, "past_observed_mask").to(device=device).bool()
        complete = self._required(batch, "past_complete_mask").to(device=device).bool()
        source_span = self._required(batch, "past_source_span_mask").to(device=device).bool()
        station_index = self._required(batch, "station_index").to(device=device, dtype=torch.long)
        coordinates = self._required(batch, "station_coordinates").to(
            device=device, dtype=torch.float32
        )
        self._validate_shape(
            "past_features",
            past_features,
            (batch_size, self.context_length, self.past_feature_dim),
        )
        self._validate_shape(
            "future_features",
            future_features,
            (batch_size, self.prediction_length, self.future_feature_dim),
        )
        for name, value in (
            ("past_observed_mask", observed),
            ("past_complete_mask", complete),
            ("past_source_span_mask", source_span),
        ):
            self._validate_shape(name, value, (batch_size, self.context_length, 1))
        self._validate_shape("station_index", station_index, (batch_size,))
        self._validate_shape("station_coordinates", coordinates, (batch_size, 2))
        if (station_index < 0).any().item() or (
            station_index >= self.num_stations
        ).any().item():
            raise ValueError("station_index is outside the configured range")

        power_valid = observed[:, :, 0] & complete[:, :, 0] & source_span[:, :, 0]
        power = torch.where(power_valid, past_power[:, :, 0], 0.0)
        history_parts = (
            past_features,
            observed.to(torch.float32),
            complete.to(torch.float32),
        )
        if self.use_historical_nwp:
            past_nwp = self._required(batch, "past_nwp").to(
                device=device, dtype=torch.float32
            )
            past_nwp_mask = self._required(batch, "past_nwp_mask").to(device=device).bool()
            self._validate_shape(
                "past_nwp",
                past_nwp,
                (batch_size, self.context_length, self.nwp_feature_dim),
            )
            self._validate_shape(
                "past_nwp_mask",
                past_nwp_mask,
                (batch_size, self.context_length, self.nwp_feature_dim),
            )
            past_nwp = torch.where(past_nwp_mask, past_nwp, 0.0)
            history_parts = history_parts + (
                past_nwp,
                past_nwp_mask.to(torch.float32),
            )
        history_exog = self.history_exog_projection(torch.cat(history_parts, dim=-1))
        history_exog = history_exog * source_span.to(history_exog.dtype)

        future_parts = (future_features,)
        if self.use_nwp:
            future_nwp = self._required(batch, "future_nwp").to(
                device=device, dtype=torch.float32
            )
            future_nwp_mask = self._required(batch, "future_nwp_mask").to(device=device).bool()
            self._validate_shape(
                "future_nwp",
                future_nwp,
                (batch_size, self.prediction_length, self.nwp_feature_dim),
            )
            self._validate_shape(
                "future_nwp_mask",
                future_nwp_mask,
                (batch_size, self.prediction_length, self.nwp_feature_dim),
            )
            future_nwp = torch.where(future_nwp_mask, future_nwp, 0.0)
            future_parts = future_parts + (
                future_nwp,
                future_nwp_mask.to(torch.float32),
            )
        future_exog = self.future_exog_projection(torch.cat(future_parts, dim=-1))

        normalized_coordinates = torch.stack(
            (coordinates[:, 0] / 90.0, coordinates[:, 1] / 180.0),
            dim=-1,
        )
        static = torch.cat(
            (
                self.station_embedding(station_index),
                self.coordinate_projection(normalized_coordinates),
            ),
            dim=-1,
        )

        last_power = self._last_valid(power, power_valid)
        residual = power
        forecast = last_power[:, None].expand(-1, self.prediction_length).clone()
        valid_float = power_valid.to(power.dtype)
        for block in self.blocks:
            backcast, block_forecast = block(
                residual,
                history_exog,
                future_exog,
                static,
            )
            residual = (residual - backcast) * valid_float
            forecast = forecast + block_forecast
        return forecast[:, :, None]


def build_pv_nhitsx_from_dataset(
    dataset,
    *,
    nwp_feature_dim: int,
    use_nwp: bool = True,
    use_historical_nwp: bool = True,
    **kwargs,
) -> PVNHITSx:
    required = (
        "station_ids",
        "context_length",
        "prediction_length",
        "past_feature_names",
        "time_feature_names",
    )
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise TypeError("dataset is missing attributes: " + ", ".join(missing))
    if bool(getattr(dataset, "use_nwp", False)) != bool(use_nwp):
        raise ValueError("dataset NWP mode differs from model use_nwp")
    if use_historical_nwp and not bool(
        getattr(dataset, "nwp_has_historical", False)
    ):
        raise ValueError("historical NWP model requires a schema-v2 NWP overlay")
    return PVNHITSx(
        num_stations=len(dataset.station_ids),
        context_length=int(dataset.context_length),
        prediction_length=int(dataset.prediction_length),
        past_feature_dim=len(dataset.past_feature_names),
        future_feature_dim=len(dataset.time_feature_names),
        nwp_feature_dim=int(nwp_feature_dim),
        use_nwp=use_nwp,
        use_historical_nwp=use_historical_nwp,
        **kwargs,
    )


__all__ = ["PVNHITSx", "build_pv_nhitsx_from_dataset"]
