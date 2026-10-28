"""Local TimeMixer adaptation for direct photovoltaic power forecasts.

Past decomposable mixing and future multipredictor mixing combine temporal scales.
Separate projections condition historical and future weather covariates.
Returns capacity-normalized power [batch, 16, 1].
This is the local implementation used in the study, not an unmodified official model."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class _MovingAverageDecomposition(nn.Module):
    """Replicate-padded moving-average trend and residual seasonal signal."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if isinstance(kernel_size, bool) or int(kernel_size) <= 0:
            raise ValueError("moving_average must be a positive odd integer")
        if int(kernel_size) % 2 == 0:
            raise ValueError("moving_average must be odd")
        self.kernel_size = int(kernel_size)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = (self.kernel_size - 1) // 2
        padded = F.pad(
            values.transpose(1, 2),
            (padding, padding),
            mode="replicate",
        )
        trend = F.avg_pool1d(
            padded,
            kernel_size=self.kernel_size,
            stride=1,
        ).transpose(1, 2)
        return values - trend, trend


class _TemporalProjection(nn.Module):
    """Apply an MLP along time independently for every latent channel."""

    def __init__(self, input_length: int, output_length: int) -> None:
        super().__init__()
        hidden_length = max(input_length, output_length)
        self.network = nn.Sequential(
            nn.Linear(input_length, hidden_length),
            nn.GELU(),
            nn.Linear(hidden_length, output_length),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.transpose(1, 2)).transpose(1, 2)


class _PastDecomposableMixing(nn.Module):
    """One PDM block with bottom-up season and top-down trend mixing."""

    def __init__(
        self,
        *,
        lengths: tuple[int, ...],
        d_model: int,
        d_ff: int,
        moving_average: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decomposition = _MovingAverageDecomposition(moving_average)
        self.component_projection = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.season_down = nn.ModuleList(
            _TemporalProjection(lengths[index], lengths[index + 1])
            for index in range(len(lengths) - 1)
        )
        self.trend_up = nn.ModuleList(
            _TemporalProjection(lengths[index + 1], lengths[index])
            for index in range(len(lengths) - 1)
        )
        self.output_norms = nn.ModuleList(
            nn.LayerNorm(d_model) for _ in lengths
        )
        self.output_projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in lengths
        )

    def forward(self, scales: list[torch.Tensor]) -> list[torch.Tensor]:
        seasons: list[torch.Tensor] = []
        trends: list[torch.Tensor] = []
        for values in scales:
            season, trend = self.decomposition(values)
            seasons.append(self.component_projection(season))
            trends.append(self.component_projection(trend))

        mixed_seasons = [seasons[0]]
        for index, projection in enumerate(self.season_down):
            mixed_seasons.append(
                seasons[index + 1] + projection(mixed_seasons[-1])
            )

        mixed_trends: list[torch.Tensor | None] = [None] * len(trends)
        mixed_trends[-1] = trends[-1]
        for index in range(len(trends) - 2, -1, -1):
            assert mixed_trends[index + 1] is not None
            mixed_trends[index] = trends[index] + self.trend_up[index](
                mixed_trends[index + 1]
            )

        outputs: list[torch.Tensor] = []
        for original, season, trend, norm, projection in zip(
            scales,
            mixed_seasons,
            mixed_trends,
            self.output_norms,
            self.output_projections,
        ):
            assert trend is not None
            mixed = season + trend
            outputs.append(original + projection(norm(mixed)))
        return outputs


class _FutureScalePredictor(nn.Module):
    """One FMM scale-specific direct predictor."""

    def __init__(
        self,
        *,
        input_length: int,
        prediction_length: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_projection = nn.Linear(input_length, prediction_length)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        encoded: torch.Tensor,
        future_context: torch.Tensor,
        static_context: torch.Tensor,
    ) -> torch.Tensor:
        forecast_state = self.temporal_projection(
            encoded.transpose(1, 2)
        ).transpose(1, 2)
        forecast_state = forecast_state + future_context + static_context[:, None, :]
        return self.output(forecast_state)


class PVTimeMixer(nn.Module):
    """Multi-station TimeMixer using causal historical and future weather."""

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
        d_model: int = 256,
        d_ff: int = 512,
        e_layers: int = 3,
        down_sampling_layers: int = 2,
        down_sampling_window: int = 2,
        moving_average: int = 25,
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
            ("d_model", d_model),
            ("d_ff", d_ff),
            ("e_layers", e_layers),
            ("down_sampling_window", down_sampling_window),
            ("station_embedding_dim", station_embedding_dim),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(down_sampling_layers, bool) or int(down_sampling_layers) < 0:
            raise ValueError("down_sampling_layers cannot be negative")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if int(moving_average) <= 0 or int(moving_average) % 2 == 0:
            raise ValueError("moving_average must be a positive odd integer")
        if use_historical_nwp and not use_nwp:
            raise ValueError("historical NWP requires use_nwp=True")

        lengths = [int(context_length)]
        for _ in range(int(down_sampling_layers)):
            previous = lengths[-1]
            if previous % int(down_sampling_window) != 0:
                raise ValueError(
                    "context_length must be divisible by every down-sampling scale"
                )
            lengths.append(previous // int(down_sampling_window))

        self.num_stations = int(num_stations)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.past_feature_dim = int(past_feature_dim)
        self.future_feature_dim = int(future_feature_dim)
        self.nwp_feature_dim = int(nwp_feature_dim)
        self.use_nwp = bool(use_nwp)
        self.use_historical_nwp = bool(use_historical_nwp)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.e_layers = int(e_layers)
        self.down_sampling_layers = int(down_sampling_layers)
        self.down_sampling_window = int(down_sampling_window)
        self.moving_average = int(moving_average)
        self.station_embedding_dim = int(station_embedding_dim)
        self.dropout = float(dropout)
        self.scale_lengths = tuple(lengths)

        # Normalized power, time/solar features, and three distinct power masks.
        history_input_dim = 1 + self.past_feature_dim + 3
        if self.use_historical_nwp:
            history_input_dim += 2 * self.nwp_feature_dim
        future_input_dim = self.future_feature_dim
        if self.use_nwp:
            future_input_dim += 2 * self.nwp_feature_dim

        self.history_projection = nn.Sequential(
            nn.Linear(history_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.future_projection = nn.Sequential(
            nn.Linear(future_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.station_embedding = nn.Embedding(
            self.num_stations,
            self.station_embedding_dim,
        )
        self.static_projection = nn.Sequential(
            nn.Linear(self.station_embedding_dim + 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.pdm_blocks = nn.ModuleList(
            _PastDecomposableMixing(
                lengths=self.scale_lengths,
                d_model=self.d_model,
                d_ff=self.d_ff,
                moving_average=self.moving_average,
                dropout=self.dropout,
            )
            for _ in range(self.e_layers)
        )
        self.future_predictors = nn.ModuleList(
            _FutureScalePredictor(
                input_length=length,
                prediction_length=self.prediction_length,
                d_model=self.d_model,
                dropout=self.dropout,
            )
            for length in self.scale_lengths
        )
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
        # The direct forecast starts close to the valid-history instance mean.
        for predictor in self.future_predictors:
            final = predictor.output[-1]
            assert isinstance(final, nn.Linear)
            nn.init.xavier_uniform_(final.weight, gain=0.01)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _required(batch: Mapping, name: str) -> torch.Tensor:
        value = batch.get(name)
        if not torch.is_tensor(value):
            raise KeyError(f"TimeMixer batch requires Tensor {name!r}")
        return value

    @staticmethod
    def _validate_shape(
        name: str,
        value: torch.Tensor,
        shape: tuple[int, ...],
    ) -> None:
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(value.shape)}"
            )

    def _instance_normalize_power(
        self,
        power: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = valid.sum(dim=1, keepdim=True)
        if (count == 0).any().item():
            raise ValueError("Every sample needs valid historical power")
        valid_float = valid.to(power.dtype)
        mean = (power * valid_float).sum(dim=1, keepdim=True) / count
        variance = (
            (power - mean).square() * valid_float
        ).sum(dim=1, keepdim=True) / count
        scale = variance.clamp_min(1e-6).sqrt()
        normalized = torch.where(valid, (power - mean) / scale, 0.0)
        return normalized, mean, scale

    def _build_scales(self, embedded: torch.Tensor) -> list[torch.Tensor]:
        scales = [embedded]
        current = embedded
        for _ in range(self.down_sampling_layers):
            current = F.avg_pool1d(
                current.transpose(1, 2),
                kernel_size=self.down_sampling_window,
                stride=self.down_sampling_window,
            ).transpose(1, 2)
            scales.append(current)
        return scales

    def forward(self, batch: Mapping) -> torch.Tensor:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        device = self.station_embedding.weight.device
        past_power = self._required(batch, "past_power").to(
            device=device,
            dtype=torch.float32,
        )
        batch_size = past_power.shape[0]
        self._validate_shape(
            "past_power",
            past_power,
            (batch_size, self.context_length, 1),
        )
        past_features = self._required(batch, "past_features").to(
            device=device,
            dtype=torch.float32,
        )
        future_features = self._required(batch, "future_features").to(
            device=device,
            dtype=torch.float32,
        )
        observed = self._required(batch, "past_observed_mask").to(device).bool()
        complete = self._required(batch, "past_complete_mask").to(device).bool()
        source_span = self._required(batch, "past_source_span_mask").to(device).bool()
        station_index = self._required(batch, "station_index").to(
            device=device,
            dtype=torch.long,
        )
        coordinates = self._required(batch, "station_coordinates").to(
            device=device,
            dtype=torch.float32,
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
        normalized_power, power_mean, power_scale = self._instance_normalize_power(
            power,
            power_valid,
        )
        history_parts: tuple[torch.Tensor, ...] = (
            normalized_power[:, :, None],
            past_features,
            observed.to(torch.float32),
            complete.to(torch.float32),
            source_span.to(torch.float32),
        )
        if self.use_historical_nwp:
            past_nwp = self._required(batch, "past_nwp").to(
                device=device,
                dtype=torch.float32,
            )
            past_nwp_mask = self._required(batch, "past_nwp_mask").to(device).bool()
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
            history_parts += (
                torch.where(past_nwp_mask, past_nwp, 0.0),
                past_nwp_mask.to(torch.float32),
            )
        embedded_history = self.history_projection(
            torch.cat(history_parts, dim=-1)
        )
        embedded_history = embedded_history * source_span.to(embedded_history.dtype)

        future_parts: tuple[torch.Tensor, ...] = (future_features,)
        if self.use_nwp:
            future_nwp = self._required(batch, "future_nwp").to(
                device=device,
                dtype=torch.float32,
            )
            future_nwp_mask = self._required(batch, "future_nwp_mask").to(device).bool()
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
            future_parts += (
                torch.where(future_nwp_mask, future_nwp, 0.0),
                future_nwp_mask.to(torch.float32),
            )
        future_context = self.future_projection(torch.cat(future_parts, dim=-1))

        normalized_coordinates = torch.stack(
            (coordinates[:, 0] / 90.0, coordinates[:, 1] / 180.0),
            dim=-1,
        )
        static_context = self.static_projection(
            torch.cat(
                (self.station_embedding(station_index), normalized_coordinates),
                dim=-1,
            )
        )

        scales = self._build_scales(embedded_history)
        for block in self.pdm_blocks:
            scales = block(scales)
        scale_forecasts = [
            predictor(encoded, future_context, static_context)
            for predictor, encoded in zip(self.future_predictors, scales)
        ]
        normalized_forecast = torch.stack(scale_forecasts, dim=0).sum(dim=0)
        forecast = normalized_forecast * power_scale[:, :, None] + power_mean[:, :, None]
        return forecast


def build_pv_timemixer_from_dataset(
    dataset,
    *,
    nwp_feature_dim: int,
    use_nwp: bool = True,
    use_historical_nwp: bool = True,
    **kwargs,
) -> PVTimeMixer:
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
    return PVTimeMixer(
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


__all__ = ["PVTimeMixer", "build_pv_timemixer_from_dataset"]
