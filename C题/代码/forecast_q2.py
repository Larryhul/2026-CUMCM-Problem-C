"""Causal photovoltaic forecasting utilities for Question 2.

The forecaster is deliberately independent from the optimization model.  It
fits one multi-output Fourier ridge regression over the 144 intraday slots,
while the training rows are calendar days strictly before the target day.
Early dates use an explicit cold-start curve because an annual Fourier model
cannot be identified reliably from only a few observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourierConfig:
    """Configuration for causal Fourier regression and blocked validation."""

    harmonics: tuple[int, ...] = (1, 2, 3, 4)
    ridge_values: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    min_history: int = 21
    validation_window: int = 30
    overprediction_weight: float = 5.0
    days_per_year: int = 365
    nonnegative_tolerance: float = 1e-9


@dataclass(frozen=True)
class ForecastSelection:
    """Selected hyperparameters and validation diagnostics for one target day."""

    harmonics: int
    ridge: float
    validation_loss: float | None
    validation_days: int
    used_cold_start: bool


def _validate_pv_matrix(actual_pv_kwh: np.ndarray) -> np.ndarray:
    values = np.asarray(actual_pv_kwh, dtype=float)
    if values.ndim != 2 or values.shape[1] != 144:
        raise ValueError("光伏矩阵必须为(n_days,144)")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("光伏矩阵必须为有限非负数")
    return values


def _design_matrix(day_indices: np.ndarray, harmonics: int, days_per_year: int) -> np.ndarray:
    indices = np.asarray(day_indices, dtype=float).reshape(-1)
    if harmonics < 1:
        raise ValueError("Fourier谐波阶数必须为正整数")
    theta = 2.0 * np.pi * indices / float(days_per_year)
    columns = [np.ones(indices.size)]
    for k in range(1, harmonics + 1):
        columns.extend((np.sin(k * theta), np.cos(k * theta)))
    return np.column_stack(columns)


def fit_fourier_ridge(
    actual_pv_kwh: np.ndarray,
    target_index: int,
    harmonics: int,
    ridge: float,
    days_per_year: int = 365,
) -> np.ndarray:
    """Fit on rows ``0..target_index-1`` and predict ``target_index``.

    The target row is never read by this function.  The returned vector is in
    the same energy unit as the input (kWh per ten-minute interval).
    """
    values = _validate_pv_matrix(actual_pv_kwh)
    if not 1 <= target_index <= len(values):
        raise ValueError("target_index必须位于1到历史数据长度之间")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge必须为有限非负数")

    training = values[:target_index]
    design = _design_matrix(np.arange(target_index), harmonics, days_per_year)
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    normal = design.T @ design + ridge * penalty
    coefficients = np.linalg.solve(normal, design.T @ training)
    target_row = _design_matrix(np.array([target_index]), harmonics, days_per_year)[0]
    forecast = target_row @ coefficients

    # Night slots that are always zero in history stay exactly zero.  This
    # prevents a seasonal intercept from inventing night-time generation.
    active_mask = np.max(training, axis=0) > 1e-9
    forecast = np.where(active_mask, forecast, 0.0)
    return np.maximum(forecast, 0.0)


def forecast_fourier_ridge_from_origin(
    actual_pv_kwh: np.ndarray,
    origin_index: int,
    target_indices: np.ndarray,
    harmonics: int,
    ridge: float,
    days_per_year: int = 365,
) -> np.ndarray:
    """Forecast multiple dates using only observations before ``origin_index``."""
    values = _validate_pv_matrix(actual_pv_kwh)
    targets = np.asarray(target_indices, dtype=int).reshape(-1)
    if not 1 <= origin_index <= len(values):
        raise ValueError("origin_index必须位于1到历史数据长度之间")
    if targets.size == 0 or np.any(targets < origin_index):
        raise ValueError("多步预测目标日期不得早于预测原点")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge必须为有限非负数")

    training = values[:origin_index]
    design = _design_matrix(np.arange(origin_index), harmonics, days_per_year)
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    normal = design.T @ design + ridge * penalty
    coefficients = np.linalg.solve(normal, design.T @ training)
    target_design = _design_matrix(targets, harmonics, days_per_year)
    forecast = target_design @ coefficients
    active_mask = np.max(training, axis=0) > 1e-9
    forecast = np.where(active_mask[None, :], forecast, 0.0)
    return np.maximum(forecast, 0.0)


def weighted_overprediction_loss(
    forecast_kwh: np.ndarray,
    actual_kwh: np.ndarray,
    overprediction_weight: float = 5.0,
) -> float:
    """Penalize dangerous PV overprediction at the emergency-price ratio."""
    forecast = np.asarray(forecast_kwh, dtype=float)
    actual = np.asarray(actual_kwh, dtype=float)
    if forecast.shape != actual.shape:
        raise ValueError("预测和实际光伏形状必须一致")
    if overprediction_weight < 1.0:
        raise ValueError("overprediction_weight应不小于1")
    error = forecast - actual
    return float(
        np.mean(
            np.where(
                error > 0,
                overprediction_weight * error,
                -error,
            )
        )
    )


def select_fourier_parameters(
    actual_pv_kwh: np.ndarray,
    target_index: int,
    config: FourierConfig,
    cold_start_kwh: np.ndarray,
) -> ForecastSelection:
    """Select ``(K, ridge)`` using only blocked historical validation.

    For each validation date ``v``, its forecast is fitted on rows before
    ``v``.  The target date and all later rows are never used for selection.
    """
    values = _validate_pv_matrix(actual_pv_kwh)
    fallback = np.asarray(cold_start_kwh, dtype=float)
    if (
        fallback.shape != (values.shape[1],)
        or not np.isfinite(fallback).all()
        or np.any(fallback < 0)
    ):
        raise ValueError("cold_start_kwh必须是长度为144的非负向量")
    if target_index < 1:
        return ForecastSelection(1, config.ridge_values[0], None, 0, True)
    if target_index < config.min_history:
        return ForecastSelection(1, config.ridge_values[0], None, 0, True)

    validation_start = max(config.min_history, target_index - config.validation_window)
    validation_days = np.arange(validation_start, target_index, dtype=int)
    best: tuple[float, int, float] | None = None
    for harmonics in config.harmonics:
        # At least two rows per Fourier coefficient keeps the validation fit
        # away from the nearly-interpolating regime.
        if config.min_history < 2 * (2 * harmonics + 1):
            continue
        for ridge in config.ridge_values:
            losses: list[float] = []
            for validation_day in validation_days:
                prediction = fit_fourier_ridge(
                    values,
                    int(validation_day),
                    harmonics,
                    ridge,
                    config.days_per_year,
                )
                losses.append(
                    weighted_overprediction_loss(
                        prediction,
                        values[validation_day],
                        config.overprediction_weight,
                    )
                )
            if not losses:
                continue
            candidate = (float(np.mean(losses)), harmonics, ridge)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return ForecastSelection(1, config.ridge_values[0], None, 0, True)
    return ForecastSelection(
        harmonics=best[1],
        ridge=best[2],
        validation_loss=best[0],
        validation_days=len(validation_days),
        used_cold_start=False,
    )


def build_causal_forecast_history(
    actual_pv_kwh: np.ndarray,
    cold_start_kwh: np.ndarray,
    config: FourierConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[ForecastSelection, ...]]:
    """Build one forecast for every date with no target-day leakage.

    Returns ``(forecasts, valid_mask, selections)``.  ``valid_mask`` marks
    dates with enough history for a Fourier error sample.  Cold-start forecasts
    remain available in ``forecasts`` but are excluded from error scenarios.
    """
    values = _validate_pv_matrix(actual_pv_kwh)
    fallback = np.asarray(cold_start_kwh, dtype=float)
    if (
        fallback.shape != (values.shape[1],)
        or not np.isfinite(fallback).all()
        or np.any(fallback < 0)
    ):
        raise ValueError("cold_start_kwh必须是长度为144的非负向量")
    cfg = config or FourierConfig()
    forecasts = np.tile(fallback, (len(values), 1))
    valid = np.zeros(len(values), dtype=bool)
    selections: list[ForecastSelection] = []
    for target_index in range(len(values)):
        selection = select_fourier_parameters(values, target_index, cfg, fallback)
        selections.append(selection)
        if selection.used_cold_start:
            continue
        forecasts[target_index] = fit_fourier_ridge(
            values,
            target_index,
            selection.harmonics,
            selection.ridge,
            cfg.days_per_year,
        )
        valid[target_index] = True
    return forecasts, valid, tuple(selections)


def build_causal_multiday_forecast_history(
    actual_pv_kwh: np.ndarray,
    cold_start_kwh: np.ndarray,
    selections: tuple[ForecastSelection, ...],
    max_horizon_days: int,
    config: FourierConfig | None = None,
) -> np.ndarray:
    """Build rolling multi-step forecasts without advancing the information set.

    ``result[d, lead]`` predicts day ``d + lead`` using observations strictly
    before day ``d``.  Consequently all leads in one planning window share the
    same origin information and hyperparameter selection.
    """
    values = _validate_pv_matrix(actual_pv_kwh)
    fallback = np.asarray(cold_start_kwh, dtype=float)
    if fallback.shape != (values.shape[1],) or np.any(fallback < 0):
        raise ValueError("cold_start_kwh必须是长度为144的非负向量")
    if len(selections) != len(values):
        raise ValueError("selections长度必须与日期数一致")
    if max_horizon_days < 1:
        raise ValueError("max_horizon_days必须为正整数")

    cfg = config or FourierConfig()
    forecasts = np.full(
        (len(values), max_horizon_days, values.shape[1]), np.nan, dtype=float
    )
    for origin_index, selection in enumerate(selections):
        horizon = min(max_horizon_days, len(values) - origin_index)
        if selection.used_cold_start:
            forecasts[origin_index, :horizon] = fallback
            continue
        targets = np.arange(origin_index, origin_index + horizon, dtype=int)
        forecasts[origin_index, :horizon] = forecast_fourier_ridge_from_origin(
            values,
            origin_index,
            targets,
            selection.harmonics,
            selection.ridge,
            cfg.days_per_year,
        )
    return forecasts


def build_causal_error_history(
    actual_pv_kwh: np.ndarray,
    forecasts_kwh: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return only valid historical residual curves in chronological order."""
    actual = _validate_pv_matrix(actual_pv_kwh)
    forecasts = _validate_pv_matrix(forecasts_kwh)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != (len(actual),):
        raise ValueError("valid_mask长度错误")
    if np.any(mask & (np.arange(len(actual)) < 1)):
        raise ValueError("第1天没有历史数据，不能作为有效预测误差")
    return actual[mask] - forecasts[mask]
