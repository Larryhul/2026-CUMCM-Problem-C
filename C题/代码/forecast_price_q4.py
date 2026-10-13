"""Causal price forecasts and price-risk scenarios for Question 4.

The attachment-4 price matrix is treated as an observed history.  Every
forecast is fitted with rows strictly before the target date; the target row
is never used as a feature or as a calibration sample.  The module keeps the
forecasting layer independent from the optimisation and settlement layers so
that an audit can distinguish information available to the controller from
the realised tariff used for back-testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import numpy as np


PERIODS = 144
HOURS = 24
SLOTS_PER_HOUR = 6
DAY_PERIOD = 365.0


def release_first_index(release_hour: int) -> int:
    """Return the first ten-minute interval revised at a release.

    Attachment columns are timestamps at the *start* of each source interval:
    ``0:10`` denotes ``0:10-0:20`` and the final ``0:00+1`` denotes
    ``24:00-24:10``.  Consequently the first source row covered by a forecast
    published at ``h:00`` is the row labelled ``h:00``, whose zero-based index
    is ``6*h-1`` for ``h>0``.  The 00:00 release is the special day-start
    case and begins at index 0.
    """
    release_hour = int(release_hour)
    if release_hour not in (0, 6, 12, 18):
        raise ValueError("发布时间必须为0、6、12或18时")
    return 0 if release_hour == 0 else release_hour * SLOTS_PER_HOUR - 1


def release_lead_hours(release_hour: int) -> int:
    """Number of same-day hourly forecast values at a release."""
    release_hour = int(release_hour)
    release_first_index(release_hour)
    return HOURS - release_hour


def release_slot_hour_index(slot: int, release_hour: int) -> int | None:
    """Map a ten-minute slot to the Q3 release-relative hour block.

    The first block begins at the source row labelled with the release
    timestamp and contains six slots.  Subsequent hourly blocks also contain
    six slots.
    """
    slot = int(slot)
    if slot < 0 or slot >= PERIODS:
        raise ValueError("十分钟区间索引超出范围")
    release_hour = int(release_hour)
    start = release_first_index(release_hour)
    if slot < start:
        return None
    # The source row labelled h:00 is the first row after an h:00 update.
    # Index relative to that row so every complete hour block has exactly six
    # ten-minute source rows.  The final 0:00+1 row is retained in the last
    # same-day block because the supplied daily horizon has 144 rows.
    hour = (slot - start) // SLOTS_PER_HOUR
    return min(hour, release_lead_hours(release_hour) - 1)


@dataclass(frozen=True)
class PriceForecastConfig:
    """Configuration for the causal Fourier-ridge price model."""

    harmonics: tuple[int, ...] = (1, 2, 3, 4)
    weekly_harmonics: tuple[int, ...] = (1, 2)
    ridge: float = 10.0
    min_history: int = 14
    recent_window: int = 7
    weekday_window: int = 4
    weekday_blend: float = 0.70
    error_scenario_count: int = 8
    stress_probability: float = 0.10
    stress_quantile: float = 0.90


def _validate_prices(prices: np.ndarray) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or values.shape[1] != PERIODS:
        raise ValueError(f"价格历史必须为(n,{PERIODS})矩阵")
    if values.shape[0] and (not np.isfinite(values).all() or np.any(values < 0.0)):
        raise ValueError("价格历史必须为有限非负数")
    return values


def _design(day_indices: np.ndarray, config: PriceForecastConfig) -> np.ndarray:
    """Return annual and weekly Fourier features for integer day indices."""
    day_indices = np.asarray(day_indices, dtype=float).reshape(-1)
    columns = [np.ones(day_indices.size, dtype=float)]
    for harmonic in config.harmonics:
        angle = 2.0 * np.pi * float(harmonic) * day_indices / DAY_PERIOD
        columns.extend((np.sin(angle), np.cos(angle)))
    for harmonic in config.weekly_harmonics:
        angle = 2.0 * np.pi * float(harmonic) * day_indices / 7.0
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def _fit_ridge(history: np.ndarray, target_index: int, config: PriceForecastConfig) -> np.ndarray:
    """Fit all 144 price slots jointly using only rows before target_index."""
    if target_index <= 0:
        raise ValueError("至少需要一个历史价格日")
    x = _design(np.arange(target_index), config)
    xtx = x.T @ x
    penalty = np.eye(x.shape[1], dtype=float) * float(config.ridge)
    penalty[0, 0] = 0.0
    # solve is more stable than explicitly forming the inverse and is fast for
    # this small feature matrix; lstsq provides a fallback for rank deficiency.
    try:
        coefficients = np.linalg.solve(xtx + penalty, x.T @ history[:target_index])
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(x.T @ x + penalty, x.T @ history[:target_index], rcond=None)[0]
    return _design(np.asarray([target_index]), config) @ coefficients


def forecast_day_price(
    prices: np.ndarray,
    target_index: int,
    base_curve: np.ndarray,
    config: PriceForecastConfig = PriceForecastConfig(),
) -> np.ndarray:
    """Forecast a complete 144-slot tariff curve for ``target_index``.

    The cold-start curve is blended with the most recent observed week until
    the Fourier model has ``min_history`` rows.  All blending inputs are
    historical rows, so the function remains causal at the beginning of the
    year as well.
    """
    values = _validate_prices(prices)
    base = np.asarray(base_curve, dtype=float)
    if base.shape != (PERIODS,) or not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("价格冷启动曲线必须是长度144的有限非负向量")
    if not 0 <= int(target_index) <= values.shape[0]:
        raise ValueError("目标日期索引超出价格历史范围")
    target_index = int(target_index)
    if target_index == 0:
        return base.copy()
    history = values[:target_index]
    recent = history[-min(len(history), max(1, int(config.recent_window))):].mean(axis=0)
    if len(history) < max(1, int(config.min_history)):
        weight = len(history) / float(max(1, int(config.min_history)))
        return np.maximum((1.0 - weight) * base + weight * recent, 0.0)
    prediction = _fit_ridge(values, target_index, config).reshape(PERIODS)
    prediction = _blend_same_weekday_price(
        values, target_index, prediction, config, history_limit=target_index
    )
    # Prices are non-negative.  A very small numerical negative value can be
    # clipped without changing the fitted model; no target-day information is
    # introduced by this projection.
    return np.maximum(prediction, 0.0)


def forecast_from_origin(
    prices: np.ndarray,
    origin_index: int,
    target_indices: np.ndarray | list[int],
    base_curve: np.ndarray,
    config: PriceForecastConfig = PriceForecastConfig(),
) -> np.ndarray:
    """Forecast one or more target dates from one common information origin.

    This is the multi-day analogue of :func:`forecast_day_price`: the ridge
    coefficients are fitted once on rows ``< origin_index`` and then evaluated
    at every requested future date.  Consequently a three-day planning window
    cannot accidentally use the realised price of its first forecast day when
    predicting the second or third day.
    """
    values = _validate_prices(prices)
    origin_index = int(origin_index)
    targets = np.asarray(target_indices, dtype=int).reshape(-1)
    if targets.size == 0:
        return np.zeros((0, PERIODS), dtype=float)
    if not 0 <= origin_index <= values.shape[0]:
        raise ValueError("信息原点索引超出价格历史范围")
    if np.any(targets < origin_index) or np.any(targets >= values.shape[0] + 3):
        raise ValueError("目标日期必须不早于信息原点")
    base = np.asarray(base_curve, dtype=float)
    if base.shape != (PERIODS,) or not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("价格冷启动曲线必须是长度144的有限非负向量")
    if origin_index == 0:
        return np.repeat(base[None, :], len(targets), axis=0)
    history = values[:origin_index]
    recent = history[-min(len(history), max(1, int(config.recent_window))):].mean(axis=0)
    if len(history) < max(1, int(config.min_history)):
        weight = len(history) / float(max(1, int(config.min_history)))
        curve = np.maximum((1.0 - weight) * base + weight * recent, 0.0)
        return np.repeat(curve[None, :], len(targets), axis=0)
    x = _design(np.arange(origin_index), config)
    xtx = x.T @ x
    penalty = np.eye(x.shape[1], dtype=float) * float(config.ridge)
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(xtx + penalty, x.T @ history)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(xtx + penalty, x.T @ history, rcond=None)[0]
    prediction = _design(targets, config) @ coefficients
    prediction = np.vstack([
        _blend_same_weekday_price(
            values,
            int(target),
            prediction[pos],
            config,
            history_limit=origin_index,
        )
        for pos, target in enumerate(targets)
    ])
    return np.maximum(prediction, 0.0)


def _blend_same_weekday_price(
    values: np.ndarray,
    target_index: int,
    fourier_curve: np.ndarray,
    config: PriceForecastConfig,
    history_limit: int | None = None,
) -> np.ndarray:
    """Causally blend Fourier output with recent same-weekday curves.

    The attachment has a strong weekday-dependent tariff shape.  The blend
    only uses rows ``j < history_limit`` with ``j % 7 == target_index % 7``.
    For a one-day forecast ``history_limit`` equals ``target_index``; for a
    common-origin multi-day forecast it is the shared information boundary, so
    later target days cannot introduce their own newly revealed observations.
    Fourier remains the dominant component so the annual/weekly ridge model
    controls the cold and sparse-history regimes.
    """
    curve = np.asarray(fourier_curve, dtype=float)
    blend = float(config.weekday_blend)
    window = int(config.weekday_window)
    if curve.shape != (PERIODS,):
        raise ValueError("Fourier价格曲线必须是长度144")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("同星期价格混合权重必须位于[0,1]")
    if window <= 0 or target_index <= 0 or blend >= 1.0:
        return curve
    # A common-origin forecast must never use rows revealed after that origin.
    # For a one-day forecast the target itself is the information boundary;
    # for a multi-day forecast ``history_limit`` is the shared origin.
    limit = target_index if history_limit is None else int(history_limit)
    limit = min(max(limit, 0), len(values))
    history_indices = np.arange(limit, dtype=int)
    history_indices = history_indices[
        (history_indices % 7) == (int(target_index) % 7)
    ]
    if history_indices.size == 0:
        return curve
    selected = history_indices[-min(window, history_indices.size):]
    weekday_curve = np.asarray(values[selected].mean(axis=0), dtype=float)
    return blend * curve + (1.0 - blend) * weekday_curve


def build_causal_price_history(
    prices: np.ndarray,
    base_curve: np.ndarray,
    config: PriceForecastConfig = PriceForecastConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return one causal full-day forecast and history count per date."""
    values = _validate_prices(prices)
    forecasts = np.vstack(
        [forecast_day_price(values, i, base_curve, config) for i in range(values.shape[0])]
    )
    history_counts = np.arange(values.shape[0], dtype=int)
    return forecasts, history_counts


def build_price_error_history(
    actual_prices: np.ndarray,
    causal_forecasts: np.ndarray,
    current_index: int,
    min_history: int = 0,
) -> np.ndarray:
    """Return prior-date residual curves ``actual - forecast`` only.

    ``min_history`` can exclude cold-start forecasts from the error sample;
    this prevents the fallback curve used during the first few days from
    becoming an artificial long-run price-risk scenario.
    """
    actual = _validate_prices(actual_prices)
    forecasts = _validate_prices(causal_forecasts)
    if forecasts.shape != actual.shape:
        raise ValueError("实际价格和因果预测形状不一致")
    if not 0 <= int(current_index) <= actual.shape[0]:
        raise ValueError("当前日期索引超出价格历史范围")
    start = min(max(int(min_history), 0), int(current_index))
    return actual[start: int(current_index)] - forecasts[start: int(current_index)]


def reduce_error_scenarios(
    error_history: np.ndarray,
    max_count: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress residual curves by norm while preserving sample mass."""
    errors = np.asarray(error_history, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != PERIODS:
        raise ValueError(f"价格误差场景必须为(n,{PERIODS})矩阵")
    if errors.shape[0] == 0:
        return np.zeros((0, PERIODS), dtype=float), np.zeros(0, dtype=float)
    max_count = max(1, min(int(max_count), errors.shape[0]))
    if errors.shape[0] <= max_count:
        return errors.copy(), np.full(errors.shape[0], 1.0 / errors.shape[0])
    order = np.argsort(np.linalg.norm(errors, axis=1), kind="mergesort")
    groups = np.array_split(order, max_count)
    centres = np.vstack([errors[group].mean(axis=0) for group in groups])
    probabilities = np.asarray([len(group) / errors.shape[0] for group in groups], dtype=float)
    return centres, probabilities


def append_high_price_stress_scenario(
    scenarios: np.ndarray,
    probabilities: np.ndarray,
    error_history: np.ndarray,
    forecast_price: np.ndarray,
    config: PriceForecastConfig = PriceForecastConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Append a causal upper-price tail scenario for risk-aware planning."""
    curves = np.asarray(scenarios, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    history = np.asarray(error_history, dtype=float)
    forecast = np.asarray(forecast_price, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != PERIODS:
        raise ValueError("价格场景矩阵必须为(m,144)")
    if probs.shape != (curves.shape[0],) or np.any(probs < 0.0):
        raise ValueError("价格场景概率维度或取值错误")
    if history.ndim != 2 or history.shape[1] != PERIODS:
        raise ValueError("价格误差历史必须为(n,144)")
    if forecast.shape != (PERIODS,) or np.any(~np.isfinite(forecast)) or np.any(forecast < 0.0):
        raise ValueError("价格预测必须是长度144的有限非负向量")
    probability = float(config.stress_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("价格压力场景概率必须位于(0,1)")
    if history.shape[0]:
        q = np.quantile(history, float(config.stress_quantile), axis=0)
        stress_error = np.maximum(q, 0.0)
    else:
        stress_error = np.maximum(0.15 * forecast, 0.01)
    stress = np.maximum(forecast + stress_error, 0.0)
    if curves.shape[0] == 0:
        return stress.reshape(1, PERIODS), np.asarray([1.0])
    probs = probs / probs.sum()
    return np.vstack((curves, stress)), np.concatenate((probs * (1.0 - probability), [probability]))


def robust_price_curve(
    forecast_price: np.ndarray,
    error_history: np.ndarray,
    quantile: float = 0.90,
    weight: float = 1.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Return an upper-tail price curve for the planning objective.

    The additive one-sided quantile is a deterministic robust counterpart for
    price uncertainty.  It lets the existing PV-scenario LP account for both
    uncertainty sources without changing its decision variables: PV errors
    remain scenario based, while high-price residuals are priced at a causal
    upper-tail coefficient.  The historical residual matrix is always built
    from dates before the planning origin.
    """
    forecast = np.asarray(forecast_price, dtype=float)
    errors = np.asarray(error_history, dtype=float)
    if forecast.shape != (PERIODS,) or not np.isfinite(forecast).all() or np.any(forecast < 0.0):
        raise ValueError("价格预测必须是长度144的有限非负向量")
    if errors.ndim != 2 or errors.shape[1] != PERIODS:
        raise ValueError("价格误差历史必须为(n,144)矩阵")
    if not 0.0 <= float(quantile) <= 1.0 or float(weight) < 0.0:
        raise ValueError("价格风险分位数或权重非法")
    if errors.shape[0]:
        premium = np.maximum(np.quantile(errors, float(quantile), axis=0), 0.0)
    else:
        premium = np.maximum(0.15 * forecast, 0.01)
    return np.maximum(float(floor) + forecast + float(weight) * premium, 0.0)


def hourly_price_forecast_from_curve(curve: np.ndarray) -> np.ndarray:
    """Aggregate a 144-slot curve into 24 hourly mean prices."""
    values = np.asarray(curve, dtype=float)
    if values.shape != (PERIODS,):
        raise ValueError("日价格曲线必须为长度144")
    return values.reshape(HOURS, SLOTS_PER_HOUR).mean(axis=1)


def build_release_price_curve(
    actual_prices: np.ndarray,
    target_index: int,
    release_hour: int,
    full_day_forecast: np.ndarray,
    next_day_forecast: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a release-specific 10-minute curve and its 24-hour forecast.

    At 06:00/12:00/18:00 the prefix ending at the release instant is already
    observed.  A residual level correction from that prefix is applied to the
    still-unobserved hourly forecast.  The correction uses no future target-day
    prices.  The returned hourly vector is a 24-value forecast starting at the
    release time.  When ``next_day_forecast`` is supplied, the tail after
    midnight is filled with the next day's causal forecast instead of repeating
    the final hour of the current day.
    """
    actual = _validate_prices(actual_prices)
    forecast = np.asarray(full_day_forecast, dtype=float)
    if forecast.shape != (PERIODS,) or np.any(~np.isfinite(forecast)) or np.any(forecast < 0.0):
        raise ValueError("完整日价格预测必须为长度144的有限非负向量")
    if next_day_forecast is not None:
        next_day_forecast = np.asarray(next_day_forecast, dtype=float)
        if (
            next_day_forecast.shape != (PERIODS,)
            or np.any(~np.isfinite(next_day_forecast))
            or np.any(next_day_forecast < 0.0)
        ):
            raise ValueError("下一日价格预测必须为长度144的有限非负向量")
    if not 0 <= int(target_index) < actual.shape[0]:
        raise ValueError("目标日期索引超出价格范围")
    release_hour = int(release_hour)
    start = release_first_index(release_hour)
    lead_hours = release_lead_hours(release_hour)
    curve = forecast.copy()
    if start > 0:
        observed_prefix = actual[target_index, :start]
        base_prefix = forecast[:start]
        shift = float(np.mean(observed_prefix - base_prefix)) if observed_prefix.size else 0.0
        curve[start:] = np.maximum(curve[start:] + shift, 0.0)
        curve[:start] = actual[target_index, :start]
    # Match Q3's left-endpoint convention: the first block after each release
    # contains six slots beginning at the release timestamp.  The interval
    # ending exactly at the release instant is already observed and remains in
    # the prefix.
    hourly = np.zeros(HOURS, dtype=float)
    for h in range(lead_hours):
        slots = [
            t for t in range(start, PERIODS)
            if release_slot_hour_index(t, release_hour) == h
        ]
        if slots:
            hourly[h] = float(np.mean(curve[slots]))
    if lead_hours < HOURS:
        if next_day_forecast is None:
            fill = float(hourly[lead_hours - 1]) if lead_hours else float(curve[-1])
            hourly[lead_hours:] = fill
        else:
            next_hourly = hourly_price_forecast_from_curve(next_day_forecast)
            hourly[lead_hours:] = next_hourly[: HOURS - lead_hours]
    expanded = curve.copy()
    for t in range(start, PERIODS):
        h = release_slot_hour_index(t, release_hour)
        if h is not None:
            expanded[t] = hourly[h]
    return expanded, hourly


def build_release_price_error_history(
    actual_prices: np.ndarray,
    causal_forecasts: np.ndarray,
    current_index: int,
    release_hour: int,
    min_history: int = 0,
) -> np.ndarray:
    """Build same-release causal price residuals for a planning origin.

    Each historical date is evaluated with the forecast available at that
    date's 00:00 origin, then receives the same observed-prefix level
    correction used by :func:`build_release_price_curve`.  Residuals before
    ``release_hour`` are zeroed because those prices are already observed at
    the release and cannot affect the future risk premium.
    """
    actual = _validate_prices(actual_prices)
    forecasts = _validate_prices(causal_forecasts)
    if forecasts.shape != actual.shape:
        raise ValueError("实际价格和因果预测形状不一致")
    current_index = int(current_index)
    if not 0 <= current_index <= actual.shape[0]:
        raise ValueError("当前日期索引超出价格历史范围")
    release_hour = int(release_hour)
    if release_hour not in (0, 6, 12, 18):
        raise ValueError("价格发布时间必须为0、6、12或18时")
    first_future = release_first_index(release_hour)
    start = max(0, int(min_history))
    if start > current_index:
        return np.zeros((0, PERIODS), dtype=float)
    residuals: list[np.ndarray] = []
    for index in range(start, current_index):
        release_curve, _ = build_release_price_curve(
            actual, index, release_hour, forecasts[index]
        )
        residual = actual[index] - release_curve
        residual[:first_future] = 0.0
        residuals.append(residual)
    if not residuals:
        return np.zeros((0, PERIODS), dtype=float)
    return np.vstack(residuals)


def correlation_report(
    prices: np.ndarray,
    pv_kwh: np.ndarray,
    dates: list[date] | None = None,
) -> dict[str, float | int | str]:
    """Report raw and progressively de-seasonalised PV-price correlations.

    The raw statistic is dominated by the common day/night shape.  The first
    residual removes the two-way slot/day effects.  The second additionally
    removes weekday-by-slot means, and the third removes month/weekday-by-slot
    means.  The last pair is the conservative statistic used by the
    independence rule in the Q4 model.
    """
    price = _validate_prices(prices)
    pv = np.asarray(pv_kwh, dtype=float)
    if pv.shape != price.shape or not np.isfinite(pv).all():
        raise ValueError("光伏与价格矩阵必须同为有限的365×144形状")
    if dates is not None:
        if len(dates) != price.shape[0]:
            raise ValueError("相关性分析日期轴长度与价格历史不一致")
        date_axis = list(dates)
        if any(not isinstance(value, date) for value in date_axis):
            raise ValueError("相关性分析日期轴必须由date对象组成")
    else:
        # A relative weekday axis is still useful for direct unit tests when
        # the caller has no date metadata.  Production calls pass real dates.
        date_axis = None

    def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        x = x.reshape(-1)
        y = y.reshape(-1)
        if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return 0.0, 0.0
        pearson = float(np.corrcoef(x, y)[0, 1])
        ranks_x = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
        ranks_y = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(float)
        spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        return pearson, spearman

    def group_slot_residual(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
        residual = np.empty_like(values, dtype=float)
        for group in np.unique(groups):
            mask = groups == group
            residual[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True)
        return residual

    raw_p, raw_s = corr(price, pv)
    price_slot = price - price.mean(axis=0, keepdims=True)
    pv_slot = pv - pv.mean(axis=0, keepdims=True)
    price_residual = price_slot - price_slot.mean(axis=1, keepdims=True)
    pv_residual = pv_slot - pv_slot.mean(axis=1, keepdims=True)
    daily_p, daily_s = corr(price_residual, pv_residual)

    if date_axis is None:
        weekday = np.arange(price.shape[0], dtype=int) % 7
        month_weekday = weekday
    else:
        weekday = np.asarray([value.weekday() for value in date_axis], dtype=int)
        month_weekday = np.asarray(
            [
                f"{value.year:04d}-{value.month:02d}-{value.weekday()}"
                for value in date_axis
            ],
            dtype=object,
        )
    price_weekday = group_slot_residual(price, weekday)
    pv_weekday = group_slot_residual(pv, weekday)
    price_weekday -= price_weekday.mean(axis=1, keepdims=True)
    pv_weekday -= pv_weekday.mean(axis=1, keepdims=True)
    weekday_p, weekday_s = corr(price_weekday, pv_weekday)

    # Each calendar month has enough samples for a stable weekday baseline
    # while keeping seasonal level differences out of the residual statistic.
    price_seasonal = group_slot_residual(price, month_weekday)
    pv_seasonal = group_slot_residual(pv, month_weekday)
    price_seasonal -= price_seasonal.mean(axis=1, keepdims=True)
    pv_seasonal -= pv_seasonal.mean(axis=1, keepdims=True)
    seasonal_p, seasonal_s = corr(price_seasonal, pv_seasonal)
    return {
        "sample_days": int(price.shape[0]),
        "sample_slots": int(price.size),
        "raw_pearson": raw_p,
        "raw_spearman": raw_s,
        "deseasonalized_pearson": daily_p,
        "deseasonalized_spearman": daily_s,
        "weekday_deseasonalized_pearson": weekday_p,
        "weekday_deseasonalized_spearman": weekday_s,
        "seasonally_deseasonalized_pearson": seasonal_p,
        "seasonally_deseasonalized_spearman": seasonal_s,
        "independence_rule": (
            "仅在去月份/星期/时段结构后的Pearson与Spearman绝对值均小于0.10时采用独立场景"
        ),
    }


def date_index_map(dates: list[date]) -> dict[date, int]:
    """Validate a continuous date axis and return its index map."""
    if not dates:
        raise ValueError("日期轴不能为空")
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days != 1:
            raise ValueError("价格日期必须连续")
    return {d: i for i, d in enumerate(dates)}
