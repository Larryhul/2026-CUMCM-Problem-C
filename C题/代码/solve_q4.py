"""Problem 4: causal fluctuating-price versions of Problems 2 and 3.

Attachment 4 supplies the realised tariff curves used for back-testing.  Under
the selected real-time information setting, the controller does not know a
target day's future tariff at 00:00: it uses a causal Fourier-ridge point
forecast, paired PV-price scenarios, and scenario CVaR for planning, while
realised prices are used only for currently observed intervals and ex-post
settlement.  Problem 4.2 uses the daily 00:00 plan; Problem 4.3 updates the
price forecast at the same four releases as the photovoltaic forecast.
"""

from __future__ import annotations

import json
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_price_q4 as price_forecast
import solve_q2
import solve_q3
from data_processing import load_attachment2
from data_processing import load_processed_attachment1
from data_processing import validate_time_axis


PERIODS = 144
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
INITIAL_SOC_KWH = 6000.0
ETA = 0.90
PRICE_CONFIG = price_forecast.PriceForecastConfig()
# Calibrated on a strictly causal 2025 back-test.  The aliases below retain
# the 00:00 values used by Q4.2 and by older audit scripts.
PRICE_RISK_QUANTILE_BY_RELEASE = {
    0: 0.925,
    6: 0.900,
    12: 0.925,
    18: 0.900,
}
PRICE_RISK_WEIGHT_BY_RELEASE = {
    0: 1.00,
    6: 1.25,
    12: 1.00,
    18: 1.00,
}
PRICE_RISK_QUANTILE = PRICE_RISK_QUANTILE_BY_RELEASE[0]
PRICE_RISK_WEIGHT = PRICE_RISK_WEIGHT_BY_RELEASE[0]
JOINT_SCENARIO_COUNT = 8
JOINT_STRESS_PROBABILITY = solve_q2.CVAR_STRESS_PROBABILITY


def joint_scenario_hash(
    scenario_pv: np.ndarray,
    scenario_price: np.ndarray,
) -> str:
    """Return a stable digest for a paired PV/price scenario matrix."""
    pv = np.asarray(scenario_pv, dtype=np.float64)
    price = np.asarray(scenario_price, dtype=np.float64)
    if pv.ndim != 2 or price.shape != pv.shape:
        raise ValueError("联合场景哈希输入必须为同形状二维矩阵")
    return hashlib.sha256(
        np.ascontiguousarray(np.hstack((pv, price)), dtype=np.float64).tobytes()
    ).hexdigest()


def scenario_hash(values: np.ndarray) -> str:
    """Return a stable digest for a marginal scenario matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("边际场景哈希输入必须为有限二维矩阵")
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.float64).tobytes()
    ).hexdigest()


def load_price_history(path: Path) -> tuple[list[date], np.ndarray]:
    """Read and validate attachment 4's daily 10-minute price curves."""
    frame = pd.read_excel(path)
    if frame.shape != (365, 145):
        raise ValueError(f"附件4应为365行、145列，实际为{frame.shape}")
    validate_time_axis(list(frame.columns[1:]), "附件4")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    if dates.isna().any():
        raise ValueError("附件4日期存在无法解析的记录")
    parsed = [x.date() for x in dates]
    expected = [parsed[0] + pd.Timedelta(days=i) for i in range(len(parsed))]
    expected = [x.date() if hasattr(x, "date") else x for x in expected]
    if parsed != expected:
        raise ValueError("附件4日期必须连续、升序且不重复")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.shape != (365, PERIODS) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("附件4价格必须是365×144的有限非负矩阵")
    return parsed, values


def load_inputs(root: Path):
    a2 = load_attachment2(root / "附件" / "附件2.xlsx")
    price_dates, prices = load_price_history(root / "附件" / "附件4.xlsx")
    forecasts = solve_q3.parse_attachment3(root / "附件" / "附件3.xlsx")
    if list(a2.dates) != price_dates:
        raise ValueError("附件2与附件4的日期轴不一致")
    return price_dates, prices, a2.load_kwh, a2.pv_actual_kwh, forecasts


def build_price_context(
    root: Path,
    prices: np.ndarray,
    dates: list[date] | None = None,
) -> dict:
    """Build the causal price history used by both Q4 subproblems."""
    base = load_processed_attachment1(
        root / "数据" / "processed_attachment1.csv"
    )["price"].to_numpy(dtype=float)
    causal, history_counts = price_forecast.build_causal_price_history(
        prices, base, PRICE_CONFIG
    )
    a2 = load_attachment2(root / "附件" / "附件2.xlsx")
    correlation = price_forecast.correlation_report(
        prices,
        a2.pv_actual_kwh,
        dates=dates,
    )
    return {
        "base_curve": base,
        "causal_forecasts": causal,
        "history_counts": history_counts,
        "correlation": correlation,
    }


def price_plan_horizon(
    prices: np.ndarray,
    causal_forecasts: np.ndarray,
    day_index: int,
    horizon_days: int,
    base_curve: np.ndarray,
    pv_error_history: np.ndarray | None = None,
    price_error_history: np.ndarray | None = None,
    conditional_price_error_history: np.ndarray | None = None,
    risk_quantile: float | None = 0.0,
    risk_weight: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a common-origin price forecast and its LP price coefficient.

    ``risk_quantile=0`` and ``risk_weight=0`` make the coefficient equal to the
    causal point forecast.  That is the default formal setting; passing
    ``None`` explicitly selects the calibrated positive values for the
    price-premium ablation.
    """
    quantile = PRICE_RISK_QUANTILE if risk_quantile is None else float(risk_quantile)
    weight = PRICE_RISK_WEIGHT if risk_weight is None else float(risk_weight)
    target_indices = np.arange(day_index, day_index + horizon_days, dtype=int)
    point = price_forecast.forecast_from_origin(
        prices, day_index, target_indices, base_curve, PRICE_CONFIG
    )
    errors = price_forecast.build_price_error_history(
        prices, causal_forecasts, day_index, min_history=PRICE_CONFIG.min_history
    )
    conditional_errors_by_day = [errors for _ in range(horizon_days)]
    if pv_error_history is not None:
        pv_errors = np.asarray(pv_error_history, dtype=float)
        candidate_price_errors = None
        candidate_source = (
            conditional_price_error_history
            if conditional_price_error_history is not None
            else price_error_history
        )
        if candidate_source is not None:
            candidate = np.asarray(candidate_source, dtype=float)
            if (
                candidate.ndim == 2
                and candidate.shape[1] == horizon_days * PERIODS
                and candidate.shape[0] == pv_errors.shape[0]
            ):
                candidate_price_errors = candidate
        if (
            pv_errors.ndim == 2
            and candidate_price_errors is not None
            and pv_errors.shape[0] == candidate_price_errors.shape[0]
            and pv_errors.shape[1] == horizon_days * PERIODS
            and pv_errors.shape[0] > 0
        ):
            # The residual correlation audit shows a strong PV-price relation
            # in this data set.  For Q4.2, use price residuals from historical
            # blocks with the largest PV shortfall instead of assuming the two
            # errors are independent.
            scores = np.sum(np.maximum(-pv_errors, 0.0), axis=1)
            keep = max(1, int(np.ceil(0.30 * len(scores))))
            selected = np.argsort(scores, kind="mergesort")[-keep:]
            conditional_errors_by_day = [
                candidate_price_errors[
                    selected,
                    day * PERIODS : (day + 1) * PERIODS,
                ]
                for day in range(horizon_days)
            ]
    robust = np.vstack([
        price_forecast.robust_price_curve(
            curve,
            conditional_errors_by_day[day],
            quantile=quantile,
            weight=weight,
        )
        for day, curve in enumerate(point)
    ])
    reduced, probabilities = price_forecast.reduce_error_scenarios(
        errors, max_count=PRICE_CONFIG.error_scenario_count
    )
    reduced, probabilities = price_forecast.append_high_price_stress_scenario(
        reduced, probabilities, errors, point[0], PRICE_CONFIG
    )
    return point, robust, reduced, probabilities


def build_multiday_price_error_history(
    prices: np.ndarray,
    causal_forecasts: np.ndarray,
    origin_index: int,
    horizon_days: int,
    base_curve: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return lead-aware price error blocks aligned with Q2 PV blocks.

    Every historical block is forecast from its own common origin.  In
    particular, the forecast for day ``d+1`` is not replaced by the separate
    one-day forecast fitted after observing day ``d``.  This prevents a
    three-day price block from using information that was unavailable when
    the historical contract would have been issued.
    """
    prices = np.asarray(prices, dtype=float)
    forecasts = np.asarray(causal_forecasts, dtype=float)
    if prices.shape != forecasts.shape or prices.ndim != 2 or prices.shape[1] != PERIODS:
        raise ValueError("价格和因果预测必须为同形状(n,144)矩阵")
    if base_curve is None:
        if len(forecasts) == 0:
            raise ValueError("价格历史为空时必须提供冷启动曲线")
        base_curve = forecasts[0]
    base_curve = np.asarray(base_curve, dtype=float)
    if base_curve.shape != (PERIODS,) or not np.isfinite(base_curve).all() or np.any(base_curve < 0.0):
        raise ValueError("价格冷启动曲线必须为长度144的有限非负向量")
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (len(prices),):
            raise ValueError("价格误差有效标记长度错误")
    blocks = []
    latest_start = int(origin_index) - int(horizon_days)
    for historical_origin in range(max(latest_start + 1, 0)):
        end = historical_origin + horizon_days
        if end > len(prices):
            continue
        if valid_mask is None:
            if historical_origin < PRICE_CONFIG.min_history:
                continue
        elif not valid_mask[historical_origin]:
            continue
        targets = np.arange(historical_origin, end, dtype=int)
        point = price_forecast.forecast_from_origin(
            prices,
            historical_origin,
            targets,
            base_curve,
            PRICE_CONFIG,
        )
        blocks.append((prices[historical_origin:end] - point).reshape(-1))
    if not blocks:
        return np.zeros((0, horizon_days * PERIODS), dtype=float)
    return np.vstack(blocks)


def build_price_valid_mask(prices: np.ndarray) -> np.ndarray:
    """Return the price-layer history mask used for causal error blocks.

    The photovoltaic forecast history has a longer warm-up period than the
    price model.  Keeping this mask separate prevents the Q2 PV validity flag
    from silently discarding otherwise usable price residual blocks.
    """
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or values.shape[1] != PERIODS:
        raise ValueError("价格历史必须为(n,144)矩阵")
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[PRICE_CONFIG.min_history:] = True
    return mask


def build_joint_multiday_error_history(
    prices: np.ndarray,
    causal_price_forecasts: np.ndarray,
    actual_pv: np.ndarray,
    multiday_pv_forecasts: np.ndarray,
    pv_valid_mask: np.ndarray,
    price_valid_mask: np.ndarray,
    origin_index: int,
    horizon_days: int,
    base_price_curve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build paired PV/price residual blocks from the same historical dates.

    A row in each returned matrix represents one historical planning origin.
    Keeping the rows paired is what allows a joint scenario to preserve the
    observed dependence between photovoltaic shortfall and price shocks.
    """
    prices = np.asarray(prices, dtype=float)
    causal_price_forecasts = np.asarray(causal_price_forecasts, dtype=float)
    actual_pv = np.asarray(actual_pv, dtype=float)
    multiday_pv_forecasts = np.asarray(multiday_pv_forecasts, dtype=float)
    pv_valid_mask = np.asarray(pv_valid_mask, dtype=bool)
    price_valid_mask = np.asarray(price_valid_mask, dtype=bool)
    if (
        prices.shape != causal_price_forecasts.shape
        or prices.ndim != 2
        or prices.shape[1] != PERIODS
        or actual_pv.shape != prices.shape
        or multiday_pv_forecasts.ndim != 3
        or multiday_pv_forecasts.shape[0] != len(prices)
        or multiday_pv_forecasts.shape[1] < horizon_days
        or multiday_pv_forecasts.shape[2] != PERIODS
        or pv_valid_mask.shape != (len(prices),)
        or price_valid_mask.shape != (len(prices),)
    ):
        raise ValueError("联合多日误差历史维度错误")
    if not 1 <= horizon_days <= multiday_pv_forecasts.shape[1]:
        raise ValueError("联合多日误差历史窗口长度错误")
    pv_blocks: list[np.ndarray] = []
    price_blocks: list[np.ndarray] = []
    latest_start = int(origin_index) - int(horizon_days)
    for historical_origin in range(max(latest_start + 1, 0)):
        end = historical_origin + horizon_days
        if end > len(prices):
            continue
        if not (pv_valid_mask[historical_origin] and price_valid_mask[historical_origin]):
            continue
        pv_forecast_block = multiday_pv_forecasts[historical_origin, :horizon_days]
        if not np.isfinite(pv_forecast_block).all():
            continue
        target_indices = np.arange(historical_origin, end, dtype=int)
        price_forecast_block = price_forecast.forecast_from_origin(
            prices,
            historical_origin,
            target_indices,
            base_price_curve,
            PRICE_CONFIG,
        )
        pv_blocks.append(
            (actual_pv[historical_origin:end] - pv_forecast_block).reshape(-1)
        )
        price_blocks.append(
            (prices[historical_origin:end] - price_forecast_block).reshape(-1)
        )
    if not pv_blocks:
        width = horizon_days * PERIODS
        return np.zeros((0, width), dtype=float), np.zeros((0, width), dtype=float)
    return np.vstack(pv_blocks), np.vstack(price_blocks)


def build_joint_release_error_history(
    current_index: int,
    release: int,
    dates: list[date],
    actual_pv: np.ndarray,
    forecasts: dict[date, dict[int, np.ndarray]],
    margins: dict[date, dict[int, np.ndarray]] | None,
    prices: np.ndarray,
    causal_price_forecasts: np.ndarray,
    base_price_curve: np.ndarray,
    min_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build same-release paired PV and price residual histories."""
    start = price_forecast.release_first_index(release)
    pv_errors: list[np.ndarray] = []
    price_errors: list[np.ndarray] = []
    for historical_index in range(max(int(min_history), 0), int(current_index)):
        historical_date = dates[historical_index]
        historical_margins = None if release == 0 else (
            margins[historical_date] if margins is not None else None
        )
        pv_curve = solve_q3.build_release_curves(
            forecasts[historical_date], historical_margins
        )[release]
        price_curve, _ = price_forecast.build_release_price_curve(
            prices,
            historical_index,
            release,
            causal_price_forecasts[historical_index],
        )
        pv_residual = np.asarray(actual_pv[historical_index] - pv_curve, dtype=float)
        price_residual = np.asarray(prices[historical_index] - price_curve, dtype=float)
        pv_residual[:start] = 0.0
        price_residual[:start] = 0.0
        pv_errors.append(pv_residual)
        price_errors.append(price_residual)
    if not pv_errors:
        return np.zeros((0, PERIODS), dtype=float), np.zeros((0, PERIODS), dtype=float)
    return np.vstack(pv_errors), np.vstack(price_errors)


def reduce_joint_error_scenarios(
    pv_errors: np.ndarray,
    price_errors: np.ndarray,
    max_count: int = JOINT_SCENARIO_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress paired residuals with one deterministic grouping order."""
    pv = np.asarray(pv_errors, dtype=float)
    price = np.asarray(price_errors, dtype=float)
    if pv.ndim != 2 or price.ndim != 2 or pv.shape != price.shape:
        raise ValueError("联合误差场景必须为同形状矩阵")
    if pv.shape[1] == 0 or pv.shape[1] % PERIODS:
        raise ValueError("联合误差场景宽度必须为144的整数倍")
    if pv.shape[0] == 0:
        return (
            np.zeros((0, pv.shape[1]), dtype=float),
            np.zeros((0, pv.shape[1]), dtype=float),
            np.zeros(0, dtype=float),
        )
    pv_norm = np.linalg.norm(pv, axis=1)
    price_norm = np.linalg.norm(price, axis=1)
    pv_scale = float(np.median(pv_norm) + 1e-9)
    price_scale = float(np.median(price_norm) + 1e-9)
    score = pv_norm / pv_scale + price_norm / price_scale
    max_count = max(1, min(int(max_count), pv.shape[0]))
    order = np.argsort(score, kind="mergesort")
    groups = np.array_split(order, max_count)
    pv_centres = np.vstack([pv[group].mean(axis=0) for group in groups])
    price_centres = np.vstack([price[group].mean(axis=0) for group in groups])
    probabilities = np.asarray(
        [len(group) / pv.shape[0] for group in groups], dtype=float
    )
    return pv_centres, price_centres, probabilities


def append_joint_stress_scenario(
    pv_scenarios: np.ndarray,
    price_scenarios: np.ndarray,
    probabilities: np.ndarray,
    pv_errors: np.ndarray,
    price_errors: np.ndarray,
    forecast_pv: np.ndarray,
    forecast_price: np.ndarray,
    stress_probability: float = JOINT_STRESS_PROBABILITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append a paired low-PV/high-price tail scenario."""
    pv_scenarios = np.asarray(pv_scenarios, dtype=float)
    price_scenarios = np.asarray(price_scenarios, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    pv_errors = np.asarray(pv_errors, dtype=float)
    price_errors = np.asarray(price_errors, dtype=float)
    forecast_pv = np.asarray(forecast_pv, dtype=float).reshape(-1)
    forecast_price = np.asarray(forecast_price, dtype=float).reshape(-1)
    if pv_errors.shape != price_errors.shape or pv_errors.ndim != 2:
        raise ValueError("联合压力场景误差维度错误")
    if forecast_pv.size != forecast_price.size or pv_errors.shape[1] != forecast_pv.size:
        raise ValueError("联合压力场景预测宽度错误")
    if pv_scenarios.ndim != 2 or price_scenarios.shape != pv_scenarios.shape:
        raise ValueError("联合压力场景矩阵维度错误")
    if probabilities.shape != (pv_scenarios.shape[0],):
        raise ValueError("联合压力场景概率维度错误")
    if not 0.0 < float(stress_probability) < 1.0:
        raise ValueError("联合压力场景概率必须位于(0,1)")
    if pv_errors.shape[0]:
        shortage = np.sum(np.maximum(-pv_errors, 0.0), axis=1)
        high_price = np.sum(np.maximum(price_errors, 0.0), axis=1)
        score = shortage / (np.median(shortage) + 1e-9) + high_price / (
            np.median(high_price) + 1e-9
        )
        count = max(1, int(np.ceil(0.10 * len(score))))
        tail = np.argsort(score, kind="mergesort")[-count:]
        pv_stress_error = pv_errors[tail].mean(axis=0)
        price_stress_error = price_errors[tail].mean(axis=0)
        price_stress_error = np.maximum(
            price_stress_error,
            np.maximum(np.quantile(price_errors, 0.90, axis=0), 0.0),
        )
    else:
        pv_stress_error = -0.30 * forecast_pv
        price_stress_error = np.maximum(0.15 * forecast_price, 0.01)
    pv_stress = np.maximum(forecast_pv + pv_stress_error, 0.0)
    price_stress = np.maximum(forecast_price + price_stress_error, 0.0)
    if len(probabilities):
        probabilities = probabilities / probabilities.sum()
        probabilities *= 1.0 - float(stress_probability)
    else:
        # With no historical blocks the fallback stress path is the only
        # available scenario, so it carries the full probability mass.
        return (
            np.asarray([pv_stress], dtype=float),
            np.asarray([price_stress], dtype=float),
            np.asarray([1.0], dtype=float),
        )
    return (
        np.vstack((pv_scenarios, pv_stress)),
        np.vstack((price_scenarios, price_stress)),
        np.concatenate((probabilities, [float(stress_probability)])),
    )


def build_joint_scenario_curves(
    forecast_pv: np.ndarray,
    forecast_price: np.ndarray,
    pv_errors: np.ndarray,
    price_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce paired residuals and return PV/price scenario curves."""
    pv_centres, price_centres, probabilities = reduce_joint_error_scenarios(
        pv_errors, price_errors, max_count=JOINT_SCENARIO_COUNT
    )
    pv_curves, price_curves, probabilities = append_joint_stress_scenario(
        np.maximum(np.asarray(forecast_pv, dtype=float).reshape(1, -1) + pv_centres, 0.0),
        np.maximum(np.asarray(forecast_price, dtype=float).reshape(1, -1) + price_centres, 0.0),
        probabilities,
        pv_errors,
        price_errors,
        forecast_pv,
        forecast_price,
    )
    return pv_curves, price_curves, probabilities


def build_pv_only_scenario_curves(
    forecast_pv: np.ndarray,
    reference_price: np.ndarray,
    pv_errors: np.ndarray,
    *,
    price_scenario_mode: str = "constant_reference_price",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build PV-risk scenarios while removing PV-price dependence.

    The ablation keeps exactly the same historical PV reduction and stress
    scenario as the production model, but assigns one common reference price
    curve to every PV path.  Thus the LP still sees PV uncertainty and CVaR,
    while price shocks cannot be paired with the PV paths.
    """
    if price_scenario_mode != "constant_reference_price":
        raise ValueError("未知的边际PV场景价格模式")
    forecast = np.asarray(forecast_pv, dtype=float).reshape(-1)
    reference = np.asarray(reference_price, dtype=float).reshape(-1)
    errors = np.asarray(pv_errors, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != forecast.size:
        raise ValueError("边际PV场景误差与预测宽度不一致")
    if reference.shape != forecast.shape or not np.isfinite(reference).all() or np.any(reference < 0.0):
        raise ValueError("边际PV场景参考电价维度或取值错误")
    reduced, probabilities = solve_q2.reduce_error_scenarios(errors)
    reduced, probabilities = solve_q2.append_pessimistic_error_scenario(
        reduced,
        probabilities,
        errors,
        forecast,
        reference,
    )
    pv_curves = np.maximum(forecast[None, :] + reduced, 0.0)
    price_curves = np.repeat(reference[None, :], len(probabilities), axis=0)
    return pv_curves, price_curves, probabilities


def release_price_context(
    prices: np.ndarray,
    causal_forecasts: np.ndarray,
    day_index: int,
    release: int,
    base_curve: np.ndarray,
    pv_error_history: np.ndarray | None = None,
    risk_quantile: float | None = 0.0,
    risk_weight: float | None = 0.0,
) -> dict:
    """Construct one causal Q3 price release at 1-hour information grain.

    The optional upper-tail premium is controlled explicitly by the two risk
    arguments so audits and ablations can reproduce the selected price mode.
    """
    quantile = (
        PRICE_RISK_QUANTILE_BY_RELEASE[int(release)]
        if risk_quantile is None
        else float(risk_quantile)
    )
    weight = (
        PRICE_RISK_WEIGHT_BY_RELEASE[int(release)]
        if risk_weight is None
        else float(risk_weight)
    )
    full_points = price_forecast.forecast_from_origin(
        prices,
        day_index,
        [day_index, day_index + 1],
        base_curve,
        PRICE_CONFIG,
    )
    full_point = full_points[0]
    expanded, hourly = price_forecast.build_release_price_curve(
        prices,
        day_index,
        release,
        full_point,
        next_day_forecast=full_points[1],
    )
    errors = price_forecast.build_release_price_error_history(
        prices,
        causal_forecasts,
        day_index,
        release,
        min_history=PRICE_CONFIG.min_history,
    )
    conditional_errors = errors
    if pv_error_history is not None:
        pv_errors = np.asarray(pv_error_history, dtype=float)
        if (
            pv_errors.ndim == 2
            and pv_errors.shape == errors.shape
            and pv_errors.shape[0] > 0
        ):
            scores = np.sum(np.maximum(-pv_errors, 0.0), axis=1)
            keep = max(1, int(np.ceil(0.30 * len(scores))))
            selected = np.argsort(scores, kind="mergesort")[-keep:]
            conditional_errors = errors[selected]
    robust = price_forecast.robust_price_curve(
        expanded,
        conditional_errors,
        quantile=quantile,
        weight=weight,
    )
    # Slots before the release have already been observed and therefore must
    # not receive a hypothetical future-price risk premium.
    first_future = price_forecast.release_first_index(release)
    if first_future:
        robust[:first_future] = prices[day_index, :first_future]
    scenarios, probabilities = price_forecast.reduce_error_scenarios(
        errors, max_count=PRICE_CONFIG.error_scenario_count
    )
    scenarios, probabilities = price_forecast.append_high_price_stress_scenario(
        scenarios, probabilities, errors, expanded, PRICE_CONFIG
    )
    return {
        "point": expanded,
        "robust": robust,
        "hourly": hourly,
        "error_history": errors,
        "conditional_error_history": conditional_errors,
        "scenario_errors": scenarios,
        "scenario_probabilities": probabilities,
        "history_count": int(day_index),
        "valid_history_count": int(errors.shape[0]),
        "conditional_history_count": int(conditional_errors.shape[0]),
        "release_hour": int(release),
        "updated_from_index": int(price_forecast.release_first_index(release)),
    }


def build_release_pv_error_history(
    current_index: int,
    release: int,
    dates: list[date],
    actual_pv: np.ndarray,
    forecasts: dict[date, dict[int, np.ndarray]],
    margin_history: dict[date, dict[int, np.ndarray]] | None = None,
    min_history: int = 0,
) -> np.ndarray:
    """Build PV residuals aligned with the same-release price residuals."""
    start = price_forecast.release_first_index(release)
    residuals: list[np.ndarray] = []
    for historical_index in range(max(int(min_history), 0), int(current_index)):
        historical_margins = None if release == 0 else (
            margin_history[dates[historical_index]]
            if margin_history is not None
            else None
        )
        curve = solve_q3.build_release_curves(
            forecasts[dates[historical_index]],
            historical_margins,
        )[release]
        residual = np.asarray(actual_pv[historical_index] - curve, dtype=float)
        residual[:start] = 0.0
        residuals.append(residual)
    if not residuals:
        return np.zeros((0, PERIODS), dtype=float)
    return np.vstack(residuals)


def load_problem2_forecasts(root: Path, actual_pv: np.ndarray):
    """Build the same causal multi-day forecast history used by Problem 2."""
    cold_start = load_processed_attachment1(
        root / "数据" / "processed_attachment1.csv"
    )["pv_kwh"].to_numpy(dtype=float)
    selections_config = solve_q2.FORECAST_CONFIG
    forecasts, valid_mask, selections = solve_q2.build_causal_forecast_history(
        actual_pv, cold_start, selections_config
    )
    multiday = solve_q2.build_causal_multiday_forecast_history(
        actual_pv,
        cold_start,
        selections,
        solve_q2.PLANNING_HORIZON_DAYS,
        selections_config,
    )
    return forecasts, multiday, valid_mask, selections


def historical_release0_errors(
    current_date: date,
    dates: list[date],
    actual_pv: np.ndarray,
    forecasts: dict[date, dict[int, np.ndarray]],
) -> np.ndarray:
    """Build causal release-0 forecast errors for Problem 4.2 scenarios."""
    index = {d: i for i, d in enumerate(dates)}
    errors: list[np.ndarray] = []
    for d in dates:
        if d >= current_date:
            break
        base = solve_q3.build_release_curves(forecasts[d])[0]
        errors.append(actual_pv[index[d]] - base)
    if not errors:
        return np.zeros((0, PERIODS), dtype=float)
    return np.vstack(errors)


def solve_problem42_day(
    price_horizon: np.ndarray,
    load_horizon: np.ndarray,
    actual_pv: np.ndarray,
    forecast_horizon: np.ndarray,
    initial_soc: float,
    error_history: np.ndarray,
    decision_price_horizon: np.ndarray | None = None,
    point_price_horizon: np.ndarray | None = None,
    scenario_pv_horizon: np.ndarray | None = None,
    scenario_price_horizon: np.ndarray | None = None,
    scenario_probabilities: np.ndarray | None = None,
) -> dict:
    """Solve one fluctuating-price Problem 2 rolling window.

    The LP plans over up to three days, but only the first day's purchase
    vector is committed.  The execution layer uses the causal decision-price
    forecast for future coefficients and the realised price only for the
    current interval and ex-post cost calculation.
    """
    # ``price_horizon`` is the realised tariff used for ex-post accounting.
    # ``decision_price_horizon`` is the causal point/decision forecast used by
    # the planning and MPC layers; an upper-tail premium is optional and is
    # enabled only by the dedicated ablation.
    realised_price_horizon = np.asarray(price_horizon, dtype=float)
    decision_price_horizon = (
        realised_price_horizon.copy()
        if decision_price_horizon is None
        else np.asarray(decision_price_horizon, dtype=float)
    )
    point_price_horizon = (
        decision_price_horizon.copy()
        if point_price_horizon is None
        else np.asarray(point_price_horizon, dtype=float)
    )
    load_horizon = np.asarray(load_horizon, dtype=float)
    forecast_horizon = np.asarray(forecast_horizon, dtype=float)
    if (
        realised_price_horizon.shape != load_horizon.shape
        or decision_price_horizon.shape != load_horizon.shape
        or point_price_horizon.shape != load_horizon.shape
        or np.any(realised_price_horizon < 0.0)
        or np.any(decision_price_horizon < 0.0)
        or np.any(point_price_horizon < 0.0)
    ):
        raise ValueError("问题四2的实际/决策/点预测价格形状不一致")
    horizon_days = load_horizon.shape[0]
    if scenario_pv_horizon is None:
        scenario_errors, probabilities = solve_q2.reduce_error_scenarios(error_history)
        scenario_errors, probabilities = solve_q2.append_pessimistic_error_scenario(
            scenario_errors,
            probabilities,
            error_history,
            forecast_horizon.reshape(-1),
            decision_price_horizon,
        )
        scenario_pv = np.maximum(
            forecast_horizon.reshape(1, -1) + scenario_errors, 0.0
        ).reshape(-1, horizon_days, PERIODS)
    else:
        scenario_pv = np.asarray(scenario_pv_horizon, dtype=float)
        if scenario_pv.ndim == 2 and scenario_pv.shape[1] == horizon_days * PERIODS:
            scenario_pv = scenario_pv.reshape(-1, horizon_days, PERIODS)
        if scenario_pv.ndim != 3 or scenario_pv.shape[1:] != (horizon_days, PERIODS):
            raise ValueError("问题四2联合场景光伏必须为(m,H,144)矩阵")
        if not np.isfinite(scenario_pv).all() or np.any(scenario_pv < 0.0):
            raise ValueError("问题四2联合场景光伏必须为有限非负数")
        probabilities = (
            np.full(scenario_pv.shape[0], 1.0 / scenario_pv.shape[0])
            if scenario_probabilities is None
            else np.asarray(scenario_probabilities, dtype=float)
        )
        if probabilities.shape != (scenario_pv.shape[0],) or np.any(probabilities < 0.0):
            raise ValueError("问题四2联合场景概率维度或取值错误")
        if probabilities.sum() <= 0.0:
            raise ValueError("问题四2联合场景概率和必须为正")
        probabilities = probabilities / probabilities.sum()
    if scenario_price_horizon is not None:
        scenario_price = np.asarray(scenario_price_horizon, dtype=float)
        if scenario_price.ndim == 2 and scenario_price.shape[1] == horizon_days * PERIODS:
            scenario_price = scenario_price.reshape(-1, horizon_days, PERIODS)
        if scenario_price.ndim != 3 or scenario_price.shape != scenario_pv.shape:
            raise ValueError("问题四2联合场景价格必须与场景光伏同形状")
        if not np.isfinite(scenario_price).all() or np.any(scenario_price < 0.0):
            raise ValueError("问题四2联合场景价格必须为有限非负数")
    else:
        scenario_price = None
    terminal_penalty = float(np.mean(decision_price_horizon) / ETA)
    plan = solve_q2.solve_multiday_risk_aware_plan(
        decision_price_horizon,
        load_horizon,
        forecast_horizon,
        scenario_pv,
        probabilities,
        initial_soc,
        terminal_penalty,
        scenario_price_horizon_kwh=scenario_price,
    )
    terminal_target = float(
        np.clip(
            plan["expected_first_day_terminal_soc_kwh"],
            solve_q2.BATTERY_MIN_KWH,
            solve_q2.BATTERY_MAX_KWH,
        )
    )
    def _first_day_reference(key: str) -> np.ndarray | None:
        raw = plan.get(key)
        if raw is None:
            return None
        vector = np.asarray(raw, dtype=float).reshape(-1)
        if vector.size < PERIODS:
            raise ValueError(f"{key}长度不足144: {vector.size}")
        vector = vector[:PERIODS]
        if not np.isfinite(vector).all():
            raise ValueError(f"{key}包含非有限值")
        # HiGHS can return tiny negative roundoff at nonnegative bounds.
        return np.maximum(vector, 0.0)

    actual = solve_q2.solve_causal_recourse(
        decision_price_horizon[0],
        load_horizon[0],
        forecast_horizon[0],
        actual_pv,
        np.asarray(plan["purchase_kwh"], dtype=float).reshape(-1)[:PERIODS],
        initial_soc,
        scenario_pv[:, 0, :],
        probabilities,
        terminal_target,
        terminal_penalty,
        objective_price_kwh=decision_price_horizon[0],
        settlement_price_kwh=realised_price_horizon[0],
        observed_price_kwh=realised_price_horizon[0],
        scenario_price_kwh=(
            scenario_price[:, 0, :]
            if scenario_price is not None
            else None
        ),
    )
    planned_cost_forecast = float(np.dot(point_price_horizon[0], plan["purchase_kwh"]))
    planned_cost_decision = float(np.dot(decision_price_horizon[0], plan["purchase_kwh"]))
    planned_cost_realized = float(np.dot(realised_price_horizon[0], plan["purchase_kwh"]))
    emergency_cost_realized = float(
        np.dot(solve_q2.EMERGENCY_PRICE_MULTIPLIER * realised_price_horizon[0], actual["emergency_kwh"])
    )
    return {
        "planned_purchase_kwh": plan["purchase_kwh"],
        "charge_kwh": actual["charge_kwh"],
        "discharge_kwh": actual["discharge_kwh"],
        "day_ahead_reference_charge_kwh": plan[
            "expected_first_day_charge_kwh"
        ],
        "day_ahead_reference_discharge_kwh": plan[
            "expected_first_day_discharge_kwh"
        ],
        "curtailment_kwh": actual["curtailment_kwh"],
        "soc_kwh": actual["soc_kwh"],
        "emergency_kwh": actual["emergency_kwh"],
        "planned_cost_yuan": planned_cost_realized,
        "planned_cost_forecast_yuan": planned_cost_forecast,
        "planned_cost_decision_yuan": planned_cost_decision,
        "planned_cost_realized_yuan": planned_cost_realized,
        "emergency_cost_yuan": emergency_cost_realized,
        "emergency_cost_realized_yuan": emergency_cost_realized,
        "total_cost_yuan": float(planned_cost_realized + emergency_cost_realized),
        "expected_emergency_cost_yuan": float(plan["expected_emergency_cost_yuan"]),
        "cvar_emergency_cost_yuan": float(plan["cvar_emergency_cost_yuan"]),
        "expected_first_day_terminal_soc_kwh": float(plan["expected_first_day_terminal_soc_kwh"]),
        "expected_terminal_soc_kwh": float(plan["expected_terminal_soc_kwh"]),
        "expected_terminal_shortfall_kwh": float(plan["expected_terminal_shortfall_kwh"]),
        "terminal_target_kwh": terminal_target,
        "trajectory_tracking_enabled": bool(
            actual.get("trajectory_tracking_enabled", False)
        ),
        "trajectory_tracking_penalty_yuan_per_kwh": float(
            solve_q2.TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH
        ),
        "scenario_count": int(plan["scenario_count"]),
        "raw_scenario_count": int(error_history.shape[0]),
        "planning_horizon_days": int(plan["planning_horizon_days"]),
        "forecast_mae_kwh": float(np.mean(np.abs(forecast_horizon[0] - actual_pv))),
        "terminal_soc_shortfall_kwh": float(actual["terminal_soc_shortfall_kwh"]),
        "max_balance_error_kwh": float(actual["max_balance_error_kwh"]),
    }


def _intervals(values: np.ndarray) -> list[dict[str, float | str]]:
    return solve_q3.contiguous_intervals(np.asarray(values, dtype=float))


def solve_all_problem42(
    root: Path,
    *,
    use_price_risk_premium: bool = False,
    use_joint_scenarios: bool = True,
    perfect_price_information: bool = False,
) -> dict:
    """Solve Q4.2, with optional controls used by auditable ablations.

    The default values define the production model.  Ablation switches only
    change information/risk inputs; all physical constraints and settlement
    calculations remain shared with the formal solve.
    """
    dates, prices, load, actual_pv, _ = load_inputs(root)
    _, multiday_forecasts, valid_mask, _ = load_problem2_forecasts(root, actual_pv)
    price_valid_mask = build_price_valid_mask(prices)
    price_context = build_price_context(root, prices, dates=dates)
    causal_prices = price_context["causal_forecasts"]
    base_price_curve = price_context["base_curve"]
    index = {d: i for i, d in enumerate(dates)}
    days: list[dict] = []
    soc = INITIAL_SOC_KWH
    carry_in: dict | None = None
    for day_index, d in enumerate(dates):
        if d == OUTPUT_START:
            carry_in = {
                "date": dates[day_index - 1].isoformat() if day_index else None,
                "terminal_soc_kwh": float(soc),
            }
        horizon_days = min(solve_q2.PLANNING_HORIZON_DAYS, len(dates) - day_index)
        load_horizon = load[day_index : day_index + horizon_days]
        forecast_horizon = multiday_forecasts[day_index, :horizon_days]
        realised_price_horizon = prices[day_index : day_index + horizon_days]
        errors = solve_q2.build_multiday_error_history(
            actual_pv,
            multiday_forecasts,
            valid_mask,
            day_index,
            horizon_days,
        )
        price_error_blocks = build_multiday_price_error_history(
            prices,
            causal_prices,
            day_index,
            horizon_days,
            base_curve=base_price_curve,
            valid_mask=price_valid_mask,
        )
        conditional_price_error_blocks = build_multiday_price_error_history(
            prices,
            causal_prices,
            day_index,
            horizon_days,
            base_curve=base_price_curve,
            valid_mask=np.asarray(valid_mask, dtype=bool) & price_valid_mask,
        )
        if perfect_price_information:
            point_price_horizon = realised_price_horizon.copy()
            decision_price_horizon = realised_price_horizon.copy()
            price_scenarios = np.zeros((0, PERIODS), dtype=float)
            price_probabilities = np.zeros(0, dtype=float)
        else:
            point_price_horizon, decision_price_horizon, price_scenarios, price_probabilities = price_plan_horizon(
                prices,
                causal_prices,
                day_index,
                horizon_days,
                base_price_curve,
                pv_error_history=errors if price_error_blocks is not None else None,
                price_error_history=price_error_blocks,
                conditional_price_error_history=conditional_price_error_blocks,
                risk_quantile=(None if use_price_risk_premium else 0.0),
                risk_weight=(None if use_price_risk_premium else 0.0),
            )
        joint_pv_errors, joint_price_errors = build_joint_multiday_error_history(
            prices,
            causal_prices,
            actual_pv,
            multiday_forecasts,
            valid_mask,
            price_valid_mask,
            day_index,
            horizon_days,
            base_price_curve,
        )
        if use_joint_scenarios and not perfect_price_information:
            joint_pv_scenarios, joint_price_scenarios, joint_probabilities = (
                build_joint_scenario_curves(
                    forecast_horizon.reshape(-1),
                    point_price_horizon.reshape(-1),
                    joint_pv_errors,
                    joint_price_errors,
                )
            )
        else:
            # Perfect-price and no-joint variants retain the PV uncertainty but
            # remove the price shock/PV dependence by using a common price path.
            joint_pv_scenarios, joint_price_scenarios, joint_probabilities = (
                build_pv_only_scenario_curves(
                    forecast_horizon.reshape(-1),
                    decision_price_horizon.reshape(-1),
                    joint_pv_errors,
                )
            )
        result = solve_problem42_day(
            realised_price_horizon,
            load_horizon,
            actual_pv[day_index],
            forecast_horizon,
            soc,
            errors,
            decision_price_horizon=decision_price_horizon,
            point_price_horizon=point_price_horizon,
            scenario_pv_horizon=joint_pv_scenarios,
            scenario_price_horizon=joint_price_scenarios,
            scenario_probabilities=joint_probabilities,
        )
        soc = float(result["soc_kwh"][-1])
        if OUTPUT_START <= d <= OUTPUT_END:
            days.append({
                "date": d.isoformat(),
                "price_yuan_per_kwh": prices[index[d]].tolist(),
                "price_actual_yuan_per_kwh": prices[index[d]].tolist(),
                "price_forecast_00_yuan_per_kwh": point_price_horizon[0].tolist(),
                "price_decision_00_yuan_per_kwh": decision_price_horizon[0].tolist(),
                "price_forecast_00_hourly_yuan_per_kwh": price_forecast.hourly_price_forecast_from_curve(
                    point_price_horizon[0]
                ).tolist(),
                "price_information_cutoff": (
                    dates[day_index - 1].isoformat() if day_index else None
                ),
                "price_model_history_cutoff_date": (
                    dates[day_index - 1].isoformat() if day_index else None
                ),
                "price_information_cutoff_timestamp": f"{d.isoformat()} 00:00",
                "price_observed_through_index": -1,
                "price_forecast_history_count": int(day_index),
                "price_forecast_mae_yuan_per_kwh": float(
                    np.mean(np.abs(point_price_horizon[0] - prices[index[d]]))
                ),
                "price_forecast_rmse_yuan_per_kwh": float(
                    np.sqrt(np.mean((point_price_horizon[0] - prices[index[d]]) ** 2))
                ),
                 "price_error_scenario_count": int(len(price_probabilities)),
                "price_error_scenario_probabilities": price_probabilities.tolist(),
                "price_error_scenario_hash": scenario_hash(price_scenarios),
                "price_error_scenario_role": (
                    "边际价格误差诊断场景；实际日前LP/MPC使用joint_scenario_*中的配对价格路径"
                ),
                "price_error_history_count": int(price_error_blocks.shape[0]),
                "price_conditional_history_count": int(conditional_price_error_blocks.shape[0]),
                "price_error_history_min_history": int(PRICE_CONFIG.min_history),
                "price_risk_quantile_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_QUANTILE_BY_RELEASE.items()
                },
                "price_risk_weight_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_WEIGHT_BY_RELEASE.items()
                },
                "current_interval_price_available_at_execution": True,
                "future_price_information_mode": "causal_forecast_and_scenarios",
                "price_observation_usage": "only_current_interval_at_each_MPC_step",
                "joint_scenario_count": int(len(joint_probabilities)),
                "joint_scenario_probabilities": joint_probabilities.tolist(),
                "joint_error_history_count": int(joint_pv_errors.shape[0]),
                "joint_scenario_hash": joint_scenario_hash(
                    joint_pv_scenarios, joint_price_scenarios
                ),
                 "joint_scenario_role": (
                     "price-PV paired historical scenarios used by day-ahead LP and MPC"
                     if use_joint_scenarios and not perfect_price_information
                     else "PV-only historical scenarios with a common reference price path"
                 ),
                 "scenario_dependence_mode": (
                     "paired_pv_price"
                     if use_joint_scenarios and not perfect_price_information
                     else "pv_only_constant_price"
                 ),
                 "price_risk_premium_enabled": bool(
                     use_price_risk_premium and not perfect_price_information
                 ),
                 "planning_price_input": (
                     "causal_point_forecast_plus_paired_pv_price_scenario_cvar"
                     if use_joint_scenarios and not perfect_price_information
                     else "causal_point_forecast_plus_pv_only_scenario_cvar"
                     if not perfect_price_information
                     else "realised_price_oracle_reference"
                 ),
                "price_risk_premium_yuan_per_kwh": (
                    decision_price_horizon[0] - point_price_horizon[0]
                ).tolist(),
                "forecast_pv_kwh": forecast_horizon[0].tolist(),
                "actual_pv_kwh": actual_pv[index[d]].tolist(),
                "load_kwh": load[index[d]].tolist(),
                "planned_purchase_kwh": result["planned_purchase_kwh"].tolist(),
                "charge_kwh": result["charge_kwh"].tolist(),
                "discharge_kwh": result["discharge_kwh"].tolist(),
                "day_ahead_reference_charge_kwh": result[
                    "day_ahead_reference_charge_kwh"
                ].tolist(),
                "day_ahead_reference_discharge_kwh": result[
                    "day_ahead_reference_discharge_kwh"
                ].tolist(),
                "curtailment_kwh": result["curtailment_kwh"].tolist(),
                "soc_kwh": result["soc_kwh"].tolist(),
                "emergency_kwh": result["emergency_kwh"].tolist(),
                "emergency_intervals": _intervals(result["emergency_kwh"]),
                "planned_cost_yuan": result["planned_cost_yuan"],
                "planned_cost_forecast_yuan": result["planned_cost_forecast_yuan"],
                "planned_cost_decision_yuan": result["planned_cost_decision_yuan"],
                "planned_cost_realized_yuan": result["planned_cost_realized_yuan"],
                "emergency_cost_yuan": result["emergency_cost_yuan"],
                "total_cost_yuan": result["total_cost_yuan"],
                "expected_emergency_cost_yuan": result["expected_emergency_cost_yuan"],
                "cvar_emergency_cost_yuan": result["cvar_emergency_cost_yuan"],
                "expected_first_day_terminal_soc_kwh": result["expected_first_day_terminal_soc_kwh"],
                "expected_terminal_soc_kwh": result["expected_terminal_soc_kwh"],
                "expected_terminal_shortfall_kwh": result["expected_terminal_shortfall_kwh"],
                "planned_terminal_soc_target_kwh": result["terminal_target_kwh"],
                "terminal_soc_shortfall_kwh": result["terminal_soc_shortfall_kwh"],
                "scenario_count": result["scenario_count"],
                "raw_scenario_count": result["raw_scenario_count"],
                "planning_horizon_days": result["planning_horizon_days"],
                "forecast_mae_kwh": result["forecast_mae_kwh"],
                "max_balance_error_kwh": result["max_balance_error_kwh"],
                "initial_soc_kwh": float(result["soc_kwh"][0]),
                "terminal_soc_kwh": float(result["soc_kwh"][-1]),
                "trajectory_tracking_enabled": bool(
                    result["trajectory_tracking_enabled"]
                ),
                "trajectory_tracking_penalty_yuan_per_kwh": float(
                    result["trajectory_tracking_penalty_yuan_per_kwh"]
                ),
            })
    annual = _annual_result(
        "问题四（波动电价）问题二：因果光伏预测 + CVaR日前购电 + 因果滚动储能执行",
        days,
        {
            "price_source": "附件4，每天144个十分钟价格",
            "forecast_source": "价格采用截至目标日前的因果Fourier-岭回归（年周期+周周期）；光伏沿用问题二因果多日预测",
            "strategy": "每天0:00制定当前日计划购电量；实际光伏逐时揭示并执行因果滚动储能控制",
            "trajectory_tracking": "Q2日前风险计划的概率加权第一日充放电轨迹通过L1软惩罚传入因果执行层；实际偏差时保持可行性",
            "trajectory_tracking_penalty_yuan_per_kwh": solve_q2.TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH,
            "spill_definition": "问题二复用的curtailment变量表示总供给溢出（光伏与已计划购电的未利用量）；问题三另行报告unused_grid",
            "emergency_price_multiplier": 5.0,
            "battery_efficiency": ETA,
            "cvar_alpha": solve_q2.CVAR_ALPHA,
            "risk_aversion_lambda": solve_q2.RISK_AVERSION_LAMBDA,
            "stress_scenario": "配对历史PV-价格场景并加入低光伏/高价格压力场景；价格风险通过场景紧急费用与CVaR进入LP",
            "price_known_for_daily_optimization": bool(perfect_price_information),
            "price_risk_premium_enabled": bool(
                use_price_risk_premium and not perfect_price_information
            ),
            "price_forecast_model": {
                "harmonics": list(PRICE_CONFIG.harmonics),
                "weekly_harmonics": list(PRICE_CONFIG.weekly_harmonics),
                "ridge": PRICE_CONFIG.ridge,
                "min_history": PRICE_CONFIG.min_history,
                "weekday_window": PRICE_CONFIG.weekday_window,
                "weekday_blend": PRICE_CONFIG.weekday_blend,
                "risk_quantile": PRICE_RISK_QUANTILE,
                "risk_weight": PRICE_RISK_WEIGHT,
                "risk_quantile_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_QUANTILE_BY_RELEASE.items()
                },
                "risk_weight_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_WEIGHT_BY_RELEASE.items()
                },
                "granularity": "10分钟点预测",
            },
            "price_risk_method": (
                "causal point forecast + paired PV-price scenario CVaR"
                if use_joint_scenarios and not perfect_price_information
                else "causal point forecast + PV-only scenario CVaR"
                if not perfect_price_information
                else "realised price oracle reference"
            ),
            "price_scenario_role": (
                "paired historical PV-price scenarios enter scenario emergency costs and CVaR; no additive first-stage price premium"
                if use_joint_scenarios and not perfect_price_information
                else "PV-only scenarios with a common reference price path; no price-PV pairing"
            ),
            "price_risk_conditioning": "first-stage price premium is disabled in the formal model; same-origin PV-price residual pairs are retained in the joint scenario CVaR",
            "price_settlement": "计划层使用因果点预测价（正式模型不叠加首阶段价格溢价）；最终计划和紧急费用按附件4实际价结算",
            "execution_price_information": {
                "current_interval_price_available": True,
                "future_intervals_use": "causal_forecast_and_scenarios",
                "usage": "each MPC solve replaces only its first coefficient with the current observed price",
            },
            "price_pv_correlation": price_context["correlation"],
        },
    )
    annual["carry_in"] = carry_in
    return annual


def solve_all_problem43(
    root: Path,
    *,
    use_price_risk_premium: bool = False,
    use_joint_scenarios: bool = True,
    perfect_price_information: bool = False,
    enabled_releases: tuple[int, ...] = (0, 6, 12, 18),
) -> dict:
    """Solve Q4.3, with optional controls used by auditable ablations.

    The default values define the production four-release strategy.  A
    shorter ``enabled_releases`` tuple is the no-intraday-update benchmark.
    """
    dates, prices, load, actual_pv, forecasts = load_inputs(root)
    price_context = build_price_context(root, prices, dates=dates)
    causal_prices = price_context["causal_forecasts"]
    base_price_curve = price_context["base_curve"]
    index = {d: i for i, d in enumerate(dates)}
    days: list[dict] = []
    soc = INITIAL_SOC_KWH
    carry_in: dict | None = None
    margin_history = {
        d: solve_q3.causal_error_margins(
            d, dates, actual_pv, forecasts, quantile=0.30, shrink=1.0
        )
        for d in dates
    }
    for d in dates:
        margins = margin_history[d]
        day_index = index[d]
        if d == OUTPUT_START:
            carry_in = {
                "date": dates[day_index - 1].isoformat() if day_index else None,
                "terminal_soc_kwh": float(soc),
            }
        forecast_00_raw = solve_q3.build_release_curves(forecasts[d])[0]
        base_error_history = solve_q3.build_release0_error_history(
            day_index, dates, actual_pv, forecasts
        )
        release_pv_errors = {
            release: build_release_pv_error_history(
                day_index,
                release,
                dates,
                actual_pv,
                forecasts,
                margin_history,
                min_history=PRICE_CONFIG.min_history,
            )
            for release in (0, 6, 12, 18)
        }
        joint_release_errors = {
            release: build_joint_release_error_history(
                day_index,
                release,
                dates,
                actual_pv,
                forecasts,
                margin_history,
                prices,
                causal_prices,
                base_price_curve,
                min_history=PRICE_CONFIG.min_history,
            )
            for release in (0, 6, 12, 18)
        }
        release_prices = {}
        for release in (0, 6, 12, 18):
            context = release_price_context(
                prices,
                causal_prices,
                day_index,
                release,
                base_price_curve,
                pv_error_history=release_pv_errors[release],
                risk_quantile=(None if use_price_risk_premium else 0.0),
                risk_weight=(None if use_price_risk_premium else 0.0),
            )
            if perfect_price_information:
                # The oracle benchmark knows the realised full-day curve at
                # every release; historical residuals are retained only for
                # diagnostics and are not used as price scenarios below.
                realised = prices[day_index].copy()
                context["point"] = realised
                context["robust"] = realised.copy()
                context["hourly"] = price_forecast.hourly_price_forecast_from_curve(realised)
            release_prices[release] = context
        # Use one paired scenario reduction for the base contract.  Keeping
        # the three arrays from the same call is essential: their row order
        # is the dependence structure used by the joint LP.
        if use_joint_scenarios and not perfect_price_information:
            base_joint_pv, base_joint_price, base_joint_probabilities = (
                build_joint_scenario_curves(
                    forecast_00_raw,
                    release_prices[0]["point"],
                    *joint_release_errors[0],
                )
            )
        else:
            base_joint_pv, base_joint_price, base_joint_probabilities = (
                build_pv_only_scenario_curves(
                    forecast_00_raw,
                    release_prices[0]["point"],
                    joint_release_errors[0][0],
                )
            )
        # Use the same risk-aware one-day base contract as the current
        # Question 3 model before applying the four intraday updates.
        base_plan = solve_q3.solve_q2_base_plan(
            release_prices[0]["robust"],
            load[day_index],
            forecast_00_raw,
            soc,
            base_error_history,
            scenario_pv_horizon_kwh=base_joint_pv,
            scenario_price_horizon_kwh=base_joint_price,
            scenario_probabilities=base_joint_probabilities,
        )
        scenario_curves = {}
        scenario_price_curves = {}
        scenario_probabilities = {}
        for release in enabled_releases:
            current_curve = (
                solve_q3.build_release_curves(forecasts[d])[release]
                if release == 0
                else solve_q3.build_release_curves(forecasts[d], margins)[release]
            )
            if use_joint_scenarios and not perfect_price_information:
                curves, price_curves, probabilities = build_joint_scenario_curves(
                    current_curve,
                    release_prices[release]["point"],
                    *joint_release_errors[release],
                )
            else:
                curves, price_curves, probabilities = build_pv_only_scenario_curves(
                    current_curve,
                    release_prices[release]["point"],
                    joint_release_errors[release][0],
                )
            scenario_curves[release] = curves
            scenario_price_curves[release] = price_curves
            scenario_probabilities[release] = probabilities
        result = solve_q3.solve_day(
            release_prices[0]["robust"],
            load[day_index],
            actual_pv[day_index],
            forecasts[d],
            soc,
            margins,
            enabled_releases=enabled_releases,
            initial_plan=base_plan,
            scenario_curves=scenario_curves,
            scenario_probabilities=scenario_probabilities,
            scenario_price_curves=scenario_price_curves,
            price_by_release={
                release: release_prices[release]["robust"]
                for release in (0, 6, 12, 18)
            },
            settlement_price=prices[day_index],
            observed_price=prices[day_index],
        )
        for log in result["update_log"]:
            release = int(str(log["release_time"]).split(":")[0])
            start = price_forecast.release_first_index(release)
            log["price_scenario_count"] = int(
                len(release_prices[release]["scenario_probabilities"])
            )
            log["price_conditional_history_count"] = int(
                release_prices[release]["conditional_history_count"]
            )
            log["price_forecast_granularity"] = "1小时"
            log["price_observed_through_index"] = int(start - 1)
            log["current_interval_price_available_at_execution"] = True
            log["future_price_information_mode"] = "causal_forecast_and_scenarios"
            log["price_observation_usage"] = "only_current_interval_at_each_MPC_step"
            log["price_risk_premium_enabled"] = bool(
                use_price_risk_premium and not perfect_price_information
            )
            log["planning_price_input"] = (
                "causal_point_forecast_plus_paired_pv_price_scenario_cvar"
                if use_joint_scenarios and not perfect_price_information
                else "causal_point_forecast_plus_pv_only_scenario_cvar"
                if not perfect_price_information
                else "realised_price_oracle_reference"
            )
            log["price_information_cutoff"] = (
                dates[day_index - 1].isoformat() if day_index else None
            )
            log["price_model_history_cutoff_date"] = (
                dates[day_index - 1].isoformat() if day_index else None
            )
            log["price_information_cutoff_timestamp"] = (
                f"{d.isoformat()} {release:02d}:00"
            )
            log["joint_scenario_count"] = int(
                len(scenario_probabilities[release])
            )
            log["joint_scenario_probability_sum"] = float(
                np.sum(scenario_probabilities[release])
            )
            log["joint_error_history_count"] = int(
                joint_release_errors[release][0].shape[0]
            )
            log["joint_scenario_hash"] = joint_scenario_hash(
                scenario_curves[release], scenario_price_curves[release]
            )
        soc = float(result["soc"][-1])
        if OUTPUT_START <= d <= OUTPUT_END:
            curves = solve_q3.build_release_curves(forecasts[d], margins)
            planned_cost_forecast = float(np.dot(
                release_prices[0]["point"], result["plan"]
            ))
            planned_cost_decision = float(np.dot(
                release_prices[0]["robust"], result["plan"]
            ))
            days.append({
                "date": d.isoformat(),
                "price_yuan_per_kwh": prices[index[d]].tolist(),
                "price_actual_yuan_per_kwh": prices[index[d]].tolist(),
                "price_forecast_00_yuan_per_kwh": release_prices[0]["point"].tolist(),
                "price_forecast_06_yuan_per_kwh": release_prices[6]["point"].tolist(),
                "price_forecast_12_yuan_per_kwh": release_prices[12]["point"].tolist(),
                "price_forecast_18_yuan_per_kwh": release_prices[18]["point"].tolist(),
                "price_decision_00_yuan_per_kwh": release_prices[0]["robust"].tolist(),
                "price_decision_06_yuan_per_kwh": release_prices[6]["robust"].tolist(),
                "price_decision_12_yuan_per_kwh": release_prices[12]["robust"].tolist(),
                "price_decision_18_yuan_per_kwh": release_prices[18]["robust"].tolist(),
                "price_forecast_00_hourly_yuan_per_kwh": release_prices[0]["hourly"].tolist(),
                "price_forecast_06_hourly_yuan_per_kwh": release_prices[6]["hourly"].tolist(),
                "price_forecast_12_hourly_yuan_per_kwh": release_prices[12]["hourly"].tolist(),
                "price_forecast_18_hourly_yuan_per_kwh": release_prices[18]["hourly"].tolist(),
                "price_information_cutoff": (
                    dates[day_index - 1].isoformat() if day_index else None
                ),
                "price_model_history_cutoff_date": (
                    dates[day_index - 1].isoformat() if day_index else None
                ),
                "price_information_cutoff_timestamp_by_release": {
                    str(release): f"{d.isoformat()} {release:02d}:00"
                    for release in (0, 6, 12, 18)
                },
                "price_observed_through_index_by_release": {
                    str(release): int(price_forecast.release_first_index(release) - 1)
                    for release in (0, 6, 12, 18)
                },
                "price_forecast_history_count": int(day_index),
                "price_risk_quantile_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_QUANTILE_BY_RELEASE.items()
                },
                "price_risk_weight_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_WEIGHT_BY_RELEASE.items()
                },
                "current_interval_price_available_at_execution": True,
                "future_price_information_mode": "causal_forecast_and_scenarios",
                "price_observation_usage": "only_current_interval_at_each_MPC_step",
                "price_forecast_mae_yuan_per_kwh": {
                    str(release): float(np.mean(np.abs(
                        release_prices[release]["point"] - prices[index[d]]
                    )))
                    for release in (0, 6, 12, 18)
                },
                "price_forecast_rmse_yuan_per_kwh": {
                    str(release): float(np.sqrt(np.mean(
                        (release_prices[release]["point"] - prices[index[d]]) ** 2
                    )))
                    for release in (0, 6, 12, 18)
                },
                "price_forecast_future_mae_yuan_per_kwh": {
                    str(release): float(np.mean(np.abs(
                        release_prices[release]["point"][
                            price_forecast.release_first_index(release):
                        ] - prices[index[d],
                            price_forecast.release_first_index(release):]
                    )))
                    for release in (0, 6, 12, 18)
                },
                "price_forecast_future_rmse_yuan_per_kwh": {
                    str(release): float(np.sqrt(np.mean(
                        (
                            release_prices[release]["point"][
                            price_forecast.release_first_index(release):
                            ] - prices[index[d],
                                price_forecast.release_first_index(release):]
                        ) ** 2
                    )))
                    for release in (0, 6, 12, 18)
                },
                "price_error_scenario_count": {
                    str(release): int(len(release_prices[release]["scenario_probabilities"]))
                    for release in (0, 6, 12, 18)
                },
                "price_conditional_history_count": {
                    str(release): int(release_prices[release]["conditional_history_count"])
                    for release in (0, 6, 12, 18)
                },
                "price_error_scenario_probabilities": {
                    str(release): release_prices[release]["scenario_probabilities"].tolist()
                    for release in (0, 6, 12, 18)
                },
                "price_error_scenario_hash_by_release": {
                    str(release): scenario_hash(
                        release_prices[release]["scenario_errors"]
                    )
                    for release in (0, 6, 12, 18)
                },
                "price_error_scenario_role": (
                    "边际价格误差诊断场景；实际调整LP/MPC使用joint_scenario_*中的配对价格路径"
                ),
                "joint_scenario_count_by_release": {
                    str(release): int(len(scenario_probabilities[release]))
                    for release in (0, 6, 12, 18)
                },
                "joint_scenario_probabilities_by_release": {
                    str(release): scenario_probabilities[release].tolist()
                    for release in (0, 6, 12, 18)
                },
                "joint_error_history_count_by_release": {
                    str(release): int(joint_release_errors[release][0].shape[0])
                    for release in (0, 6, 12, 18)
                },
                "joint_scenario_hash_by_release": {
                    str(release): joint_scenario_hash(
                        scenario_curves[release], scenario_price_curves[release]
                    )
                    for release in (0, 6, 12, 18)
                },
             "joint_scenario_role": (
                 "paired PV-price scenarios passed to each adjustment-contract LP and MPC"
                 if use_joint_scenarios and not perfect_price_information
                 else "PV-only scenarios with a common reference price path"
             ),
             "scenario_dependence_mode": (
                 "paired_pv_price"
                 if use_joint_scenarios and not perfect_price_information
                 else "pv_only_constant_price"
             ),
             "price_risk_premium_enabled": bool(
                 use_price_risk_premium and not perfect_price_information
             ),
             "planning_price_input": (
                 "causal_point_forecast_plus_paired_pv_price_scenario_cvar"
                 if use_joint_scenarios and not perfect_price_information
                 else "causal_point_forecast_plus_pv_only_scenario_cvar"
                 if not perfect_price_information
                 else "realised_price_oracle_reference"
             ),
             "price_risk_premium_yuan_per_kwh_by_release": {
                 str(release): (
                     release_prices[release]["robust"]
                     - release_prices[release]["point"]
                 ).tolist()
                 for release in (0, 6, 12, 18)
             },
                "forecast_00_kwh": curves[0].tolist(),
                "forecast_06_kwh": curves[6].tolist(),
                "forecast_12_kwh": curves[12].tolist(),
                "forecast_18_kwh": curves[18].tolist(),
                "actual_pv_kwh": actual_pv[index[d]].tolist(),
                "load_kwh": load[index[d]].tolist(),
                "planned_purchase_kwh": result["plan"].tolist(),
                "adjusted_purchase_kwh": result["adjusted"].tolist(),
                "charge_kwh": result["charge"].tolist(),
                "discharge_kwh": result["discharge"].tolist(),
                "day_ahead_reference_charge_kwh": base_plan[
                    "expected_first_day_charge_kwh"
                ].tolist(),
                "day_ahead_reference_discharge_kwh": base_plan[
                    "expected_first_day_discharge_kwh"
                ].tolist(),
                "curtailment_kwh": result["curtailment"].tolist(),
                "unused_grid_kwh": result["unused_grid"].tolist(),
                "soc_kwh": result["soc"].tolist(),
                "emergency_kwh": result["emergency"].tolist(),
                "emergency_intervals": _intervals(result["emergency"]),
                "planned_cost_yuan": result["planned_cost"],
                "planned_cost_forecast_yuan": planned_cost_forecast,
                "planned_cost_decision_yuan": planned_cost_decision,
                "planned_cost_realized_yuan": result["planned_cost"],
                "base_settlement_cost_yuan": result["base_settlement_cost"],
                "adjustment_cost_yuan": result["adjustment_cost"],
                "normal_settlement_cost_yuan": result["normal_settlement_cost"],
                "penalty_only_settlement_cost_yuan": result["penalty_only_settlement_cost"],
                "emergency_cost_yuan": result["emergency_cost"],
                "total_cost_yuan": result["total_cost"],
             "base_plan_scenario_count": int(base_plan["scenario_count"]),
                "base_plan_expected_emergency_cost_yuan": float(base_plan["expected_emergency_cost_yuan"]),
                "base_plan_cvar_emergency_cost_yuan": float(base_plan["cvar_emergency_cost_yuan"]),
                "base_plan_expected_terminal_soc_kwh": float(base_plan["expected_terminal_soc_kwh"]),
                "base_plan_expected_terminal_shortfall_kwh": float(base_plan["expected_terminal_shortfall_kwh"]),
                "base_plan_planning_horizon_days": int(base_plan["planning_horizon_days"]),
                "max_balance_error_kwh": result["max_balance_error"],
                "initial_soc_kwh": float(result["soc"][0]),
                "terminal_soc_kwh": float(result["soc"][-1]),
                "trajectory_tracking_enabled": bool(
                    result["trajectory_tracking_enabled"]
                ),
                "trajectory_tracking_penalty_yuan_per_kwh": float(
                    result["trajectory_tracking_penalty_yuan_per_kwh"]
                ),
                "update_log": result["update_log"],
            })
    annual = _annual_result(
        "问题四（波动电价）问题三：四阶段预测更新 + 计划/调整购电合同 + 因果储能执行",
        days,
        {
            "price_source": "附件4，每天144个十分钟价格",
            "forecast_source": "附件3四阶段光伏预测 + 截至目标日前的因果价格模型；价格每6小时发布1小时粒度预测",
            "strategy": "四阶段调整合同，更新时刻只修改尚未发生的购电区间；每次价格预测扩展到10分钟后进入对应LP/MPC",
            "trajectory_tracking": "Q2/Q3日前风险计划的概率加权第一日充放电轨迹通过L1软惩罚传入因果执行层；实际偏差时保持可行性",
            "trajectory_tracking_penalty_yuan_per_kwh": solve_q2.TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH,
            "settlement": "p·min(P,A)+0.5p·(P-A)^+ +1.5p·(A-P)^+",
            "emergency_price_multiplier": 5.0,
            "battery_efficiency": ETA,
            "cvar_alpha": solve_q2.CVAR_ALPHA,
             "base_plan": "00:00调用问题三当前版本的问题二风险感知一日LP，输入00:00因果点预测价格和配对PV-价格场景",
             "stress_scenario": "四个发布时间均使用配对历史PV-价格场景，并加入低光伏/高价格压力场景",
             "risk_aversion_lambda": solve_q2.RISK_AVERSION_LAMBDA,
             "price_known_for_daily_optimization": bool(perfect_price_information),
             "price_risk_premium_enabled": bool(
                 use_price_risk_premium and not perfect_price_information
             ),
            "price_forecast_model": {
                "harmonics": list(PRICE_CONFIG.harmonics),
                "weekly_harmonics": list(PRICE_CONFIG.weekly_harmonics),
                "ridge": PRICE_CONFIG.ridge,
                "min_history": PRICE_CONFIG.min_history,
                "weekday_window": PRICE_CONFIG.weekday_window,
                "weekday_blend": PRICE_CONFIG.weekday_blend,
                "risk_quantile": PRICE_RISK_QUANTILE,
                "risk_weight": PRICE_RISK_WEIGHT,
                "risk_quantile_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_QUANTILE_BY_RELEASE.items()
                },
                "risk_weight_by_release": {
                    str(k): float(v) for k, v in PRICE_RISK_WEIGHT_BY_RELEASE.items()
                },
                "release_hours": [0, 6, 12, 18],
                "forecast_granularity": "1小时，再阶梯映射到10分钟",
            },
            "price_risk_method": (
                "causal point forecast + paired PV-price scenario CVaR"
                if use_joint_scenarios and not perfect_price_information
                else "causal point forecast + PV-only scenario CVaR"
                if not perfect_price_information
                else "realised price oracle reference"
            ),
            "price_scenario_role": (
                "paired historical PV-price scenarios enter each release adjustment LP and MPC scenario emergency costs/CVaR; no additive first-stage price premium"
                if use_joint_scenarios and not perfect_price_information
                else "PV-only scenarios with a common reference price path; no price-PV pairing"
            ),
             "enabled_releases": [int(value) for value in enabled_releases],
             "price_risk_conditioning": "first-stage price premium is disabled in the formal model; same-origin PV-price residual pairs are retained in each release's joint scenario CVaR",
            "price_settlement": "预测价格仅用于相应发布时间之后的LP/MPC；最终计划、调整和紧急费用按附件4真实价结算",
            "execution_price_information": {
                "current_interval_price_available": True,
                "future_intervals_use": "release_specific_causal_forecast_and_scenarios",
                "usage": "each MPC solve replaces only its first coefficient with the current observed price",
            },
            "price_pv_correlation": price_context["correlation"],
        },
    )
    annual["carry_in"] = carry_in
    return annual


def _annual_result(name: str, days: list[dict], model: dict) -> dict:
    model = {"name": name, **model}
    emergency = float(sum(sum(d["emergency_kwh"]) for d in days))
    summary = {
        "day_count": len(days),
        "planned_cost_yuan": float(sum(d["planned_cost_yuan"] for d in days)),
        "emergency_cost_yuan": float(sum(d["emergency_cost_yuan"] for d in days)),
        "total_cost_yuan": float(sum(d["total_cost_yuan"] for d in days)),
        "emergency_energy_kwh": emergency,
        "emergency_day_count": int(sum(any(v >= 1e-3 for v in d["emergency_kwh"]) for d in days)),
        "max_balance_error_kwh": float(max((d["max_balance_error_kwh"] for d in days), default=0.0)),
    }
    if days and "planned_cost_forecast_yuan" in days[0]:
        summary.update({
            "planned_cost_forecast_yuan": float(sum(
                d["planned_cost_forecast_yuan"] for d in days
            )),
            "planned_cost_decision_yuan": float(sum(
                d["planned_cost_decision_yuan"] for d in days
            )),
            "planned_cost_realized_yuan": float(sum(
                d["planned_cost_realized_yuan"] for d in days
            )),
        })
    if days and "normal_settlement_cost_yuan" in days[0]:
        summary.update({
            "base_settlement_cost_yuan": float(sum(d["base_settlement_cost_yuan"] for d in days)),
            "adjustment_cost_yuan": float(sum(d["adjustment_cost_yuan"] for d in days)),
            "normal_settlement_cost_yuan": float(sum(d["normal_settlement_cost_yuan"] for d in days)),
            "penalty_only_settlement_cost_yuan": float(sum(d["penalty_only_settlement_cost_yuan"] for d in days)),
        })
    if days and "curtailment_kwh" in days[0]:
        summary.update({
            "curtailment_energy_kwh": float(sum(sum(d["curtailment_kwh"]) for d in days)),
            "simultaneous_charge_discharge_intervals": int(sum(
                np.sum((np.asarray(d["charge_kwh"]) > 1e-7)
                       & (np.asarray(d["discharge_kwh"]) > 1e-7))
                for d in days
            )),
        })
        if "unused_grid_kwh" in days[0]:
            summary["unused_grid_energy_kwh"] = float(
                sum(sum(d["unused_grid_kwh"]) for d in days)
            )
    return {"model": model, "interval_labels": solve_q3.interval_labels(), "days": days, "summary": summary}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out2 = solve_all_problem42(root)
    out3 = solve_all_problem43(root)
    (root / "结果" / "q4_2_solution.json").write_text(json.dumps(out2, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "结果" / "q4_3_solution.json").write_text(json.dumps(out3, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"q4_2": out2["summary"], "q4_3": out3["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
