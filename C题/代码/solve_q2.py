"""Question 2: causal PV forecasting and risk-aware purchase planning.

The load is known at 00:00.  Photovoltaic generation is forecast with a
causal Fourier ridge model, and historical forecast errors are compressed into
whole-day scenarios for the day-ahead risk LP.  The fixed purchase plan is
then executed by a causal rolling-horizon LP as actual PV values are revealed.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from data_processing import (
    DT_HOURS,
    PERIODS_PER_DAY,
    build_interval_labels,
    load_attachment2,
    load_processed_attachment1,
)
from forecast_q2 import (
    FourierConfig,
    ForecastSelection,
    build_causal_error_history,
    build_causal_forecast_history,
    build_causal_multiday_forecast_history,
)


PERIODS = PERIODS_PER_DAY
ETA = 0.90
BATTERY_MIN_KWH = 1200.0
BATTERY_MAX_KWH = 10800.0
BATTERY_POWER_KW = 5000.0
BATTERY_POWER_KWH = BATTERY_POWER_KW * DT_HOURS
INITIAL_SOC_KWH = 6000.0
EMERGENCY_PRICE_MULTIPLIER = 5.0
SCENARIO_MAX_COUNT = 12
CVAR_ALPHA = 0.90
RISK_AVERSION_LAMBDA = 0.25
REPORTING_TOLERANCE_KWH = 1e-3
PLANNING_HORIZON_DAYS = 3
THROUGHPUT_PENALTY_YUAN_PER_KWH = 1e-4
# The day-ahead stochastic LP supplies a nominal storage trajectory.  The
# causal controller tracks it softly; emergency supply and physical bounds
# always take priority when realised PV differs from the forecast.
TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH = 0.10
NONANTICIPATIVITY_TOLERANCE_RATIO = 0.05
# The stress mass covers the 1-alpha CVaR tail, so the added low-PV path is
# fully represented in the upper tail at the selected confidence level.
CVAR_STRESS_PROBABILITY = 1.0 - CVAR_ALPHA
CVAR_STRESS_TAIL_FRACTION = 0.10
CVAR_STRESS_FALLBACK_SHORTFALL_RATIO = 0.30
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
FORECAST_CONFIG = FourierConfig()


def interval_labels() -> list[str]:
    """Return labels aligned with attachment 2 and result2.xlsx."""
    return build_interval_labels()


def read_inputs(root: Path) -> tuple[np.ndarray, list[date], np.ndarray, np.ndarray]:
    attachment1 = load_processed_attachment1(root / "数据" / "processed_attachment1.csv")
    attachment2 = load_attachment2(root / "附件" / "附件2.xlsx")
    return (
        attachment1["price"].to_numpy(dtype=float),
        list(attachment2.dates),
        attachment2.load_kwh,
        attachment2.pv_actual_kwh,
    )


def reduce_error_scenarios(
    error_history: np.ndarray,
    max_count: int = SCENARIO_MAX_COUNT,
    score_mode: str = "norm",
    price: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress historical error curves while preserving represented mass.

    Curves are sorted by their Euclidean error magnitude and split into
    contiguous groups. Each group is replaced by its mean curve and receives
    probability equal to its original sample proportion. This deterministic
    reduction avoids a random clustering seed and keeps the LP sparse.
    """
    if (
        error_history.ndim != 2
        or error_history.shape[1] < PERIODS
        or error_history.shape[1] % PERIODS != 0
    ):
        raise ValueError("历史误差场景必须为(n,144H)矩阵")
    if score_mode not in {"norm", "price_shortage"}:
        raise ValueError(f"未知场景压缩评分方式: {score_mode}")
    sample_count = error_history.shape[0]
    if sample_count == 0:
        # Keep the complete horizon width so an empty historical set remains
        # compatible with both one-day and multi-day scenario tensors.
        return np.zeros((1, error_history.shape[1]), dtype=float), np.ones(1, dtype=float)
    if sample_count <= max_count:
        return error_history.copy(), np.full(sample_count, 1.0 / sample_count)

    if score_mode == "norm":
        scores = np.linalg.norm(error_history, axis=1)
    elif score_mode == "price_shortage":
        if price is None:
            raise ValueError("price_shortage场景压缩需要提供电价向量")
        price = np.asarray(price, dtype=float)
        if price.shape != (PERIODS,) or np.any(price < 0):
            raise ValueError("场景压缩电价必须是长度144的非负向量")
        horizon_days = error_history.shape[1] // PERIODS
        price_horizon = np.tile(price, horizon_days)
        scores = np.sum(
            price_horizon[None, :] * np.maximum(-error_history, 0.0), axis=1
        )
    else:
        # The mode was validated above; this branch is kept for readability.
        raise ValueError(f"未知场景压缩评分方式: {score_mode}")
    order = np.argsort(scores, kind="mergesort")
    groups = np.array_split(order, max_count)
    centers = np.vstack([error_history[group].mean(axis=0) for group in groups])
    probabilities = np.asarray([len(group) / sample_count for group in groups], dtype=float)
    return centers, probabilities


def build_information_groups(
    scenario_pv_flat_kwh: np.ndarray,
    forecast_pv_flat_kwh: np.ndarray,
    observed_end: int,
    tolerance_ratio: float = NONANTICIPATIVITY_TOLERANCE_RATIO,
    scenario_price_flat: np.ndarray | None = None,
    forecast_price_flat: np.ndarray | None = None,
    price_tolerance_ratio: float = NONANTICIPATIVITY_TOLERANCE_RATIO,
) -> tuple[tuple[int, ...], ...]:
    """Group scenarios that are observationally indistinguishable so far.

    A decision at interval ``t`` may use the actual PV observed for that
    interval, so callers pass ``observed_end=t`` for storage recourse.  For a
    next-day preview purchase, callers pass the previous day's last interval.
    The tolerance is relative to the point forecast and avoids treating tiny
    numerical differences as immediate scenario identification.
    """
    scenarios = np.asarray(scenario_pv_flat_kwh, dtype=float)
    forecast = np.asarray(forecast_pv_flat_kwh, dtype=float).reshape(-1)
    if scenarios.ndim != 2 or scenarios.shape[1] != forecast.size:
        raise ValueError("场景与预测光伏展平矩阵维度错误")
    if not 0.0 <= tolerance_ratio:
        raise ValueError("非预见性观测容差比例不得为负")
    scenario_count = scenarios.shape[0]
    if scenario_count == 0:
        return ()
    prices = None
    price_forecast = None
    if scenario_price_flat is not None or forecast_price_flat is not None:
        if scenario_price_flat is None or forecast_price_flat is None:
            raise ValueError("场景价格和价格预测必须同时提供")
        prices = np.asarray(scenario_price_flat, dtype=float)
        price_forecast = np.asarray(forecast_price_flat, dtype=float).reshape(-1)
        if prices.shape != scenarios.shape or price_forecast.shape != forecast.shape:
            raise ValueError("场景价格与场景光伏维度不一致")
        if not np.isfinite(prices).all() or np.any(prices < 0.0):
            raise ValueError("场景价格必须为有限非负数")
        if not np.isfinite(price_forecast).all() or np.any(price_forecast < 0.0):
            raise ValueError("价格预测必须为有限非负数")
        if price_tolerance_ratio < 0.0:
            raise ValueError("价格观测容差比例不得为负")
    if observed_end < 0:
        return (tuple(range(scenario_count)),)
    if observed_end >= scenarios.shape[1]:
        raise ValueError("非预见性观测时段超出场景范围")

    observed = scenarios[:, : observed_end + 1]
    tolerance = np.maximum(
        1.0,
        tolerance_ratio * np.maximum(np.abs(forecast[: observed_end + 1]), 1.0),
    )
    groups: list[tuple[int, ...]] = []
    unassigned = set(range(scenario_count))
    observed_prices = (
        None if prices is None else prices[:, : observed_end + 1]
    )
    price_tolerance = (
        None
        if price_forecast is None
        else price_tolerance_ratio
        * np.maximum(np.abs(price_forecast[: observed_end + 1]), 0.01)
    )
    while unassigned:
        anchor = min(unassigned)
        members = tuple(
            candidate
            for candidate in sorted(unassigned)
            if np.all(np.abs(observed[candidate] - observed[anchor]) <= tolerance)
            and (
                observed_prices is None
                or np.all(
                    np.abs(
                        observed_prices[candidate] - observed_prices[anchor]
                    )
                    <= price_tolerance
                )
            )
        )
        groups.append(members)
        unassigned.difference_update(members)
    return tuple(groups)


def append_pessimistic_error_scenario(
    scenario_errors: np.ndarray,
    scenario_probabilities: np.ndarray,
    historical_errors: np.ndarray,
    forecast_horizon_kwh: np.ndarray,
    price: np.ndarray,
    stress_probability: float | None = None,
    tail_fraction: float = CVAR_STRESS_TAIL_FRACTION,
    fallback_shortfall_ratio: float = CVAR_STRESS_FALLBACK_SHORTFALL_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a low-PV stress scenario so CVaR sees an empirical tail.

    Historical scenarios are retained.  The extra scenario is the mean of the
    worst historical error blocks ranked by price-weighted PV shortage.  When
    no historical block exists, a conservative downward perturbation of the
    point forecast is used for cold start.
    """
    scenarios = np.asarray(scenario_errors, dtype=float)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    history = np.asarray(historical_errors, dtype=float)
    forecast = np.asarray(forecast_horizon_kwh, dtype=float)
    daily_price = np.asarray(price, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != forecast.size:
        raise ValueError("压力场景与误差场景维度不一致")
    if probabilities.shape != (len(scenarios),) or np.any(probabilities < 0):
        raise ValueError("压力场景概率维度或取值错误")
    if history.ndim != 2 or history.shape[1] != forecast.size:
        raise ValueError("历史误差与预测窗口维度不一致")
    if daily_price.shape == (PERIODS,):
        price_horizon_input = None
    elif daily_price.size == forecast.size:
        # Problem 4 supplies a day-specific tariff matrix.  Keep the
        # original 144-point input unchanged while allowing the same stress
        # scenario logic to rank shortages under a fluctuating tariff.
        price_horizon_input = daily_price.reshape(-1)
    else:
        raise ValueError("压力场景电价必须为144向量或与预测窗口等长")
    if stress_probability is None:
        stress_probability = CVAR_STRESS_PROBABILITY
    if not 0.0 < stress_probability < 1.0:
        raise ValueError("压力场景概率必须位于(0,1)")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("压力场景尾部比例必须位于(0,1]")
    if not 0.0 <= fallback_shortfall_ratio <= 1.0:
        raise ValueError("冷启动压力场景下调比例必须位于[0,1]")

    if history.shape[0]:
        if forecast.size % PERIODS:
            raise ValueError("压力场景预测窗口长度必须是144的整数倍")
        horizon_days = forecast.size // PERIODS
        price_horizon = (
            np.tile(daily_price, horizon_days)
            if price_horizon_input is None
            else price_horizon_input
        )
        shortage_cost = np.sum(
            EMERGENCY_PRICE_MULTIPLIER
            * price_horizon[None, :]
            * np.maximum(-history, 0.0),
            axis=1,
        )
        tail_count = max(1, int(np.ceil(tail_fraction * history.shape[0])))
        tail_indices = np.argsort(shortage_cost, kind="mergesort")[-tail_count:]
        stress_error = history[tail_indices].mean(axis=0)
    else:
        stress_error = (
            -fallback_shortfall_ratio * forecast.reshape(-1)
        )

    if len(scenarios):
        probabilities = probabilities / probabilities.sum()
        probabilities *= 1.0 - stress_probability
    else:
        probabilities = np.zeros(0, dtype=float)
    scenarios = np.vstack((scenarios, stress_error.reshape(1, -1)))
    probabilities = np.append(probabilities, stress_probability)
    return scenarios, probabilities


def build_multiday_error_history(
    actual_pv_kwh: np.ndarray,
    multiday_forecasts_kwh: np.ndarray,
    valid_mask: np.ndarray,
    origin_index: int,
    horizon_days: int,
) -> np.ndarray:
    """Build historical lead-aware error blocks available at one origin."""
    actual = np.asarray(actual_pv_kwh, dtype=float)
    forecasts = np.asarray(multiday_forecasts_kwh, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if actual.ndim != 2 or actual.shape[1] != PERIODS:
        raise ValueError("实际光伏必须为(n,144)矩阵")
    if forecasts.ndim != 3 or forecasts.shape[0] != len(actual):
        raise ValueError("多日光伏预测矩阵维度错误")
    if valid.shape != (len(actual),):
        raise ValueError("valid_mask长度错误")
    if not 0 <= origin_index <= len(actual):
        raise ValueError("预测原点超出日期范围")
    if not 1 <= horizon_days <= forecasts.shape[1]:
        raise ValueError("误差场景窗口长度错误")

    blocks: list[np.ndarray] = []
    latest_start = origin_index - horizon_days
    for historical_origin in range(max(latest_start + 1, 0)):
        if not valid[historical_origin]:
            continue
        block = forecasts[
            historical_origin, :horizon_days
        ]
        if not np.isfinite(block).all():
            continue
        actual_block = actual[
            historical_origin : historical_origin + horizon_days
        ]
        blocks.append((actual_block - block).reshape(-1))
    if not blocks:
        return np.zeros((0, horizon_days * PERIODS), dtype=float)
    return np.vstack(blocks)


def empirical_shortage_cvar(
    shortage_kwh: np.ndarray,
    alpha: float = CVAR_ALPHA,
) -> np.ndarray:
    """Return the empirical upper-tail CVaR of shortage for each interval."""
    if shortage_kwh.size == 0:
        return np.zeros(shortage_kwh.shape[1], dtype=float)
    sorted_values = np.sort(shortage_kwh, axis=0)
    quantile_index = min(sorted_values.shape[0] - 1, int(np.ceil(alpha * sorted_values.shape[0])) - 1)
    threshold = sorted_values[quantile_index]
    tail = np.where(shortage_kwh >= threshold, shortage_kwh, np.nan)
    return np.nanmean(tail, axis=0)


def solve_day_ahead_plan(
    price: np.ndarray,
    load_kwh: np.ndarray,
    forecast_pv_kwh: np.ndarray,
    initial_soc_kwh: float,
) -> dict[str, np.ndarray | float]:
    """Solve one day's deterministic LP from the carried-over SOC.

    There is no artificial daily terminal target.  The terminal SOC is a
    state carried into the next calendar day.
    """
    n = PERIODS
    if not BATTERY_MIN_KWH - 1e-9 <= initial_soc_kwh <= BATTERY_MAX_KWH + 1e-9:
        raise ValueError("日前模型的初始储能量超出允许范围")
    grid = slice(0, n)
    charge = slice(n, 2 * n)
    discharge = slice(2 * n, 3 * n)
    curtailment = slice(3 * n, 4 * n)
    soc = slice(4 * n, 4 * n + n + 1)
    variable_count = 5 * n + 1

    objective = np.zeros(variable_count)
    objective[grid] = price
    objective[charge] = 1e-9
    objective[discharge] = 1e-9

    rows = 2 * n + 1
    matrix = np.zeros((rows, variable_count), dtype=float)
    rhs = np.zeros(rows)
    row_index = 0
    for t in range(n):
        matrix[row_index, grid.start + t] = 1.0
        matrix[row_index, charge.start + t] = -1.0
        matrix[row_index, discharge.start + t] = 1.0
        matrix[row_index, curtailment.start + t] = -1.0
        rhs[row_index] = load_kwh[t] - forecast_pv_kwh[t]
        row_index += 1

        matrix[row_index, soc.start + t + 1] = 1.0
        matrix[row_index, soc.start + t] = -1.0
        matrix[row_index, charge.start + t] = -ETA
        matrix[row_index, discharge.start + t] = 1.0 / ETA
        row_index += 1

    matrix[row_index, soc.start] = 1.0
    rhs[row_index] = initial_soc_kwh

    bounds = (
        [(0.0, None)] * n
        + [(0.0, BATTERY_POWER_KWH)] * n
        + [(0.0, BATTERY_POWER_KWH)] * n
        + [(0.0, None)] * n
        + [(BATTERY_MIN_KWH, BATTERY_MAX_KWH)] * (n + 1)
    )
    power_ub = np.zeros((n, variable_count), dtype=float)
    for t in range(n):
        power_ub[t, charge.start + t] = 1.0
        power_ub[t, discharge.start + t] = 1.0
    solution = linprog(
        objective,
        A_ub=power_ub,
        b_ub=np.full(n, BATTERY_POWER_KWH),
        A_eq=matrix,
        b_eq=rhs,
        bounds=bounds,
        method="highs-ds",
    )
    if not solution.success:
        raise RuntimeError(f"日前计划线性规划失败: {solution.message}")

    values = solution.x
    return {
        "purchase_kwh": values[grid].copy(),
        "charge_kwh": values[charge].copy(),
        "discharge_kwh": values[discharge].copy(),
        "soc_kwh": values[soc].copy(),
        "planned_cost_yuan": float(np.dot(price, values[grid])),
    }


def solve_risk_aware_plan(
    price: np.ndarray,
    load_kwh: np.ndarray,
    forecast_pv_kwh: np.ndarray,
    scenario_pv_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    initial_soc_kwh: float,
    terminal_value_yuan_per_kwh: float,
) -> dict[str, np.ndarray | float | int]:
    """Solve the risk-aware day-ahead LP with scenario recourse.

    The common first-stage variable is the fixed purchase plan. Scenario
    recourse is used to price forecast risk; actual operation is still carried
    out by the causal rolling controller below. The scenario states start from
    the same observed SOC, while their terminal states may differ.
    """
    n = PERIODS
    scenario_pv_kwh = np.asarray(scenario_pv_kwh, dtype=float)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    if scenario_pv_kwh.ndim != 2 or scenario_pv_kwh.shape[1] != n:
        raise ValueError("场景光伏矩阵必须为(m,144)")
    scenario_count = scenario_pv_kwh.shape[0]
    if probabilities.shape != (scenario_count,) or np.any(probabilities < 0):
        raise ValueError("场景概率维度或取值错误")
    probability_sum = float(probabilities.sum())
    if probability_sum <= 0:
        raise ValueError("场景概率和必须为正")
    probabilities = probabilities / probability_sum
    if len(price) != n or len(load_kwh) != n or len(forecast_pv_kwh) != n:
        raise ValueError("日前风险模型输入必须包含144个时段")
    if not BATTERY_MIN_KWH - 1e-9 <= initial_soc_kwh <= BATTERY_MAX_KWH + 1e-9:
        raise ValueError("日前风险模型的初始储能量超出允许范围")

    # Per-scenario blocks: gL, gC, unused, rL, rC, curtailment, charge,
    # discharge, SOC, emergency, and CVaR excess variable.
    block_size = 10 * n + 2
    blocks = {
        "g_load": 0,
        "g_charge": n,
        "g_unused": 2 * n,
        "r_load": 3 * n,
        "r_charge": 4 * n,
        "curtailment": 5 * n,
        "charge": 6 * n,
        "discharge": 7 * n,
        "soc": 8 * n,
        "emergency": 9 * n + 1,
        "xi": 10 * n + 1,
    }
    eta_index = n + scenario_count * block_size
    variable_count = eta_index + 1

    objective = np.zeros(variable_count, dtype=float)
    objective[:n] = price
    objective[eta_index] = RISK_AVERSION_LAMBDA

    equal_rows = scenario_count * (5 * n + 1)
    # One aggregate charge/discharge power row per interval plus one CVaR row
    # per scenario.  The aggregate row removes the otherwise degenerate
    # simultaneous charge/discharge solutions returned by an LP basis.
    upper_rows = scenario_count * (n + 1)
    eq_row: list[int] = []
    eq_col: list[int] = []
    eq_value: list[float] = []
    eq_rhs = np.zeros(equal_rows, dtype=float)
    ub_row: list[int] = []
    ub_col: list[int] = []
    ub_value: list[float] = []
    ub_rhs = np.zeros(upper_rows, dtype=float)

    def variable(scenario: int, block: str, period: int = 0) -> int:
        return n + scenario * block_size + blocks[block] + period

    for scenario in range(scenario_count):
        offset = scenario * (5 * n + 1)
        probability = float(probabilities[scenario])
        objective[variable(scenario, "emergency"): variable(scenario, "emergency") + n] = (
            probability * EMERGENCY_PRICE_MULTIPLIER * price
        )
        objective[variable(scenario, "charge"): variable(scenario, "charge") + n] = probability * 1e-4
        objective[variable(scenario, "discharge"): variable(scenario, "discharge") + n] = probability * 1e-4
        objective[variable(scenario, "xi")] = RISK_AVERSION_LAMBDA * probability / (1.0 - CVAR_ALPHA)
        objective[variable(scenario, "soc", n)] = -probability * terminal_value_yuan_per_kwh

        for t in range(n):
            row = offset + t
            eq_row.extend((row, row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "g_load", t),
                    variable(scenario, "g_charge", t),
                    variable(scenario, "g_unused", t),
                    t,
                )
            )
            eq_value.extend((1.0, 1.0, 1.0, -1.0))

            row = offset + n + t
            eq_row.extend((row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "r_load", t),
                    variable(scenario, "r_charge", t),
                    variable(scenario, "curtailment", t),
                )
            )
            eq_value.extend((1.0, 1.0, 1.0))
            eq_rhs[row] = scenario_pv_kwh[scenario, t]

            row = offset + 2 * n + t
            eq_row.extend((row, row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "g_load", t),
                    variable(scenario, "r_load", t),
                    variable(scenario, "discharge", t),
                    variable(scenario, "emergency", t),
                )
            )
            eq_value.extend((1.0, 1.0, 1.0, 1.0))
            eq_rhs[row] = load_kwh[t]

            row = offset + 3 * n + t
            eq_row.extend((row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "g_charge", t),
                    variable(scenario, "r_charge", t),
                    variable(scenario, "charge", t),
                )
            )
            eq_value.extend((1.0, 1.0, -1.0))

            row = offset + 4 * n + t
            eq_row.extend((row, row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "soc", t + 1),
                    variable(scenario, "soc", t),
                    variable(scenario, "charge", t),
                    variable(scenario, "discharge", t),
                )
            )
            eq_value.extend((1.0, -1.0, -ETA, 1.0 / ETA))

        row = offset + 5 * n
        eq_row.append(row)
        eq_col.append(variable(scenario, "soc"))
        eq_value.append(1.0)
        eq_rhs[row] = initial_soc_kwh

        for t in range(n):
            row = scenario * (n + 1) + t
            ub_row.extend([row, row])
            ub_col.extend([variable(scenario, "charge", t), variable(scenario, "discharge", t)])
            ub_value.extend([1.0, 1.0])
            ub_rhs[row] = BATTERY_POWER_KWH
        row = scenario * (n + 1) + n
        ub_row.extend([row] * (n + 2))
        ub_col.extend([variable(scenario, "emergency", t) for t in range(n)] + [eta_index, variable(scenario, "xi")])
        ub_value.extend(list(EMERGENCY_PRICE_MULTIPLIER * price) + [-1.0, -1.0])
        ub_rhs[row] = 0.0

    equality_matrix = coo_matrix(
        (eq_value, (eq_row, eq_col)), shape=(equal_rows, variable_count)
    ).tocsr()
    upper_matrix = coo_matrix(
        (ub_value, (ub_row, ub_col)), shape=(upper_rows, variable_count)
    ).tocsr()

    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n
    for _ in range(scenario_count):
        bounds.extend([(0.0, None)] * (6 * n))
        bounds.extend([(0.0, BATTERY_POWER_KWH)] * (2 * n))
        bounds.extend([(BATTERY_MIN_KWH, BATTERY_MAX_KWH)] * (n + 1))
        bounds.extend([(0.0, None)] * n)
        bounds.append((0.0, None))
    bounds.append((0.0, None))

    solution = linprog(
        objective,
        A_ub=upper_matrix,
        b_ub=ub_rhs,
        A_eq=equality_matrix,
        b_eq=eq_rhs,
        bounds=bounds,
        method="highs-ds",
    )
    if not solution.success:
        raise RuntimeError(f"风险感知日前线性规划失败: {solution.message}")

    values = solution.x
    emergency_costs = np.asarray(
        [
            float(
                np.dot(
                    EMERGENCY_PRICE_MULTIPLIER * price,
                    values[variable(scenario, "emergency"): variable(scenario, "emergency") + n],
                )
            )
            for scenario in range(scenario_count)
        ]
    )
    cvar = float(
        values[eta_index]
        + np.dot(probabilities, values[[variable(scenario, "xi") for scenario in range(scenario_count)]])
        / (1.0 - CVAR_ALPHA)
    )
    return {
        "purchase_kwh": values[:n].copy(),
        "planned_cost_yuan": float(np.dot(price, values[:n])),
        "expected_emergency_cost_yuan": float(np.dot(probabilities, emergency_costs)),
        "cvar_emergency_cost_yuan": cvar,
        "expected_terminal_soc_kwh": float(
            np.dot(
                probabilities,
                [values[variable(scenario, "soc", n)] for scenario in range(scenario_count)],
            )
        ),
        "objective_value_yuan": float(solution.fun),
        "scenario_count": int(scenario_count),
    }


def solve_multiday_risk_aware_plan(
    price: np.ndarray,
    load_horizon_kwh: np.ndarray,
    forecast_pv_horizon_kwh: np.ndarray,
    scenario_pv_horizon_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    initial_soc_kwh: float,
    terminal_shortfall_penalty_yuan_per_kwh: float,
    cvar_alpha: float | None = None,
    risk_aversion_lambda: float | None = None,
    scenario_price_horizon_kwh: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Optimize a rolling multi-day window and commit only the first day.

    The current day's 144 purchase quantities are common to every scenario.
    Purchases after the first day are scenario-dependent preview decisions,
    representing the fact that they will be replanned at the next 00:00.
    A soft terminal inventory constraint penalizes depletion below the opening
    SOC but never rewards surplus inventory at the artificial horizon end.
    """
    price_input = np.asarray(price, dtype=float)
    alpha = CVAR_ALPHA if cvar_alpha is None else float(cvar_alpha)
    risk_lambda = (
        RISK_AVERSION_LAMBDA
        if risk_aversion_lambda is None
        else float(risk_aversion_lambda)
    )
    if not 0.0 < alpha < 1.0:
        raise ValueError("CVaR置信水平必须位于(0,1)")
    if risk_lambda < 0.0:
        raise ValueError("CVaR风险权重不得为负")
    load = np.asarray(load_horizon_kwh, dtype=float)
    forecast = np.asarray(forecast_pv_horizon_kwh, dtype=float)
    scenario_pv = np.asarray(scenario_pv_horizon_kwh, dtype=float)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    if load.ndim != 2 or load.shape[1] != PERIODS or load.shape != forecast.shape:
        raise ValueError("三日负荷和光伏预测必须为(H,144)矩阵")
    horizon_days = load.shape[0]
    if not 1 <= horizon_days <= PLANNING_HORIZON_DAYS:
        raise ValueError("滚动计划窗口必须为1至3天")
    if price_input.shape == (PERIODS,):
        daily_price = price_input
        price_horizon = np.tile(daily_price, horizon_days)
    elif price_input.shape == (horizon_days, PERIODS):
        daily_price = price_input[0]
        price_horizon = price_input.reshape(-1)
    else:
        raise ValueError("三日计划电价必须为144向量或(H,144)矩阵")
    if scenario_pv.ndim != 3 or scenario_pv.shape[1:] != load.shape:
        raise ValueError("三日场景光伏必须为(m,H,144)矩阵")
    scenario_count = scenario_pv.shape[0]
    if probabilities.shape != (scenario_count,) or np.any(probabilities < 0):
        raise ValueError("三日场景概率维度或取值错误")
    probability_sum = float(probabilities.sum())
    if probability_sum <= 0.0:
        raise ValueError("三日场景概率和必须为正")
    probabilities = probabilities / probability_sum
    n = horizon_days * PERIODS
    scenario_price_flat = None
    if scenario_price_horizon_kwh is not None:
        scenario_price = np.asarray(scenario_price_horizon_kwh, dtype=float)
        if scenario_price.ndim == 3:
            if scenario_price.shape != scenario_pv.shape:
                raise ValueError("三日场景电价必须与场景光伏同为(m,H,144)矩阵")
            scenario_price_flat = scenario_price.reshape(scenario_count, n)
        elif scenario_price.ndim == 2:
            if scenario_price.shape != (scenario_count, n):
                raise ValueError("三日场景电价必须为(m,H*144)矩阵")
            scenario_price_flat = scenario_price.copy()
        else:
            raise ValueError("三日场景电价维度错误")
        if not np.isfinite(scenario_price_flat).all() or np.any(scenario_price_flat < 0.0):
            raise ValueError("三日场景电价必须为有限非负数")
    if not BATTERY_MIN_KWH - 1e-9 <= initial_soc_kwh <= BATTERY_MAX_KWH + 1e-9:
        raise ValueError("三日计划的初始储能量超出允许范围")
    if terminal_shortfall_penalty_yuan_per_kwh < 0.0:
        raise ValueError("末端库存缺口惩罚不得为负")

    future_n = n - PERIODS
    load_flat = load.reshape(-1)
    forecast_flat = forecast.reshape(-1)
    scenario_flat = scenario_pv.reshape(scenario_count, n)

    # A scenario can only receive a different recourse decision after its
    # observed PV history is distinguishable from another scenario.  Future
    # preview purchases are decided at the next day's 00:00, so their
    # information set ends at the previous day rather than the current slot.
    nonanticipativity_constraints: list[tuple[str, int, int, int]] = []
    for t in range(n):
        groups = build_information_groups(
            scenario_flat,
            forecast_flat,
            observed_end=t,
            scenario_price_flat=scenario_price_flat,
            forecast_price_flat=(
                price_horizon if scenario_price_flat is not None else None
            ),
        )
        for group in groups:
            anchor = group[0]
            for scenario in group[1:]:
                for block in ("charge", "discharge", "emergency"):
                    nonanticipativity_constraints.append((block, t, scenario, anchor))
    for day_start in range(PERIODS, n, PERIODS):
        groups = build_information_groups(
            scenario_flat,
            forecast_flat,
            observed_end=day_start - 1,
            scenario_price_flat=scenario_price_flat,
            forecast_price_flat=(
                price_horizon if scenario_price_flat is not None else None
            ),
        )
        for group in groups:
            anchor = group[0]
            for scenario in group[1:]:
                # Only the next calendar day's purchase block is decided at
                # this information origin. Later days are decided after more
                # observations and must not inherit this earlier information.
                for t in range(day_start, min(day_start + PERIODS, n)):
                    nonanticipativity_constraints.append(
                        ("future_grid", t - PERIODS, scenario, anchor)
                    )

    # Per-scenario blocks: source allocation, battery operation, SOC,
    # emergency purchase, future preview purchases, CVaR excess, and terminal
    # inventory shortfall.
    blocks = {
        "g_load": 0,
        "g_charge": n,
        "g_unused": 2 * n,
        "r_load": 3 * n,
        "r_charge": 4 * n,
        "curtailment": 5 * n,
        "charge": 6 * n,
        "discharge": 7 * n,
        "soc": 8 * n,
        "emergency": 9 * n + 1,
        "future_grid": 10 * n + 1,
        "xi": 10 * n + 1 + future_n,
        "terminal_deficit": 10 * n + 2 + future_n,
    }
    block_size = 10 * n + 3 + future_n
    eta_index = PERIODS + scenario_count * block_size
    variable_count = eta_index + 1

    def variable(scenario: int, block: str, period: int = 0) -> int:
        return PERIODS + scenario * block_size + blocks[block] + period

    objective = np.zeros(variable_count, dtype=float)
    objective[:PERIODS] = daily_price
    objective[eta_index] = risk_lambda

    base_equal_rows = scenario_count * (5 * n + 1)
    equal_rows = base_equal_rows + len(nonanticipativity_constraints)
    upper_rows = scenario_count * (n + 2)
    eq_row: list[int] = []
    eq_col: list[int] = []
    eq_value: list[float] = []
    eq_rhs = np.zeros(equal_rows, dtype=float)
    ub_row: list[int] = []
    ub_col: list[int] = []
    ub_value: list[float] = []
    ub_rhs = np.zeros(upper_rows, dtype=float)

    for scenario in range(scenario_count):
        probability = float(probabilities[scenario])
        scenario_price_row = (
            price_horizon
            if scenario_price_flat is None
            else scenario_price_flat[scenario]
        )
        offset = scenario * (5 * n + 1)
        if future_n:
            objective[
                variable(scenario, "future_grid") :
                variable(scenario, "future_grid") + future_n
            ] = probability * scenario_price_row[PERIODS:]
        objective[
            variable(scenario, "emergency") :
            variable(scenario, "emergency") + n
        ] = probability * EMERGENCY_PRICE_MULTIPLIER * scenario_price_row
        objective[
            variable(scenario, "charge") : variable(scenario, "charge") + n
        ] = probability * THROUGHPUT_PENALTY_YUAN_PER_KWH
        objective[
            variable(scenario, "discharge") : variable(scenario, "discharge") + n
        ] = probability * THROUGHPUT_PENALTY_YUAN_PER_KWH
        objective[variable(scenario, "xi")] = (
            risk_lambda * probability / (1.0 - alpha)
        )
        objective[variable(scenario, "terminal_deficit")] = (
            probability * terminal_shortfall_penalty_yuan_per_kwh
        )

        for t in range(n):
            plan_variable = (
                t
                if t < PERIODS
                else variable(scenario, "future_grid", t - PERIODS)
            )
            row = offset + t
            eq_row.extend((row, row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "g_load", t),
                    variable(scenario, "g_charge", t),
                    variable(scenario, "g_unused", t),
                    plan_variable,
                )
            )
            eq_value.extend((1.0, 1.0, 1.0, -1.0))

            row = offset + n + t
            eq_row.extend((row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "r_load", t),
                    variable(scenario, "r_charge", t),
                    variable(scenario, "curtailment", t),
                )
            )
            eq_value.extend((1.0, 1.0, 1.0))
            eq_rhs[row] = scenario_flat[scenario, t]

            row = offset + 2 * n + t
            eq_row.extend((row, row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "g_load", t),
                    variable(scenario, "r_load", t),
                    variable(scenario, "discharge", t),
                    variable(scenario, "emergency", t),
                )
            )
            eq_value.extend((1.0, 1.0, 1.0, 1.0))
            eq_rhs[row] = load_flat[t]

            row = offset + 3 * n + t
            eq_row.extend((row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "g_charge", t),
                    variable(scenario, "r_charge", t),
                    variable(scenario, "charge", t),
                )
            )
            eq_value.extend((1.0, 1.0, -1.0))

            row = offset + 4 * n + t
            eq_row.extend((row, row, row, row))
            eq_col.extend(
                (
                    variable(scenario, "soc", t + 1),
                    variable(scenario, "soc", t),
                    variable(scenario, "charge", t),
                    variable(scenario, "discharge", t),
                )
            )
            eq_value.extend((1.0, -1.0, -ETA, 1.0 / ETA))

        row = offset + 5 * n
        eq_row.append(row)
        eq_col.append(variable(scenario, "soc"))
        eq_value.append(1.0)
        eq_rhs[row] = initial_soc_kwh

        upper_offset = scenario * (n + 2)
        for t in range(n):
            row = upper_offset + t
            ub_row.extend((row, row))
            ub_col.extend(
                (
                    variable(scenario, "charge", t),
                    variable(scenario, "discharge", t),
                )
            )
            ub_value.extend((1.0, 1.0))
            ub_rhs[row] = BATTERY_POWER_KWH

        row = upper_offset + n
        ub_row.extend([row] * (n + 2))
        ub_col.extend(
            [variable(scenario, "emergency", t) for t in range(n)]
            + [eta_index, variable(scenario, "xi")]
        )
        ub_value.extend(
            list(EMERGENCY_PRICE_MULTIPLIER * scenario_price_row) + [-1.0, -1.0]
        )

        row = upper_offset + n + 1
        ub_row.extend((row, row))
        ub_col.extend(
            (
                variable(scenario, "soc", n),
                variable(scenario, "terminal_deficit"),
            )
        )
        ub_value.extend((-1.0, -1.0))
        ub_rhs[row] = -initial_soc_kwh

    for row, (block, period, scenario, anchor) in enumerate(
        nonanticipativity_constraints,
        start=base_equal_rows,
    ):
        eq_row.extend((row, row))
        eq_col.extend(
            (
                variable(scenario, block, period),
                variable(anchor, block, period),
            )
        )
        eq_value.extend((1.0, -1.0))

    equality_matrix = coo_matrix(
        (eq_value, (eq_row, eq_col)), shape=(equal_rows, variable_count)
    ).tocsr()
    upper_matrix = coo_matrix(
        (ub_value, (ub_row, ub_col)), shape=(upper_rows, variable_count)
    ).tocsr()

    bounds: list[tuple[float, float | None]] = [(0.0, None)] * PERIODS
    for _ in range(scenario_count):
        bounds.extend([(0.0, None)] * (6 * n))
        bounds.extend([(0.0, BATTERY_POWER_KWH)] * (2 * n))
        bounds.extend([(BATTERY_MIN_KWH, BATTERY_MAX_KWH)] * (n + 1))
        bounds.extend([(0.0, None)] * n)
        bounds.extend([(0.0, None)] * future_n)
        bounds.extend([(0.0, None), (0.0, None)])
    bounds.append((0.0, None))

    solution = linprog(
        objective,
        A_ub=upper_matrix,
        b_ub=ub_rhs,
        A_eq=equality_matrix,
        b_eq=eq_rhs,
        bounds=bounds,
        method="highs-ds",
    )
    if not solution.success:
        raise RuntimeError(f"三日风险日前线性规划失败: {solution.message}")

    values = solution.x
    first_day_emergency_costs = np.asarray(
        [
            np.dot(
                EMERGENCY_PRICE_MULTIPLIER * (
                    daily_price
                    if scenario_price_flat is None
                    else scenario_price_flat[scenario, :PERIODS]
                ),
                values[
                    variable(scenario, "emergency") :
                    variable(scenario, "emergency") + PERIODS
                ],
            )
            for scenario in range(scenario_count)
        ],
        dtype=float,
    )
    horizon_emergency_costs = np.asarray(
        [
            np.dot(
                EMERGENCY_PRICE_MULTIPLIER * (
                    price_horizon
                    if scenario_price_flat is None
                    else scenario_price_flat[scenario]
                ),
                values[
                    variable(scenario, "emergency") :
                    variable(scenario, "emergency") + n
                ],
            )
            for scenario in range(scenario_count)
        ],
        dtype=float,
    )
    future_plan_costs = np.zeros(scenario_count, dtype=float)
    if future_n:
        future_plan_costs = np.asarray(
            [
                np.dot(
                    (
                        price_horizon[PERIODS:]
                        if scenario_price_flat is None
                        else scenario_price_flat[scenario, PERIODS:]
                    ),
                    values[
                        variable(scenario, "future_grid") :
                        variable(scenario, "future_grid") + future_n
                    ],
                )
                for scenario in range(scenario_count)
            ],
            dtype=float,
        )
    first_day_terminal = np.asarray(
        [values[variable(scenario, "soc", PERIODS)] for scenario in range(scenario_count)]
    )
    first_day_charge = np.asarray(
        [
            values[
                variable(scenario, "charge") : variable(scenario, "charge") + PERIODS
            ]
            for scenario in range(scenario_count)
        ],
        dtype=float,
    )
    first_day_discharge = np.asarray(
        [
            values[
                variable(scenario, "discharge") : variable(scenario, "discharge") + PERIODS
            ]
            for scenario in range(scenario_count)
        ],
        dtype=float,
    )
    first_day_soc = np.asarray(
        [
            values[
                variable(scenario, "soc") : variable(scenario, "soc") + PERIODS + 1
            ]
            for scenario in range(scenario_count)
        ],
        dtype=float,
    )
    horizon_terminal = np.asarray(
        [values[variable(scenario, "soc", n)] for scenario in range(scenario_count)]
    )
    terminal_deficit = np.asarray(
        [values[variable(scenario, "terminal_deficit")] for scenario in range(scenario_count)]
    )
    cvar = float(
        values[eta_index]
        + np.dot(
            probabilities,
            values[[variable(scenario, "xi") for scenario in range(scenario_count)]],
        )
        / (1.0 - alpha)
    )
    current_plan_cost = float(np.dot(daily_price, values[:PERIODS]))
    return {
        "purchase_kwh": values[:PERIODS].copy(),
        "planned_cost_yuan": current_plan_cost,
        "expected_emergency_cost_yuan": float(
            np.dot(probabilities, first_day_emergency_costs)
        ),
        "horizon_expected_plan_cost_yuan": float(
            current_plan_cost + np.dot(probabilities, future_plan_costs)
        ),
        "horizon_expected_emergency_cost_yuan": float(
            np.dot(probabilities, horizon_emergency_costs)
        ),
        "cvar_emergency_cost_yuan": cvar,
        "expected_first_day_terminal_soc_kwh": float(
            np.dot(probabilities, first_day_terminal)
        ),
        "expected_first_day_charge_kwh": np.average(
            first_day_charge, axis=0, weights=probabilities
        ),
        "expected_first_day_discharge_kwh": np.average(
            first_day_discharge, axis=0, weights=probabilities
        ),
        "expected_first_day_soc_kwh": np.average(
            first_day_soc, axis=0, weights=probabilities
        ),
        "expected_terminal_soc_kwh": float(
            np.dot(probabilities, horizon_terminal)
        ),
        "expected_terminal_shortfall_kwh": float(
            np.dot(probabilities, terminal_deficit)
        ),
        "objective_value_yuan": float(solution.fun),
        "scenario_count": int(scenario_count),
        "nonanticipativity_constraint_count": int(
            len(nonanticipativity_constraints)
        ),
        "nonanticipativity_tolerance_ratio": float(
            NONANTICIPATIVITY_TOLERANCE_RATIO
        ),
        "planning_horizon_days": int(horizon_days),
    }


def _charge_bounds(load_kwh: np.ndarray, baseline_kwh: np.ndarray) -> list[tuple[float, float]]:
    """Allow charging only when planned purchase plus PV covers the load."""
    surplus = baseline_kwh >= load_kwh - 1e-9
    upper = np.where(surplus, BATTERY_POWER_KWH, 0.0)
    return [(0.0, float(value)) for value in upper]


def _emergency_bounds(
    load_kwh: np.ndarray, baseline_kwh: np.ndarray
) -> list[tuple[float, float]]:
    """Allow emergency purchase only in baseline shortage periods."""
    shortage = baseline_kwh < load_kwh - 1e-9
    upper = np.where(shortage, np.maximum(load_kwh, 0.0), 0.0)
    return [(0.0, float(value)) for value in upper]


def solve_real_time_recourse(
    price: np.ndarray,
    load_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    planned_purchase_kwh: np.ndarray,
    initial_soc_kwh: float,
) -> dict[str, np.ndarray | float]:
    """Perfect-information recourse benchmark after the full PV curve is known.

    This function is intentionally kept as a lower-bound benchmark. It must not
    be used as the operational policy because it can use future actual PV.
    """
    n = PERIODS
    if not BATTERY_MIN_KWH - 1e-9 <= initial_soc_kwh <= BATTERY_MAX_KWH + 1e-9:
        raise ValueError("实时修正模型的初始储能量超出允许范围")
    charge = slice(0, n)
    discharge = slice(n, 2 * n)
    curtailment = slice(2 * n, 3 * n)
    soc = slice(3 * n, 3 * n + n + 1)
    emergency = slice(4 * n + 1, 5 * n + 1)
    variable_count = 5 * n + 1

    objective = np.zeros(variable_count)
    objective[emergency] = EMERGENCY_PRICE_MULTIPLIER * price
    objective[charge] = 1e-9
    objective[discharge] = 1e-9

    rows = 2 * n + 1
    matrix = np.zeros((rows, variable_count), dtype=float)
    rhs = np.zeros(rows)
    row_index = 0
    for t in range(n):
        # planned purchase + actual PV + discharge + emergency
        # = load + charge + curtailment
        matrix[row_index, charge.start + t] = -1.0
        matrix[row_index, discharge.start + t] = 1.0
        matrix[row_index, curtailment.start + t] = -1.0
        matrix[row_index, emergency.start + t] = 1.0
        rhs[row_index] = load_kwh[t] - planned_purchase_kwh[t] - actual_pv_kwh[t]
        row_index += 1

        matrix[row_index, soc.start + t + 1] = 1.0
        matrix[row_index, soc.start + t] = -1.0
        matrix[row_index, charge.start + t] = -ETA
        matrix[row_index, discharge.start + t] = 1.0 / ETA
        row_index += 1

    matrix[row_index, soc.start] = 1.0
    rhs[row_index] = initial_soc_kwh

    baseline = planned_purchase_kwh + actual_pv_kwh
    bounds = (
        _charge_bounds(load_kwh, baseline)
        + [(0.0, BATTERY_POWER_KWH)] * n
        + [(0.0, None)] * n
        + [(BATTERY_MIN_KWH, BATTERY_MAX_KWH)] * (n + 1)
        + _emergency_bounds(load_kwh, baseline)
    )
    power_ub = np.zeros((n, variable_count), dtype=float)
    for t in range(n):
        power_ub[t, charge.start + t] = 1.0
        power_ub[t, discharge.start + t] = 1.0
    solution = linprog(
        objective,
        A_ub=power_ub,
        b_ub=np.full(n, BATTERY_POWER_KWH),
        A_eq=matrix,
        b_eq=rhs,
        bounds=bounds,
        method="highs-ds",
    )
    if not solution.success:
        raise RuntimeError(f"实时修正线性规划失败: {solution.message}")

    values = solution.x
    emergency_values = values[emergency].copy()
    balance = (
        planned_purchase_kwh
        + actual_pv_kwh
        + values[discharge]
        + emergency_values
        - load_kwh
        - values[charge]
        - values[curtailment]
    )
    return {
        "charge_kwh": values[charge].copy(),
        "discharge_kwh": values[discharge].copy(),
        "curtailment_kwh": values[curtailment].copy(),
        "soc_kwh": values[soc].copy(),
        "emergency_kwh": emergency_values,
        "emergency_cost_yuan": float(
            np.dot(EMERGENCY_PRICE_MULTIPLIER * price, emergency_values)
        ),
        "max_balance_error_kwh": float(np.max(np.abs(balance))),
    }


def solve_causal_recourse(
    price: np.ndarray,
    load_kwh: np.ndarray,
    forecast_pv_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    planned_purchase_kwh: np.ndarray,
    initial_soc_kwh: float,
    scenario_pv_kwh: np.ndarray | None = None,
    scenario_probabilities: np.ndarray | None = None,
    terminal_soc_target_kwh: float | None = None,
    terminal_shortfall_penalty_yuan_per_kwh: float = 0.0,
    objective_price_kwh: np.ndarray | None = None,
    settlement_price_kwh: np.ndarray | None = None,
    observed_price_kwh: np.ndarray | None = None,
    scenario_price_kwh: np.ndarray | None = None,
    reference_charge_kwh: np.ndarray | None = None,
    reference_discharge_kwh: np.ndarray | None = None,
    trajectory_tracking_penalty_yuan_per_kwh: float = TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH,
) -> dict[str, np.ndarray | float]:
    """Execute the plan with causal rolling-horizon LP control.

    At the beginning of interval ``t`` only the current actual PV value is
    revealed. Future intervals use a soft likelihood-weighted scenario
    forecast. The LP is solved again after each interval, while the day-ahead
    grid purchase remains fixed.
    """
    n = PERIODS
    # ``price`` remains the backwards-compatible single price input.  Q4
    # supplies a causal decision-price curve and the realised tariff
    # separately: the former is allowed in future MPC coefficients, while the
    # latter is used only for the ex-post emergency settlement.
    decision_price = (
        np.asarray(price, dtype=float)
        if objective_price_kwh is None
        else np.asarray(objective_price_kwh, dtype=float)
    )
    realised_price = (
        np.asarray(price, dtype=float)
        if settlement_price_kwh is None
        else np.asarray(settlement_price_kwh, dtype=float)
    )
    observed_price = (
        realised_price
        if observed_price_kwh is None
        else np.asarray(observed_price_kwh, dtype=float)
    )
    if (
        decision_price.shape != (n,)
        or realised_price.shape != (n,)
        or observed_price.shape != (n,)
        or not np.isfinite(decision_price).all()
        or not np.isfinite(realised_price).all()
        or not np.isfinite(observed_price).all()
        or np.any(decision_price < 0.0)
        or np.any(realised_price < 0.0)
        or np.any(observed_price < 0.0)
    ):
        raise ValueError("决策价格和结算价格必须是长度144的有限非负向量")
    if not (
        len(price)
        == len(load_kwh)
        == len(forecast_pv_kwh)
        == len(actual_pv_kwh)
        == len(planned_purchase_kwh)
        == n
    ):
        raise ValueError("滚动修正输入必须包含144个时段")
    tracking_enabled = reference_charge_kwh is not None or reference_discharge_kwh is not None
    if tracking_enabled:
        if reference_charge_kwh is None or reference_discharge_kwh is None:
            raise ValueError("日前储能参考轨迹必须同时提供充电和放电向量")
        reference_charge_kwh = np.asarray(reference_charge_kwh, dtype=float)
        reference_discharge_kwh = np.asarray(reference_discharge_kwh, dtype=float)
        if (
            reference_charge_kwh.shape != (n,)
            or reference_discharge_kwh.shape != (n,)
            or not np.isfinite(reference_charge_kwh).all()
            or not np.isfinite(reference_discharge_kwh).all()
            or np.any(reference_charge_kwh < 0.0)
            or np.any(reference_discharge_kwh < 0.0)
        ):
            raise ValueError("日前储能参考轨迹必须是长度144的有限非负向量")
        if trajectory_tracking_penalty_yuan_per_kwh < 0.0:
            raise ValueError("储能轨迹跟踪惩罚不得为负")
    if not BATTERY_MIN_KWH - 1e-9 <= initial_soc_kwh <= BATTERY_MAX_KWH + 1e-9:
        raise ValueError("滚动修正模型的初始储能量超出允许范围")
    if terminal_soc_target_kwh is None:
        terminal_soc_target_kwh = BATTERY_MIN_KWH
    if not BATTERY_MIN_KWH <= terminal_soc_target_kwh <= BATTERY_MAX_KWH:
        raise ValueError("滚动修正模型的日末储能目标超出允许范围")
    if terminal_shortfall_penalty_yuan_per_kwh < 0.0:
        raise ValueError("滚动修正模型的日末储能缺口惩罚不得为负")
    if scenario_pv_kwh is not None:
        scenario_pv_kwh = np.asarray(scenario_pv_kwh, dtype=float)
        if scenario_pv_kwh.ndim != 2 or scenario_pv_kwh.shape[1] != n:
            raise ValueError("滚动修正场景光伏矩阵必须为(m,144)")
        if not np.isfinite(scenario_pv_kwh).all() or np.any(scenario_pv_kwh < 0.0):
            raise ValueError("滚动修正场景光伏必须为有限非负数")

    if scenario_price_kwh is not None:
        scenario_price_kwh = np.asarray(scenario_price_kwh, dtype=float)
        if scenario_price_kwh.ndim != 2 or scenario_price_kwh.shape[1] != n:
            raise ValueError("滚动修正场景价格矩阵必须为(m,144)")
        if not np.isfinite(scenario_price_kwh).all() or np.any(scenario_price_kwh < 0.0):
            raise ValueError("滚动修正场景价格必须为有限非负数")

    scenario_count = 0
    if scenario_pv_kwh is not None:
        scenario_count = scenario_pv_kwh.shape[0]
    if scenario_price_kwh is not None:
        if scenario_count and scenario_price_kwh.shape[0] != scenario_count:
            raise ValueError("滚动修正场景光伏与价格数量不一致")
        scenario_count = scenario_price_kwh.shape[0]
    if scenario_count:
        if scenario_probabilities is None:
            scenario_probabilities = np.full(
                scenario_count, 1.0 / scenario_count
            )
        else:
            scenario_probabilities = np.asarray(scenario_probabilities, dtype=float)
        if scenario_probabilities.shape != (scenario_count,) or np.any(scenario_probabilities < 0):
            raise ValueError("滚动修正场景概率维度或取值错误")
        if scenario_probabilities.sum() <= 0:
            raise ValueError("滚动修正场景概率和必须为正")
        scenario_probabilities = scenario_probabilities / scenario_probabilities.sum()
        log_weights = np.log(np.maximum(scenario_probabilities, 1e-300))

    charge_result = np.zeros(n)
    discharge_result = np.zeros(n)
    curtailment_result = np.zeros(n)
    soc_result = np.zeros(n + 1)
    emergency_result = np.zeros(n)
    soc_result[0] = initial_soc_kwh
    soc_current = initial_soc_kwh

    for t in range(n):
        remaining = n - t
        # Current actual PV is observed. Future PV uses a soft update of the
        # historical scenarios; exact scenario matching is not required.
        normalized = None
        if scenario_count:
            if scenario_pv_kwh is not None:
                residuals = scenario_pv_kwh[:, t] - forecast_pv_kwh[t]
                observed_residual = actual_pv_kwh[t] - forecast_pv_kwh[t]
                scale = float(
                    np.median(np.abs(residuals - np.median(residuals))) + 1e-3
                )
                log_weights += -0.5 * ((residuals - observed_residual) / scale) ** 2
            if scenario_price_kwh is not None:
                residuals = scenario_price_kwh[:, t] - decision_price[t]
                observed_residual = observed_price[t] - decision_price[t]
                scale = float(
                    np.median(np.abs(residuals - np.median(residuals))) + 1e-4
                )
                log_weights += -0.5 * ((residuals - observed_residual) / scale) ** 2
            normalized = np.exp(log_weights - np.max(log_weights))
            normalized /= normalized.sum()

        if scenario_pv_kwh is None:
            pv_for_horizon = forecast_pv_kwh[t:].copy()
        else:
            pv_for_horizon = np.average(
                scenario_pv_kwh[:, t:], axis=0, weights=normalized
            )
        pv_for_horizon[0] = actual_pv_kwh[t]

        charge = slice(0, remaining)
        discharge = slice(remaining, 2 * remaining)
        curtailment = slice(2 * remaining, 3 * remaining)
        soc = slice(3 * remaining, 4 * remaining + 1)
        emergency = slice(4 * remaining + 1, 5 * remaining + 1)
        terminal_deficit = 5 * remaining + 1
        if tracking_enabled:
            charge_dev_pos = slice(terminal_deficit + 1, terminal_deficit + 1 + remaining)
            charge_dev_neg = slice(charge_dev_pos.stop, charge_dev_pos.stop + remaining)
            discharge_dev_pos = slice(charge_dev_neg.stop, charge_dev_neg.stop + remaining)
            discharge_dev_neg = slice(discharge_dev_pos.stop, discharge_dev_pos.stop + remaining)
            variable_count = discharge_dev_neg.stop
        else:
            variable_count = terminal_deficit + 1

        objective = np.zeros(variable_count)
        # The release-specific decision price already contains the calibrated
        # upper-tail premium.  Price scenarios update their likelihood as new
        # observations arrive, but must not silently replace this risk-aware
        # MPC coefficient with an unprotected conditional mean.
        control_price = decision_price[t:].copy()
        # The current interval's realised price is available at execution
        # time; all later coefficients remain the causal forecast.
        control_price[0] = observed_price[t]
        objective[emergency] = EMERGENCY_PRICE_MULTIPLIER * control_price
        # A small throughput cost removes degenerate charge/discharge
        # cycling while remaining negligible relative to tariff costs.
        objective[charge] = THROUGHPUT_PENALTY_YUAN_PER_KWH
        objective[discharge] = THROUGHPUT_PENALTY_YUAN_PER_KWH
        objective[terminal_deficit] = terminal_shortfall_penalty_yuan_per_kwh
        if tracking_enabled:
            for dev in (charge_dev_pos, charge_dev_neg, discharge_dev_pos, discharge_dev_neg):
                objective[dev] = trajectory_tracking_penalty_yuan_per_kwh

        extra_tracking_rows = 2 * remaining if tracking_enabled else 0
        matrix = np.zeros((2 * remaining + 1 + extra_tracking_rows, variable_count), dtype=float)
        rhs = np.zeros(2 * remaining + 1 + extra_tracking_rows)
        row_index = 0
        for k in range(remaining):
            matrix[row_index, charge.start + k] = -1.0
            matrix[row_index, discharge.start + k] = 1.0
            matrix[row_index, curtailment.start + k] = -1.0
            matrix[row_index, emergency.start + k] = 1.0
            rhs[row_index] = (
                load_kwh[t + k]
                - planned_purchase_kwh[t + k]
                - pv_for_horizon[k]
            )
            row_index += 1

            matrix[row_index, soc.start + k + 1] = 1.0
            matrix[row_index, soc.start + k] = -1.0
            matrix[row_index, charge.start + k] = -ETA
            matrix[row_index, discharge.start + k] = 1.0 / ETA
            row_index += 1

        matrix[row_index, soc.start] = 1.0
        rhs[row_index] = soc_current
        row_index += 1
        if tracking_enabled:
            for k in range(remaining):
                matrix[row_index, charge.start + k] = 1.0
                matrix[row_index, charge_dev_pos.start + k] = -1.0
                matrix[row_index, charge_dev_neg.start + k] = 1.0
                rhs[row_index] = reference_charge_kwh[t + k]
                row_index += 1
                matrix[row_index, discharge.start + k] = 1.0
                matrix[row_index, discharge_dev_pos.start + k] = -1.0
                matrix[row_index, discharge_dev_neg.start + k] = 1.0
                rhs[row_index] = reference_discharge_kwh[t + k]
                row_index += 1

        baseline = planned_purchase_kwh[t:] + pv_for_horizon
        bounds = (
            _charge_bounds(load_kwh[t:], baseline)
            + [(0.0, BATTERY_POWER_KWH)] * remaining
            # Curtailment is discarded PV plus unused scheduled supply.  It
            # cannot exceed the available PV plus scheduled purchase in the
            # interval; the explicit bound prevents the recourse LP from
            # using an unphysical arbitrary spill variable.
            + [(0.0, float(value)) for value in baseline]
            + [(BATTERY_MIN_KWH, BATTERY_MAX_KWH)] * (remaining + 1)
            + _emergency_bounds(load_kwh[t:], baseline)
            + [(0.0, None)]
        )
        if tracking_enabled:
            bounds += [(0.0, None)] * (4 * remaining)
        power_ub = np.zeros((remaining + 1, variable_count), dtype=float)
        for k in range(remaining):
            power_ub[k, charge.start + k] = 1.0
            power_ub[k, discharge.start + k] = 1.0
        power_ub[remaining, soc.start + remaining] = -1.0
        power_ub[remaining, terminal_deficit] = -1.0
        upper_rhs = np.append(
            np.full(remaining, BATTERY_POWER_KWH), -terminal_soc_target_kwh
        )
        solution = linprog(
            objective,
            A_ub=power_ub,
            b_ub=upper_rhs,
            A_eq=matrix,
            b_eq=rhs,
            bounds=bounds,
            method="highs-ds",
        )
        if not solution.success:
            raise RuntimeError(f"第{t + 1}时段滚动修正线性规划失败: {solution.message}")

        values = solution.x
        charge_result[t] = values[charge.start]
        discharge_result[t] = values[discharge.start]
        curtailment_result[t] = values[curtailment.start]
        emergency_result[t] = values[emergency.start]
        soc_result[t + 1] = values[soc.start + 1]
        soc_current = soc_result[t + 1]

    balance = (
        planned_purchase_kwh
        + actual_pv_kwh
        + discharge_result
        + emergency_result
        - load_kwh
        - charge_result
        - curtailment_result
    )
    return {
        "charge_kwh": charge_result,
        "discharge_kwh": discharge_result,
        "curtailment_kwh": curtailment_result,
        "soc_kwh": soc_result,
        "emergency_kwh": emergency_result,
        "emergency_cost_yuan": float(
            np.dot(EMERGENCY_PRICE_MULTIPLIER * realised_price, emergency_result)
        ),
        "terminal_soc_target_kwh": float(terminal_soc_target_kwh),
        "terminal_soc_shortfall_kwh": float(
            max(terminal_soc_target_kwh - soc_result[-1], 0.0)
        ),
        "max_balance_error_kwh": float(np.max(np.abs(balance))),
        "trajectory_tracking_enabled": bool(tracking_enabled),
    }


def clean_emergency_for_reporting(emergency_kwh: np.ndarray) -> np.ndarray:
    """Remove solver-scale emergency purchases from the reporting view."""
    values = np.asarray(emergency_kwh, dtype=float)
    return np.where(values >= REPORTING_TOLERANCE_KWH, values, 0.0)


def contiguous_emergency_intervals(
    emergency_kwh: np.ndarray,
) -> list[dict[str, float | str]]:
    """Compress reportable emergency values into contiguous result intervals."""
    labels = interval_labels()
    intervals: list[dict[str, float | str]] = []
    start: int | None = None
    amount = 0.0
    for t, value in enumerate(emergency_kwh):
        if value >= REPORTING_TOLERANCE_KWH:
            if start is None:
                start = t
                amount = 0.0
            amount += float(value)
        elif start is not None:
            intervals.append(
                {"time_period": f"{labels[start].split('-')[0]}-{labels[t - 1].split('-')[1]}", "purchase_kwh": amount}
            )
            start = None
    if start is not None:
        intervals.append(
            {"time_period": f"{labels[start].split('-')[0]}-{labels[-1].split('-')[1]}", "purchase_kwh": amount}
        )
    return intervals


def apply_emergency_reporting_policy(result: dict, price: np.ndarray) -> dict:
    """Add an auditable reporting view without changing the raw LP solution.

    HiGHS may return nonzero primal values at approximately its feasibility
    tolerance.  Values below 0.001 kWh per ten-minute interval are retained in
    ``emergency_kwh_raw`` for balance audits and removed from the official
    report, its interval aggregation, and its reported costs.  The operation
    is idempotent so it can also normalize an existing annual result file.
    """
    price = np.asarray(price, dtype=float)
    if price.shape != (PERIODS,):
        raise ValueError("紧急购电报告口径需要144个时段电价")

    planned_total = 0.0
    emergency_total = 0.0
    raw_emergency_total = 0.0
    emergency_energy = 0.0
    raw_emergency_energy = 0.0
    emergency_day_count = 0
    for day in result.get("days", []):
        raw_values = np.asarray(
            day.get("emergency_kwh_raw", day["emergency_kwh"]), dtype=float
        )
        if raw_values.shape != (PERIODS,):
            raise ValueError(f"{day['date']}紧急购电数组长度不为144")
        reported_values = clean_emergency_for_reporting(raw_values)
        planned_cost = float(day["planned_cost_yuan"])
        raw_emergency_cost = float(
            np.dot(EMERGENCY_PRICE_MULTIPLIER * price, raw_values)
        )
        reported_emergency_cost = float(
            np.dot(EMERGENCY_PRICE_MULTIPLIER * price, reported_values)
        )

        day["emergency_kwh_raw"] = raw_values.tolist()
        day["emergency_kwh"] = reported_values.tolist()
        day["emergency_intervals"] = contiguous_emergency_intervals(reported_values)
        day["raw_emergency_cost_yuan"] = raw_emergency_cost
        day["raw_total_cost_yuan"] = planned_cost + raw_emergency_cost
        day["emergency_cost_yuan"] = reported_emergency_cost
        day["total_cost_yuan"] = planned_cost + reported_emergency_cost

        planned_total += planned_cost
        emergency_total += reported_emergency_cost
        raw_emergency_total += raw_emergency_cost
        emergency_energy += float(reported_values.sum())
        raw_emergency_energy += float(raw_values.sum())
        emergency_day_count += bool(np.any(reported_values > 0.0))

    result.setdefault("model", {})["emergency_reporting_tolerance_kwh"] = (
        REPORTING_TOLERANCE_KWH
    )
    summary = result.setdefault("summary", {})
    summary.update(
        {
            "day_count": len(result.get("days", [])),
            "planned_cost_yuan": planned_total,
            "emergency_cost_yuan": emergency_total,
            "total_cost_yuan": planned_total + emergency_total,
            "emergency_energy_kwh": emergency_energy,
            "emergency_day_count": emergency_day_count,
            "raw_emergency_cost_yuan": raw_emergency_total,
            "raw_total_cost_yuan": planned_total + raw_emergency_total,
            "raw_emergency_energy_kwh": raw_emergency_energy,
            "filtered_numerical_noise_cost_yuan": raw_emergency_total
            - emergency_total,
            "filtered_numerical_noise_kwh": raw_emergency_energy
            - emergency_energy,
        }
    )
    return result


def solve_one_day(
    price: np.ndarray,
    current_date: date,
    load_horizon_kwh: np.ndarray,
    actual_pv_kwh: np.ndarray,
    forecast_pv_horizon_kwh: np.ndarray,
    scenario_pv_horizon_kwh: np.ndarray,
    scenario_probabilities: np.ndarray,
    initial_soc_kwh: float,
    forecast_source_count: int,
    forecast_source_start: date | None,
    forecast_source_end: date | None,
    forecast_selection: ForecastSelection,
    raw_error_history: np.ndarray,
) -> dict:
    """Solve a multi-day plan and causally execute its first day."""
    load_horizon = np.asarray(load_horizon_kwh, dtype=float)
    forecast_horizon = np.asarray(forecast_pv_horizon_kwh, dtype=float)
    scenario_horizon = np.asarray(scenario_pv_horizon_kwh, dtype=float)
    if load_horizon.ndim != 2 or load_horizon.shape[1] != PERIODS:
        raise ValueError("滚动计划负荷必须为(H,144)矩阵")
    terminal_shortfall_penalty = float(np.mean(price) / ETA)
    plan = solve_multiday_risk_aware_plan(
        price,
        load_horizon,
        forecast_horizon,
        scenario_horizon,
        scenario_probabilities,
        initial_soc_kwh,
        terminal_shortfall_penalty,
    )
    terminal_target = float(
        np.clip(
            plan["expected_first_day_terminal_soc_kwh"],
            BATTERY_MIN_KWH,
            BATTERY_MAX_KWH,
        )
    )
    actual = solve_causal_recourse(
        price,
        load_horizon[0],
        forecast_horizon[0],
        actual_pv_kwh,
        plan["purchase_kwh"],
        initial_soc_kwh,
        scenario_horizon[:, 0, :],
        scenario_probabilities,
        terminal_target,
        terminal_shortfall_penalty,
        reference_charge_kwh=plan.get("expected_first_day_charge_kwh"),
        reference_discharge_kwh=plan.get("expected_first_day_discharge_kwh"),
    )
    if actual["max_balance_error_kwh"] > 1e-7:
        raise AssertionError(f"{current_date}实际供需平衡校验失败")
    if np.min(actual["soc_kwh"]) < BATTERY_MIN_KWH - 1e-7:
        raise AssertionError(f"{current_date}储能下限校验失败")
    if np.max(actual["soc_kwh"]) > BATTERY_MAX_KWH + 1e-7:
        raise AssertionError(f"{current_date}储能上限校验失败")

    shortage_history = np.maximum(-raw_error_history, 0.0)
    if shortage_history.size:
        shortage_probability = np.mean(shortage_history > 1e-9, axis=0)
        shortage_cvar = empirical_shortage_cvar(shortage_history)
    else:
        shortage_probability = np.zeros(PERIODS, dtype=float)
        shortage_cvar = np.zeros(PERIODS, dtype=float)
    forecast_pv_kwh = forecast_horizon[0]
    risk_index = shortage_cvar / np.maximum(forecast_pv_kwh, 1e-6)
    planned_purchase = plan["purchase_kwh"]
    emergency = actual["emergency_kwh"]
    result = {
        "date": current_date.isoformat(),
        "forecast_source_count": int(forecast_source_count),
        "forecast_source_start": forecast_source_start.isoformat() if forecast_source_start else None,
        "forecast_source_end": forecast_source_end.isoformat() if forecast_source_end else None,
        "forecast_selection": {
            "method": "cold_start" if forecast_selection.used_cold_start else "fourier_ridge",
            "harmonics": int(forecast_selection.harmonics),
            "ridge": float(forecast_selection.ridge),
            "validation_loss": forecast_selection.validation_loss,
            "validation_days": int(forecast_selection.validation_days),
        },
        "forecast_pv_kwh": forecast_pv_kwh.tolist(),
        "actual_pv_kwh": actual_pv_kwh.tolist(),
        "load_kwh": load_horizon[0].tolist(),
        "planned_purchase_kwh": planned_purchase.tolist(),
        "charge_kwh": actual["charge_kwh"].tolist(),
        "discharge_kwh": actual["discharge_kwh"].tolist(),
        "day_ahead_reference_charge_kwh": plan["expected_first_day_charge_kwh"].tolist(),
        "day_ahead_reference_discharge_kwh": plan["expected_first_day_discharge_kwh"].tolist(),
        "soc_kwh": actual["soc_kwh"].tolist(),
        "emergency_kwh": emergency.tolist(),
        "emergency_intervals": contiguous_emergency_intervals(emergency),
        "planned_cost_yuan": float(plan["planned_cost_yuan"]),
        "expected_emergency_cost_yuan": float(plan["expected_emergency_cost_yuan"]),
        "cvar_emergency_cost_yuan": float(plan["cvar_emergency_cost_yuan"]),
        "cvar_active": bool(
            plan["cvar_emergency_cost_yuan"]
            > plan["horizon_expected_emergency_cost_yuan"] + 1e-7
        ),
        "stress_scenario_added": True,
        "horizon_expected_plan_cost_yuan": float(
            plan["horizon_expected_plan_cost_yuan"]
        ),
        "horizon_expected_emergency_cost_yuan": float(
            plan["horizon_expected_emergency_cost_yuan"]
        ),
        "planning_horizon_days": int(plan["planning_horizon_days"]),
        "nonanticipativity_constraint_count": int(
            plan["nonanticipativity_constraint_count"]
        ),
        "nonanticipativity_tolerance_ratio": float(
            plan["nonanticipativity_tolerance_ratio"]
        ),
        "planned_terminal_soc_target_kwh": terminal_target,
        "expected_terminal_shortfall_kwh": float(
            plan["expected_terminal_shortfall_kwh"]
        ),
        "expected_terminal_soc_kwh": float(plan["expected_terminal_soc_kwh"]),
        "risk_objective_value_yuan": float(plan["objective_value_yuan"]),
        "scenario_count": int(plan["scenario_count"]),
        "raw_scenario_count": int(raw_error_history.shape[0]),
        "emergency_cost_yuan": float(actual["emergency_cost_yuan"]),
        "total_cost_yuan": float(
            plan["planned_cost_yuan"] + actual["emergency_cost_yuan"]
        ),
        "forecast_mae_kwh": float(np.mean(np.abs(forecast_pv_kwh - actual_pv_kwh))),
        "shortage_probability": shortage_probability.tolist(),
        "shortage_cvar_kwh": shortage_cvar.tolist(),
        "risk_index": risk_index.tolist(),
        "max_balance_error_kwh": float(actual["max_balance_error_kwh"]),
        "initial_soc_kwh": float(initial_soc_kwh),
        "terminal_soc_kwh": float(actual["soc_kwh"][-1]),
        "terminal_soc_shortfall_kwh": float(
            actual["terminal_soc_shortfall_kwh"]
        ),
    }
    return result


def solve_all_days(root: Path) -> dict:
    price, dates, load, actual_pv = read_inputs(root)
    attachment1 = load_processed_attachment1(root / "数据" / "processed_attachment1.csv")
    cold_start_forecast = attachment1["pv_kwh"].to_numpy(dtype=float)
    forecasts, valid_mask, selections = build_causal_forecast_history(
        actual_pv,
        cold_start_forecast,
        FORECAST_CONFIG,
    )
    multiday_forecasts = build_causal_multiday_forecast_history(
        actual_pv,
        cold_start_forecast,
        selections,
        PLANNING_HORIZON_DAYS,
        FORECAST_CONFIG,
    )
    valid_error_history = build_causal_error_history(
        actual_pv,
        forecasts,
        valid_mask,
    )
    valid_indices = np.flatnonzero(valid_mask)
    target_dates = [
        current_date
        for current_date in dates
        if OUTPUT_START <= current_date <= OUTPUT_END
    ]
    if len(target_dates) != 334:
        raise ValueError(f"问题二应输出334天，实际得到{len(target_dates)}天")

    date_to_index = {current_date: i for i, current_date in enumerate(dates)}
    target_set = set(target_dates)
    days: list[dict] = []
    soc_current = INITIAL_SOC_KWH
    warmup_terminal_soc = None

    # January is a warm-up period: it is simulated to propagate the only
    # specified initial state (2025-01-01 00:00) to the first output day.
    for current_date in dates:
        index = date_to_index[current_date]
        horizon_days = min(PLANNING_HORIZON_DAYS, len(dates) - index)
        load_horizon = load[index : index + horizon_days]
        forecast_horizon = multiday_forecasts[index, :horizon_days]
        history_count = int(np.searchsorted(valid_indices, index, side="left"))
        raw_error_history = valid_error_history[:history_count]
        multiday_error_history = build_multiday_error_history(
            actual_pv,
            multiday_forecasts,
            valid_mask,
            index,
            horizon_days,
        )
        selection = selections[index]
        if selection.used_cold_start:
            forecast_source_count = 0
            forecast_source_start = None
            forecast_source_end = None
        else:
            forecast_source_count = index
            forecast_source_start = dates[0]
            forecast_source_end = dates[index - 1]

        scenario_errors, scenario_probabilities = reduce_error_scenarios(
            multiday_error_history
        )
        scenario_errors, scenario_probabilities = append_pessimistic_error_scenario(
            scenario_errors,
            scenario_probabilities,
            multiday_error_history,
            forecast_horizon.reshape(-1),
            price,
        )
        scenario_pv = np.maximum(
            forecast_horizon.reshape(1, -1) + scenario_errors,
            0.0,
        ).reshape(-1, horizon_days, PERIODS)

        day = solve_one_day(
            price,
            current_date,
            load_horizon,
            actual_pv[index],
            forecast_horizon,
            scenario_pv,
            scenario_probabilities,
            soc_current,
            forecast_source_count,
            forecast_source_start,
            forecast_source_end,
            selection,
            raw_error_history,
        )
        soc_current = float(day["terminal_soc_kwh"])
        if current_date == OUTPUT_START - timedelta(days=1):
            warmup_terminal_soc = soc_current
        if current_date in target_set:
            if not days or day["date"] > days[-1]["date"]:
                days.append(day)

    planned_total = sum(day["planned_cost_yuan"] for day in days)
    emergency_total = sum(day["emergency_cost_yuan"] for day in days)
    emergency_energy = sum(sum(day["emergency_kwh"]) for day in days)
    result = {
        "model": {
            "name": "因果多步Fourier预测 + 三日误差场景 + CVaR三日滚动日前LP + 因果首日执行",
            "time_contract": "t=1,...,144；0:10对应0:10-0:20；功率乘1/6转为时段电量",
            "forecast_method": "截至目标日前的因果Fourier岭回归；历史不足时使用冷启动曲线",
            "fourier_harmonics": list(FORECAST_CONFIG.harmonics),
            "ridge_values": list(FORECAST_CONFIG.ridge_values),
            "min_history": FORECAST_CONFIG.min_history,
            "validation_window": FORECAST_CONFIG.validation_window,
            "overprediction_weight": FORECAST_CONFIG.overprediction_weight,
            "cold_start_source": "附件1光伏预测曲线",
            "scenario_reduction": "历史连续三日误差块按范数排序分组取均值，概率按组样本占比保持",
            "scenario_max_count": SCENARIO_MAX_COUNT,
            "cvar_stress_probability": CVAR_STRESS_PROBABILITY,
            "cvar_stress_tail_fraction": CVAR_STRESS_TAIL_FRACTION,
            "cvar_stress_fallback_shortfall_ratio": CVAR_STRESS_FALLBACK_SHORTFALL_RATIO,
            "cvar_tail_method": "按电价加权光伏短缺选择最差历史误差块；无历史时下调点预测",
            "planning_horizon_days": PLANNING_HORIZON_DAYS,
            "nonanticipativity_policy": "场景在观测历史未能区分前共享充放电和紧急购电决策；按预测光伏幅值5%作为观测等价容差",
            "nonanticipativity_tolerance_ratio": NONANTICIPATIVITY_TOLERANCE_RATIO,
            "cvar_alpha": CVAR_ALPHA,
            "risk_aversion_lambda": RISK_AVERSION_LAMBDA,
            "load_assumption": "每天0:00可获得当天及未来两天的完整负荷计划",
            "future_purchase_role": "未来两天购电量仅为受场景信息集约束的预览决策；只锁定并执行当前日计划，次日0:00重新规划",
            "recourse_information": "第t个时段仅使用当前实际光伏和此前信息，未来光伏由当日0:00历史场景软加权预测",
            "perfect_information_benchmark": "solve_real_time_recourse为全天实际光伏完全信息下界，不作为主策略",
        "recourse_solver": "因果滚动LP；日前场景LP增加非预见性约束，日内按基础供给是否覆盖负荷设置充电和紧急购电上界，并加入 c_t+d_t≤5000/6 与吞吐量微小惩罚",
            "trajectory_tracking": "执行层软跟踪日前场景概率加权的第一日充放电轨迹；实际光伏偏差时允许偏离并由物理约束和紧急供电优先",
            "trajectory_tracking_penalty_yuan_per_kwh": TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH,
            "emergency_price_multiplier": EMERGENCY_PRICE_MULTIPLIER,
            "time_step_hours": DT_HOURS,
            "battery_efficiency": ETA,
            "battery_min_kwh": BATTERY_MIN_KWH,
            "battery_max_kwh": BATTERY_MAX_KWH,
            "battery_power_kw": BATTERY_POWER_KW,
            "initial_soc_kwh": INITIAL_SOC_KWH,
            "state_boundary": "仅固定2025-01-01 0:00=6000 kWh；每天末端储能量连续传递到下一天，不设置每日6000 kWh终端约束",
            "terminal_policy": "取消每日固定末端电量奖励；三日窗口末只对低于窗口初始SOC的库存缺口按平均购电补充成本施加软惩罚，不奖励超额库存",
            "warmup_period": "2025-01-01至2025-01-31用于从初始储能状态传播到首个输出日；1月1日使用附件1光伏预测冷启动",
            "warmup_terminal_soc_kwh": warmup_terminal_soc,
            "output_period": "2025-02-01至2025-12-31",
            "terminal_shortfall_penalty_yuan_per_kwh": float(
                np.mean(price) / ETA
            ),
        },
        "interval_labels": interval_labels(),
        "days": days,
        "summary": {
            "day_count": len(days),
            "planned_cost_yuan": planned_total,
            "emergency_cost_yuan": emergency_total,
            "total_cost_yuan": planned_total + emergency_total,
            "emergency_energy_kwh": emergency_energy,
            "emergency_day_count": sum(
                any(value > 1e-7 for value in day["emergency_kwh"]) for day in days
            ),
            "cvar_active_day_count": sum(
                day["cvar_emergency_cost_yuan"]
                > day["horizon_expected_emergency_cost_yuan"] + 1e-7
                for day in days
            ),
            "mean_forecast_mae_kwh": float(np.mean([day["forecast_mae_kwh"] for day in days])),
        },
    }
    return apply_emergency_reporting_policy(result, price)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_path = root / "C题" / "结果" / "q2_solution.json"
    result = solve_all_days(root / "C题")
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
