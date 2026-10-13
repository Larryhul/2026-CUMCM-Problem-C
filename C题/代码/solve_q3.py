"""Problem 3: four-stage PV forecast updates with contract adjustment.

The model separates the 00:00 day-ahead contract ``P`` from the quantity
``A`` finally scheduled after the 06:00/12:00/18:00 forecast updates.  The
contract quantities are optimized with a linear program; actual operation is
then executed causally every ten minutes with a rolling storage LP.
"""

from __future__ import annotations

import json
import math
from copy import copy
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from forecast_price_q4 import release_first_index, release_slot_hour_index
import solve_q2

# Shared constants are kept explicit here to make the result auditable.
PERIODS = 144
DT_HOURS = 1.0 / 6.0
ETA = 0.90
BATTERY_MIN_KWH = 1200.0
BATTERY_MAX_KWH = 10800.0
BATTERY_POWER_KWH = 5000.0 * DT_HOURS
INITIAL_SOC_KWH = 6000.0
EMERGENCY_MULTIPLIER = 5.0
# The formal Q3 model uses the expected day-ahead terminal SOC as a soft
# continuation target.  The old terminal-value reward remains available only
# in archived result files; it is no longer used by the default solver.
TERMINAL_SOC_SHORTFALL_PENALTY_FACTOR = 1.0
DEFAULT_SETTLEMENT_MODE = "min_base"
DEFAULT_SCENARIO_REDUCTION_MODE = "norm"
DEFAULT_FORECAST_EXPANSION = "step"
# Small but numerically meaningful throughput cost.  Together with the
# aggregate power cap this makes curtailment preferable to artificial
# charge/discharge cycling in otherwise degenerate LP solutions.
THROUGHPUT_PENALTY_YUAN_PER_KWH = 1.0e-4
# The causal execution layer softly follows the storage trajectory selected by
# the 00:00 risk-aware plan.  A soft penalty preserves feasibility when actual
# PV differs from the day-ahead forecast.
TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH = 0.10
REPORTING_TOLERANCE_KWH = 1.0e-3
# HiGHS method is configurable for reproducible speed comparisons.
LP_METHOD = "highs-ds"
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)


def _validate_settlement_mode(mode: str) -> str:
    if mode not in {"min_base", "planned_plus_adjustment"}:
        raise ValueError(f"未知结算口径: {mode}")
    return mode


def _terminal_target(
    target_kwh: float | None,
    initial_soc_kwh: float,
) -> float:
    target = initial_soc_kwh if target_kwh is None else float(target_kwh)
    return float(np.clip(target, BATTERY_MIN_KWH, BATTERY_MAX_KWH))


def _terminal_penalty(price: np.ndarray, value_factor: float = 1.0) -> float:
    price = np.asarray(price, dtype=float)
    if price.size == 0:
        return 0.0
    return float(value_factor * np.mean(price) / ETA)


def interval_labels() -> list[str]:
    labels = []
    for i in range(PERIODS):
        start = (i + 1) * 10
        end = start + 10
        def fmt(m: int) -> str:
            day, rem = divmod(m, 1440)
            h, mm = divmod(rem, 60)
            return f"{h}:{mm:02d}{'+' + str(day) if day else ''}"
        labels.append(f"{fmt(start)}-{fmt(end)}")
    return labels


def load_inputs(root: Path):
    """Load price, annual load/PV and attachment-3 four-release forecasts."""
    from data_processing import load_attachment2, load_processed_attachment1

    a1 = load_processed_attachment1(root / "数据" / "processed_attachment1.csv")
    a2 = load_attachment2(root / "附件" / "附件2.xlsx")
    forecast = parse_attachment3(root / "附件" / "附件3.xlsx")
    return (
        a1["price"].to_numpy(float),
        list(a2.dates),
        a2.load_kwh,
        a2.pv_actual_kwh,
        forecast,
    )


def parse_attachment3(path: Path) -> dict[date, dict[int, np.ndarray]]:
    """Parse attachment 3, forward-filling the date shown once per day."""
    frame = pd.read_excel(path)
    if frame.shape[0] != 1460:
        raise ValueError(f"附件3应有1460条预测记录，实际{frame.shape[0]}")
    date_col, release_col = frame.columns[:2]
    pv_cols = list(frame.columns[2:26])
    if len(pv_cols) != 24:
        raise ValueError("附件3每条记录必须包含24个整点预测值")
    dates = pd.to_datetime(frame[date_col], errors="coerce").ffill()
    if dates.isna().any():
        raise ValueError("附件3日期存在无法解析的值")
    result: dict[date, dict[int, np.ndarray]] = {}
    for idx, row in frame.iterrows():
        day = dates.iloc[idx].date()
        rel_text = str(row[release_col]).strip().replace("：", ":")
        if ":" not in rel_text:
            raise ValueError(f"附件3预测时刻无法解析: {rel_text}")
        hh, mm = (int(x) for x in rel_text.split(":")[:2])
        if hh not in (0, 6, 12, 18) or mm != 0:
            raise ValueError(f"附件3预测时刻应为0/6/12/18时: {rel_text}")
        values = pd.to_numeric(row[pv_cols], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"附件3存在非法光伏预测值: {day} {rel_text}")
        result.setdefault(day, {})[hh] = values
    if len(result) != 365 or any(set(v) != {0, 6, 12, 18} for v in result.values()):
        raise ValueError("附件3必须覆盖365天且每天包含四个发布时间")
    return result


def expand_release_forecast(
    hourly_kw: np.ndarray,
    release_hour: int,
    method: str = DEFAULT_FORECAST_EXPANSION,
) -> np.ndarray:
    """Expand 24 hourly forecasts (kW) to 144 ten-minute kWh values.

    Forecast h is used for the six source rows whose timestamps run from
    ``release + h`` hours through ``release + (h + 1)`` hours.  The source row
    labelled ``h:00`` is the first row covered by a forecast released at
    ``h:00``; for later releases its zero-based index is ``6*h-1``.  Values
    outside the release's same-day future are left at zero and filled by the
    previous release in ``build_release_curves``.
    """
    hourly_kw = np.asarray(hourly_kw, dtype=float)
    if hourly_kw.shape != (24,) or np.any(hourly_kw < 0):
        raise ValueError("整点光伏预测必须是长度24的非负向量")
    if method not in {"step", "linear"}:
        raise ValueError(f"未知整点到十分钟展开方式: {method}")
    out = np.zeros(PERIODS, dtype=float)
    release_min = release_hour * 60
    if method == "linear":
        interval_starts = np.arange(PERIODS, dtype=float) * 10.0 + 10.0
        delta = interval_starts - release_min
        valid = delta >= 0.0
        centers = delta[valid] + 5.0
        hourly_centers = (np.arange(24, dtype=float) + 0.5) * 60.0
        raw = np.interp(
            centers,
            hourly_centers,
            hourly_kw,
            left=float(hourly_kw[0]),
            right=float(hourly_kw[-1]),
        )
        valid_indices = np.flatnonzero(valid)
        hour_indices = np.asarray(
            [release_slot_hour_index(int(t), release_hour) for t in valid_indices],
            dtype=int,
        )
        hour_indices = np.minimum(hour_indices, 23)
        for hour in range(24):
            selected_valid = hour_indices == hour
            if not np.any(selected_valid):
                continue
            selected = np.zeros(PERIODS, dtype=bool)
            selected[valid] = selected_valid
            local_raw = raw[selected_valid]
            raw_energy = float(np.sum(local_raw) * DT_HOURS)
            if raw_energy <= 1e-12:
                out[selected] = hourly_kw[hour] * DT_HOURS
            else:
                out[selected] = (
                    local_raw * (float(hourly_kw[hour]) / raw_energy) * DT_HOURS
                )
        return out
    for t in range(PERIODS):
        start_min = (t + 1) * 10
        if start_min < release_min:
            continue
        hour_index = release_slot_hour_index(t, release_hour)
        if hour_index is None:
            continue
        out[t] = hourly_kw[hour_index] * DT_HOURS
    return out


def build_release_curves(
    day_forecasts: dict[int, np.ndarray],
    margins: dict[int, np.ndarray] | None = None,
    expansion_method: str = DEFAULT_FORECAST_EXPANSION,
) -> dict[int, np.ndarray]:
    """Build four complete same-day curves, carrying the latest old forecast.

    ``margins`` is an optional causal lower-tail error correction in kWh per
    hourly forecast.  It is estimated only from dates strictly before the
    target date and is expanded with the same step function as the forecast.
    """
    curves: dict[int, np.ndarray] = {}
    current = np.zeros(PERIODS)
    for release in (0, 6, 12, 18):
        hourly = day_forecasts[release].copy()
        if margins is not None and release in margins:
            hourly = np.maximum(hourly + margins[release], 0.0)
        updated = expand_release_forecast(hourly, release, expansion_method)
        mask = np.arange(PERIODS) * 10 + 10 >= release * 60
        current = current.copy()
        current[mask] = updated[mask]
        curves[release] = current.copy()
    return curves


def build_release0_error_history(
    current_index: int,
    dates: list[date],
    actual_pv_kwh: np.ndarray,
    forecast_by_date: dict[date, dict[int, np.ndarray]],
    expansion_method: str = DEFAULT_FORECAST_EXPANSION,
) -> np.ndarray:
    """Return causal historical errors for the 00:00 forecast release."""
    if current_index < 0 or current_index > len(dates):
        raise ValueError("当前日期索引超出范围")
    errors: list[np.ndarray] = []
    for index in range(current_index):
        raw_curve = build_release_curves(
            forecast_by_date[dates[index]], expansion_method=expansion_method
        )[0]
        errors.append(np.asarray(actual_pv_kwh[index] - raw_curve, dtype=float))
    if not errors:
        return np.zeros((0, PERIODS), dtype=float)
    return np.vstack(errors)


def solve_q2_base_plan(
    price: np.ndarray,
    load: np.ndarray,
    forecast_00: np.ndarray,
    initial_soc: float,
    error_history: np.ndarray,
    scenario_reduction_mode: str = DEFAULT_SCENARIO_REDUCTION_MODE,
    cvar_alpha: float | None = None,
    risk_aversion_lambda: float | None = None,
    stress_probability: float | None = None,
    scenario_pv_horizon_kwh: np.ndarray | None = None,
    scenario_price_horizon_kwh: np.ndarray | None = None,
    scenario_probabilities: np.ndarray | None = None,
) -> dict:
    """Use Question 2's risk-aware LP for Question 3's initial contract.

    Question 3 only supplies a 24-hour forecast at 00:00, so this is the
    one-day special case of Question 2's rolling risk LP.  The risk layer is
    otherwise kept identical: reduced historical scenarios, a low-PV stress
    scenario, CVaR, and non-anticipative recourse.
    """
    if scenario_pv_horizon_kwh is None:
        scenario_errors, probabilities = solve_q2.reduce_error_scenarios(
            error_history,
            score_mode=scenario_reduction_mode,
            price=price,
        )
        scenario_errors, probabilities = solve_q2.append_pessimistic_error_scenario(
            scenario_errors,
            probabilities,
            error_history,
            forecast_00,
            price,
            stress_probability=stress_probability,
        )
        scenario_pv = np.maximum(forecast_00[None, :] + scenario_errors, 0.0)
    else:
        scenario_pv = np.asarray(scenario_pv_horizon_kwh, dtype=float)
        if scenario_pv.ndim == 3 and scenario_pv.shape[1:] == (1, PERIODS):
            scenario_pv = scenario_pv[:, 0, :]
        if scenario_pv.ndim != 2 or scenario_pv.shape[1] != PERIODS:
            raise ValueError("问题三基准场景光伏必须为(m,144)矩阵")
        if not np.isfinite(scenario_pv).all() or np.any(scenario_pv < 0.0):
            raise ValueError("问题三基准场景光伏必须为有限非负数")
        if scenario_probabilities is None:
            probabilities = np.full(scenario_pv.shape[0], 1.0 / scenario_pv.shape[0])
        else:
            probabilities = np.asarray(scenario_probabilities, dtype=float)
        if probabilities.shape != (scenario_pv.shape[0],) or np.any(probabilities < 0.0):
            raise ValueError("问题三基准场景概率维度或取值错误")
        if probabilities.sum() <= 0.0:
            raise ValueError("问题三基准场景概率和必须为正")
        probabilities = probabilities / probabilities.sum()
    # A one-day horizon is the information available in Question 3 at 00:00.
    # Using the multi-day solver with H=1 preserves Question 2's
    # non-anticipativity and terminal-inventory treatment without inventing
    # forecasts for dates that have not been released yet.
    plan = solve_q2.solve_multiday_risk_aware_plan(
        price,
        np.asarray(load, dtype=float)[None, :],
        np.asarray(forecast_00, dtype=float)[None, :],
        scenario_pv[:, None, :],
        probabilities,
        initial_soc,
        float(np.mean(price) / solve_q2.ETA),
        cvar_alpha=cvar_alpha,
        risk_aversion_lambda=risk_aversion_lambda,
        scenario_price_horizon_kwh=scenario_price_horizon_kwh,
    )
    # Scenario-wise optimal storage actions can average to simultaneous charge
    # and discharge in the probability-weighted reference.  Project that
    # reference onto the net SOC change while preserving its expected SOC
    # trajectory; the causal execution LP then follows a physically clean
    # charge-or-discharge reference without hard-locking feasibility.
    reference_charge = np.asarray(
        plan["expected_first_day_charge_kwh"], dtype=float
    )
    reference_discharge = np.asarray(
        plan["expected_first_day_discharge_kwh"], dtype=float
    )
    net_soc_change = ETA * reference_charge - reference_discharge / ETA
    plan["expected_first_day_charge_kwh"] = (
        np.maximum(net_soc_change, 0.0) / ETA
    )
    plan["expected_first_day_discharge_kwh"] = (
        np.maximum(-net_soc_change, 0.0) * ETA
    )
    plan["expected_first_day_soc_kwh"] = np.concatenate(
        ([float(initial_soc)], float(initial_soc) + np.cumsum(net_soc_change))
    )
    plan["scenario_count"] = int(len(probabilities))
    return plan


def build_release_scenario_curves(
    current_index: int,
    release: int,
    dates: list[date],
    actual_pv_kwh: np.ndarray,
    forecast_by_date: dict[date, dict[int, np.ndarray]],
    current_curve: np.ndarray,
    price: np.ndarray,
    margin_history: dict[date, dict[int, np.ndarray]] | None = None,
    expansion_method: str = DEFAULT_FORECAST_EXPANSION,
    scenario_reduction_mode: str = DEFAULT_SCENARIO_REDUCTION_MODE,
    stress_probability: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal scenario PV curves for a forecast-release execution block.

    Historical residuals are measured against the same release and the same
    correction convention as the current point forecast.  Values before the
    release are zeroed before scenario reduction so unrelated earlier errors
    cannot affect the distance-based grouping.
    """
    start = release_first_index(release)
    errors: list[np.ndarray] = []
    for index in range(current_index):
        historical_margins = None if release == 0 else (
            margin_history[dates[index]] if margin_history is not None else None
        )
        historical_curve = build_release_curves(
            forecast_by_date[dates[index]],
            historical_margins,
            expansion_method=expansion_method,
        )[release]
        residual = np.asarray(actual_pv_kwh[index] - historical_curve, dtype=float)
        residual[:start] = 0.0
        errors.append(residual)
    error_history = (
        np.vstack(errors) if errors else np.zeros((0, PERIODS), dtype=float)
    )
    scenario_errors, probabilities = solve_q2.reduce_error_scenarios(
        error_history,
        score_mode=scenario_reduction_mode,
        price=price,
    )
    scenario_errors, probabilities = solve_q2.append_pessimistic_error_scenario(
        scenario_errors,
        probabilities,
        error_history,
        current_curve,
        np.asarray(price, dtype=float),
        stress_probability=stress_probability,
    )
    scenario_curves = np.maximum(
        np.asarray(current_curve, dtype=float)[None, :] + scenario_errors, 0.0
    )
    return scenario_curves, probabilities


def causal_error_margins(
    current_date: date,
    dates: list[date],
    actual_pv_kwh: np.ndarray,
    forecast_by_date: dict[date, dict[int, np.ndarray]],
    quantile: float = 0.10,
    shrink: float = 0.70,
    min_history: int = 7,
    seasonal: bool = False,
) -> dict[int, np.ndarray]:
    """Estimate release/lead PV error quantiles using past dates only.

    Error is ``actual hourly energy - forecast hourly power`` (the numerical
    values are both kWh over one hour).  A lower-tail quantile gives a
    conservative forecast for shortage control; positive quantiles correct a
    systematic underprediction.  Sparse early history falls back to zero.
    """
    idx = {d: i for i, d in enumerate(dates)}
    all_past = [d for d in dates if d < current_date]
    past = all_past
    if seasonal:
        season = (current_date.month - 1) // 3
        seasonal_past = [d for d in all_past if (d.month - 1) // 3 == season]
        if len(seasonal_past) >= min_history:
            past = seasonal_past
    margins: dict[int, np.ndarray] = {}
    for release in (0, 6, 12, 18):
        lead_count = 24 - release
        if len(past) < min_history:
            margins[release] = np.zeros(24)
            continue
        values: list[list[float]] = [[] for _ in range(lead_count)]
        # Attachment-3's release-hour forecast first applies to the source row
        # labelled with the release time.  This is index 0 for 00:00 and
        # ``6*release-1`` for later releases under the shared data convention.
        first_interval = release_first_index(release)
        for d in past:
            f = forecast_by_date[d][release]
            pv = actual_pv_kwh[idx[d]]
            for h in range(lead_count):
                st = first_interval + h * 6
                values[h].append(float(pv[st:st + 6].sum() - f[h]))
        q = np.asarray([np.quantile(v, quantile) for v in values], dtype=float)
        margins[release] = np.pad(shrink * q, (0, 24 - lead_count))
    return margins


def _solve_contract_scenario_lp(
    price: np.ndarray,
    load: np.ndarray,
    reference_plan: np.ndarray,
    initial_soc: float,
    start: int,
    scenario_pv: np.ndarray,
    scenario_probabilities: np.ndarray,
    terminal_soc_target_kwh: float | None = None,
    terminal_shortfall_penalty_yuan_per_kwh: float | None = None,
    cvar_alpha: float | None = None,
    risk_aversion_lambda: float | None = None,
    settlement_mode: str = DEFAULT_SETTLEMENT_MODE,
    scenario_price: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Risk-aware adjustment LP with one common contract and scenario recourse.

    The adjusted contract is common to all scenarios.  Storage and emergency
    purchases are scenario recourse, constrained by non-anticipativity until
    the observed PV histories distinguish scenarios.
    """
    p = np.asarray(price[start:], dtype=float)
    l = np.asarray(load[start:], dtype=float)
    ref = np.asarray(reference_plan[start:], dtype=float)
    scenarios = np.asarray(scenario_pv[:, start:], dtype=float)
    probabilities = np.asarray(scenario_probabilities, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != len(p):
        raise ValueError("调整阶段场景光伏维度错误")
    m, n = scenarios.shape
    if probabilities.shape != (m,) or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("调整阶段场景概率错误")
    probabilities = probabilities / probabilities.sum()
    scenario_price_suffix = None
    if scenario_price is not None:
        price_scenarios = np.asarray(scenario_price, dtype=float)
        if price_scenarios.ndim != 2 or price_scenarios.shape != (m, len(price)):
            raise ValueError("调整阶段场景价格必须为(m,144)矩阵")
        if not np.isfinite(price_scenarios).all() or np.any(price_scenarios < 0.0):
            raise ValueError("调整阶段场景价格必须为有限非负数")
        scenario_price_suffix = price_scenarios[:, start:]
    alpha = solve_q2.CVAR_ALPHA if cvar_alpha is None else float(cvar_alpha)
    risk_lambda = (
        solve_q2.RISK_AVERSION_LAMBDA
        if risk_aversion_lambda is None
        else float(risk_aversion_lambda)
    )
    if not 0.0 < alpha < 1.0:
        raise ValueError("CVaR置信水平必须位于(0,1)")
    if risk_lambda < 0.0:
        raise ValueError("CVaR风险权重不得为负")
    settlement_mode = _validate_settlement_mode(settlement_mode)
    terminal_target = _terminal_target(terminal_soc_target_kwh, initial_soc)
    terminal_penalty = (
        _terminal_penalty(p)
        if terminal_shortfall_penalty_yuan_per_kwh is None
        else float(terminal_shortfall_penalty_yuan_per_kwh)
    )
    if terminal_penalty < 0.0:
        raise ValueError("日末SOC缺口惩罚不得为负")

    # Common A,u,v followed by per-scenario source allocation and recourse.
    block_size = 10 * n + 3
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
        "terminal_deficit": 10 * n + 2,
    }
    scenario_start = 3 * n
    eta_index = scenario_start + m * block_size
    count = eta_index + 1

    def variable(scenario: int, block: str, period: int = 0) -> int:
        return scenario_start + scenario * block_size + blocks[block] + period

    objective = np.zeros(count, dtype=float)
    if settlement_mode == "min_base":
        objective[:n] = p
    objective[n:2 * n] = 0.5 * p
    objective[2 * n:3 * n] = (
        0.5 * p if settlement_mode == "min_base" else 1.5 * p
    )
    objective[eta_index] = risk_lambda

    equal_rows = n + m * (5 * n + 1)
    nonanticipativity: list[tuple[str, int, int, int]] = []
    # At the update instant the current interval has not yet been executed;
    # future recourse becomes adaptive only after observations are revealed.
    for t in range(n):
        groups = solve_q2.build_information_groups(
            scenarios,
            scenarios.mean(axis=0),
            observed_end=t,
            scenario_price_flat=scenario_price_suffix,
            forecast_price_flat=p if scenario_price_suffix is not None else None,
        )
        for group in groups:
            anchor = group[0]
            for scenario in group[1:]:
                for block in ("charge", "discharge", "emergency"):
                    nonanticipativity.append((block, t, scenario, anchor))
    equal_rows += len(nonanticipativity)
    upper_rows = m * (n + 2)
    eq_row: list[int] = []
    eq_col: list[int] = []
    eq_value: list[float] = []
    eq_rhs = np.zeros(equal_rows, dtype=float)
    ub_row: list[int] = []
    ub_col: list[int] = []
    ub_value: list[float] = []
    ub_rhs = np.zeros(upper_rows, dtype=float)

    for scenario in range(m):
        probability = float(probabilities[scenario])
        scenario_price_row = (
            p if scenario_price_suffix is None else scenario_price_suffix[scenario]
        )
        objective[variable(scenario, "emergency"):variable(scenario, "emergency") + n] = (
            probability * EMERGENCY_MULTIPLIER * scenario_price_row
        )
        objective[variable(scenario, "charge"):variable(scenario, "charge") + n] = (
            probability * THROUGHPUT_PENALTY_YUAN_PER_KWH
        )
        objective[variable(scenario, "discharge"):variable(scenario, "discharge") + n] = (
            probability * THROUGHPUT_PENALTY_YUAN_PER_KWH
        )
        objective[variable(scenario, "xi")] = (
            risk_lambda * probability / (1.0 - alpha)
        )
        objective[variable(scenario, "terminal_deficit")] = (
            probability * terminal_penalty
        )
        offset = scenario * (5 * n + 1)
        for t in range(n):
            row = offset + t
            eq_row.extend((row, row, row, row))
            eq_col.extend((variable(scenario, "g_load", t),
                           variable(scenario, "g_charge", t),
                           variable(scenario, "g_unused", t), t))
            eq_value.extend((1.0, 1.0, 1.0, -1.0))

            row = offset + n + t
            eq_row.extend((row, row, row))
            eq_col.extend((variable(scenario, "r_load", t),
                           variable(scenario, "r_charge", t),
                           variable(scenario, "curtailment", t)))
            eq_value.extend((1.0, 1.0, 1.0))
            eq_rhs[row] = scenarios[scenario, t]

            row = offset + 2 * n + t
            eq_row.extend((row, row, row, row))
            eq_col.extend((variable(scenario, "g_load", t),
                           variable(scenario, "r_load", t),
                           variable(scenario, "discharge", t),
                           variable(scenario, "emergency", t)))
            eq_value.extend((1.0, 1.0, 1.0, 1.0))
            eq_rhs[row] = l[t]

            row = offset + 3 * n + t
            eq_row.extend((row, row, row))
            eq_col.extend((variable(scenario, "g_charge", t),
                           variable(scenario, "r_charge", t),
                           variable(scenario, "charge", t)))
            eq_value.extend((1.0, 1.0, -1.0))

            row = offset + 4 * n + t
            eq_row.extend((row, row, row, row))
            eq_col.extend((variable(scenario, "soc", t + 1),
                           variable(scenario, "soc", t),
                           variable(scenario, "charge", t),
                           variable(scenario, "discharge", t)))
            eq_value.extend((1.0, -1.0, -ETA, 1.0 / ETA))

        row = offset + 5 * n
        eq_row.append(row)
        eq_col.append(variable(scenario, "soc"))
        eq_value.append(1.0)
        eq_rhs[row] = initial_soc

        for t in range(n):
            row = scenario * (n + 1) + t
            ub_row.extend((row, row))
            ub_col.extend((variable(scenario, "charge", t),
                           variable(scenario, "discharge", t)))
            ub_value.extend((1.0, 1.0))
            ub_rhs[row] = BATTERY_POWER_KWH
        row = scenario * (n + 2) + n
        ub_row.extend([row] * (n + 2))
        ub_col.extend([variable(scenario, "emergency", t) for t in range(n)] +
                      [eta_index, variable(scenario, "xi")])
        ub_value.extend(
            list(EMERGENCY_MULTIPLIER * scenario_price_row) + [-1.0, -1.0]
        )
        row = scenario * (n + 2) + n + 1
        ub_row.extend((row, row))
        ub_col.extend((variable(scenario, "soc", n),
                       variable(scenario, "terminal_deficit")))
        ub_value.extend((-1.0, -1.0))
        ub_rhs[row] = -terminal_target

    # Contract adjustment relation A+u-v=P.
    for t in range(n):
        eq_row.extend((m * (5 * n + 1) + t,) * 3)
        eq_col.extend((t, n + t, 2 * n + t))
        eq_value.extend((1.0, 1.0, -1.0))
        eq_rhs[m * (5 * n + 1) + t] = ref[t]

    for row, (block, period, scenario, anchor) in enumerate(
        nonanticipativity, start=m * (5 * n + 1) + n
    ):
        eq_row.extend((row, row))
        eq_col.extend((variable(scenario, block, period), variable(anchor, block, period)))
        eq_value.extend((1.0, -1.0))

    equality_matrix = coo_matrix((eq_value, (eq_row, eq_col)),
                                  shape=(equal_rows, count)).tocsr()
    upper_matrix = coo_matrix((ub_value, (ub_row, ub_col)),
                              shape=(upper_rows, count)).tocsr()
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * (3 * n)
    for _ in range(m):
        bounds.extend([(0.0, None)] * (6 * n))
        bounds.extend([(0.0, BATTERY_POWER_KWH)] * (2 * n))
        bounds.extend([(BATTERY_MIN_KWH, BATTERY_MAX_KWH)] * (n + 1))
        bounds.extend([(0.0, None)] * n)
        bounds.extend([(0.0, None)] * 2)
    bounds.append((0.0, None))
    solution = linprog(objective, A_ub=upper_matrix, b_ub=ub_rhs,
                       A_eq=equality_matrix, b_eq=eq_rhs,
                       bounds=bounds, method=LP_METHOD)
    if not solution.success:
        raise RuntimeError(f"场景调整购电LP失败(start={start}): {solution.message}")
    values = solution.x
    emergency_costs = np.asarray([
        np.dot(
            EMERGENCY_MULTIPLIER * (
                p if scenario_price_suffix is None else scenario_price_suffix[s]
            ),
            values[variable(s, "emergency"):variable(s, "emergency") + n],
        )
        for s in range(m)
    ])
    cvar = float(
        values[eta_index] + np.dot(
            probabilities,
            [values[variable(s, "xi")] for s in range(m)]
        ) / (1.0 - alpha)
    )
    return {
        "purchase": values[:n].copy(),
        "charge": np.zeros(n),
        "discharge": np.zeros(n),
        "soc": np.zeros(n + 1),
        "emergency": np.zeros(n),
        "objective": float(solution.fun),
        "expected_emergency_cost": float(np.dot(probabilities, emergency_costs)),
        "cvar_emergency_cost": cvar,
        "expected_terminal_shortfall": float(
            np.dot(
                probabilities,
                [values[variable(s, "terminal_deficit")] for s in range(m)],
            )
        ),
        "scenario_count": int(m),
        "nonanticipativity_constraint_count": int(len(nonanticipativity)),
    }


def _solve_contract_lp(
    price: np.ndarray,
    load: np.ndarray,
    pv_forecast: np.ndarray,
    initial_soc: float,
    reference_plan: np.ndarray | None = None,
    start: int = 0,
    scenario_pv: np.ndarray | None = None,
    scenario_probabilities: np.ndarray | None = None,
    terminal_soc_target_kwh: float | None = None,
    terminal_shortfall_penalty_yuan_per_kwh: float | None = None,
    cvar_alpha: float | None = None,
    risk_aversion_lambda: float | None = None,
    settlement_mode: str = DEFAULT_SETTLEMENT_MODE,
    scenario_price: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Optimize normal purchase and storage over ``start:144``.

    With ``reference_plan=None`` this is the 00:00 plan.  Otherwise it is a
    forecast-update adjustment LP.  The settlement is represented by
    ``g+u-v=P`` and ``p*g+0.5p*u+1.5p*v``.
    """
    if scenario_price is not None and (reference_plan is None or scenario_pv is None):
        raise ValueError("场景价格必须与调整阶段的场景光伏和参考计划同时提供")
    if reference_plan is not None and scenario_pv is not None:
        if scenario_probabilities is None:
            scenario_probabilities = np.full(len(scenario_pv), 1.0 / len(scenario_pv))
        return _solve_contract_scenario_lp(
            price, load, reference_plan, initial_soc, start,
            np.asarray(scenario_pv, dtype=float),
            np.asarray(scenario_probabilities, dtype=float),
            terminal_soc_target_kwh=terminal_soc_target_kwh,
            terminal_shortfall_penalty_yuan_per_kwh=(
                terminal_shortfall_penalty_yuan_per_kwh
            ),
            cvar_alpha=cvar_alpha,
            risk_aversion_lambda=risk_aversion_lambda,
            settlement_mode=settlement_mode,
            scenario_price=scenario_price,
        )

    n = PERIODS - start
    p = price[start:]
    l = load[start:]
    pv = pv_forecast[start:]
    settlement_mode = _validate_settlement_mode(settlement_mode)
    terminal_target = _terminal_target(terminal_soc_target_kwh, initial_soc)
    terminal_penalty = (
        _terminal_penalty(p)
        if terminal_shortfall_penalty_yuan_per_kwh is None
        else float(terminal_shortfall_penalty_yuan_per_kwh)
    )
    if terminal_penalty < 0.0:
        raise ValueError("日末SOC缺口惩罚不得为负")
    # g, c, d, w, e, s(0..n), and (u,v) only for adjustment mode.
    g0, c0, d0, w0, e0, s0 = 0, n, 2*n, 3*n, 4*n, 5*n
    u0 = 6*n + 1
    v0 = 7*n + 1
    terminal_deficit = 8*n + 1 if reference_plan is not None else 6*n + 1
    # Base blocks are g,c,d,w,e (5n), followed by SOC (n+1), then
    # adjustment slacks u,v (2n) when an update is being optimized.
    count = 6*n + 2 + (2*n if reference_plan is not None else 0)
    obj = np.zeros(count)
    if reference_plan is None:
        obj[g0:g0+n] = p
    else:
        # p*min(P,A)+0.5p(P-A)^+ + 1.5p(A-P)^+ is equivalent to
        # p*A + 0.5p*u + 0.5p*v under A+u-v=P.
        if settlement_mode == "min_base":
            obj[g0:g0+n] = p
        obj[u0:u0+n] = 0.5 * p
        # Under the alternative contract interpretation, upward adjustment
        # A-P is charged at 1.5p.  The min-base interpretation is equivalent
        # to 0.5p on both slacks after eliminating the fixed base quantity.
        obj[v0:v0+n] = 0.5 * p if settlement_mode == "min_base" else 1.5 * p
    obj[e0:e0+n] = EMERGENCY_MULTIPLIER * p
    obj[c0:c0+n] = THROUGHPUT_PENALTY_YUAN_PER_KWH
    obj[d0:d0+n] = THROUGHPUT_PENALTY_YUAN_PER_KWH
    obj[terminal_deficit] = terminal_penalty

    eq_rows = 2*n + 1
    Aeq = np.zeros((eq_rows, count)); beq = np.zeros(eq_rows)
    for k in range(n):
        r = 2*k
        Aeq[r, g0+k] = 1; Aeq[r, c0+k] = -1; Aeq[r, d0+k] = 1
        Aeq[r, e0+k] = 1; Aeq[r, w0+k] = -1
        beq[r] = l[k] - pv[k]
        Aeq[r+1, s0+k+1] = 1; Aeq[r+1, s0+k] = -1
        Aeq[r+1, c0+k] = -ETA; Aeq[r+1, d0+k] = 1/ETA
    Aeq[-1, s0] = 1; beq[-1] = initial_soc

    # Bound aggregate battery throughput in each interval.  This removes the
    # simultaneous charge/discharge LP degeneracy without binary variables.
    Aub = np.zeros((n + 1, count), dtype=float)
    bub = np.concatenate((np.full(n, BATTERY_POWER_KWH), [-terminal_target]))
    for k in range(n):
        Aub[k, c0 + k] = 1.0
        Aub[k, d0 + k] = 1.0
    Aub[n, s0 + n] = -1.0
    Aub[n, terminal_deficit] = -1.0
    if reference_plan is not None:
        ref = np.asarray(reference_plan[start:], dtype=float)
        extra_eq = np.zeros((n, count)); extra_rhs = ref.copy()
        for k in range(n):
            extra_eq[k, g0+k] = 1.0
            extra_eq[k, u0+k] = 1.0
            extra_eq[k, v0+k] = -1.0
        Aeq = np.vstack([Aeq[:-1], extra_eq, Aeq[-1:]])
        beq = np.concatenate([beq[:-1], extra_rhs, beq[-1:]])
    bounds = [(0, None)]*n + [(0, BATTERY_POWER_KWH)]*(2*n) + [(0, None)]*(2*n)
    bounds += [(BATTERY_MIN_KWH, BATTERY_MAX_KWH)]*(n+1)
    if reference_plan is not None:
        bounds += [(0, None)]*(2*n)
    bounds += [(0, None)]
    sol = linprog(obj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=bounds, method=LP_METHOD)
    if not sol.success:
        raise RuntimeError(f"合同购电LP失败(start={start}): {sol.message}")
    x = sol.x
    return {
        "purchase": x[g0:g0+n].copy(),
        "charge": x[c0:c0+n].copy(),
        "discharge": x[d0:d0+n].copy(),
        "soc": x[s0:s0+n+1].copy(),
        "emergency": x[e0:e0+n].copy(),
        "objective": float(sol.fun),
        "terminal_shortfall": float(x[terminal_deficit]),
    }


def _rolling_execute(price, load, actual_pv, schedule, forecast_curve, initial_soc,
                     terminal_soc_target_kwh: float | None = None,
                     terminal_shortfall_penalty_yuan_per_kwh: float | None = None,
                     execute_periods: int | None = None,
                     scenario_pv: np.ndarray | None = None,
                     scenario_probabilities: np.ndarray | None = None,
                     observed_price: np.ndarray | None = None,
                     scenario_price: np.ndarray | None = None,
                     reference_charge_kwh: np.ndarray | None = None,
                     reference_discharge_kwh: np.ndarray | None = None,
                     trajectory_tracking_penalty_yuan_per_kwh: float = TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH):
    """Causal MPC over the remaining day; execute only the first block.

    At each ten-minute step the LP sees the complete remaining horizon and
    the current contract schedule, but only the first control is committed.
    ``execute_periods`` limits the returned/committed prefix when a forecast
    update is due before the end of the day.
    """
    horizon = len(schedule)
    if not (len(price) == len(load) == len(actual_pv) == len(forecast_curve) == horizon):
        raise ValueError("滚动执行输入长度不一致")
    tracking_enabled = (
        reference_charge_kwh is not None or reference_discharge_kwh is not None
    )
    if tracking_enabled:
        if reference_charge_kwh is None or reference_discharge_kwh is None:
            raise ValueError("日前储能参考轨迹必须同时提供充电和放电向量")
        reference_charge_kwh = np.asarray(reference_charge_kwh, dtype=float)
        reference_discharge_kwh = np.asarray(reference_discharge_kwh, dtype=float)
        if (
            reference_charge_kwh.shape != (horizon,)
            or reference_discharge_kwh.shape != (horizon,)
            or not np.isfinite(reference_charge_kwh).all()
            or not np.isfinite(reference_discharge_kwh).all()
            or np.any(reference_charge_kwh < 0.0)
            or np.any(reference_discharge_kwh < 0.0)
        ):
            raise ValueError("日前储能参考轨迹必须是与执行区间等长的有限非负向量")
        if trajectory_tracking_penalty_yuan_per_kwh < 0.0:
            raise ValueError("储能轨迹跟踪惩罚不得为负")
        reference_net_soc_change_kwh = (
            ETA * reference_charge_kwh - reference_discharge_kwh / ETA
        )
    if observed_price is None:
        observed_price = np.asarray(price, dtype=float)
    else:
        observed_price = np.asarray(observed_price, dtype=float)
    if observed_price.shape != (horizon,) or not np.isfinite(observed_price).all() or np.any(observed_price < 0.0):
        raise ValueError("滚动执行的已观测价格长度或取值错误")
    if scenario_pv is not None:
        scenario_pv = np.asarray(scenario_pv, dtype=float)
        if scenario_pv.ndim != 2 or scenario_pv.shape[1] != horizon:
            raise ValueError("滚动执行场景预测形状错误")
        if not np.isfinite(scenario_pv).all() or np.any(scenario_pv < 0.0):
            raise ValueError("滚动执行场景光伏必须为有限非负数")
    if scenario_price is not None:
        scenario_price = np.asarray(scenario_price, dtype=float)
        if scenario_price.ndim != 2 or scenario_price.shape[1] != horizon:
            raise ValueError("滚动执行场景价格形状错误")
        if not np.isfinite(scenario_price).all() or np.any(scenario_price < 0.0):
            raise ValueError("滚动执行场景价格必须为有限非负数")
    scenario_count = 0
    if scenario_pv is not None:
        scenario_count = scenario_pv.shape[0]
    if scenario_price is not None:
        if scenario_count and scenario_price.shape[0] != scenario_count:
            raise ValueError("滚动执行场景光伏与价格数量不一致")
        scenario_count = scenario_price.shape[0]
    if scenario_count == 0 and (scenario_pv is not None or scenario_price is not None):
        raise ValueError("滚动执行场景至少需要一条光伏或价格路径")
    if scenario_count:
        if scenario_probabilities is None:
            scenario_probabilities = np.full(
                scenario_count, 1.0 / scenario_count
            )
        scenario_probabilities = np.asarray(scenario_probabilities, dtype=float)
        if scenario_probabilities.shape != (scenario_count,) or np.any(scenario_probabilities < 0):
            raise ValueError("滚动执行场景概率形状或取值错误")
        if scenario_probabilities.sum() <= 0:
            raise ValueError("滚动执行场景概率和必须为正")
        scenario_probabilities = scenario_probabilities / scenario_probabilities.sum()
        log_weights = np.log(np.maximum(scenario_probabilities, 1e-300))
    if execute_periods is None:
        execute_periods = horizon
    if not 1 <= execute_periods <= horizon:
        raise ValueError("滚动执行区间长度非法")
    terminal_target = _terminal_target(terminal_soc_target_kwh, initial_soc)
    terminal_penalty = (
        _terminal_penalty(np.asarray(price, dtype=float))
        if terminal_shortfall_penalty_yuan_per_kwh is None
        else float(terminal_shortfall_penalty_yuan_per_kwh)
    )
    if terminal_penalty < 0.0:
        raise ValueError("日末SOC缺口惩罚不得为负")
    charge = np.zeros(execute_periods); discharge = np.zeros(execute_periods)
    emergency = np.zeros(execute_periods); curtail = np.zeros(execute_periods)
    unused_grid = np.zeros(execute_periods)
    soc = np.zeros(execute_periods + 1); soc[0] = initial_soc
    soc_current = initial_soc
    for t in range(execute_periods):
        rem = horizon - t
        p = price[t:]; l = load[t:]
        weights = None
        if scenario_count:
            if scenario_pv is not None:
                residuals = scenario_pv[:, t] - forecast_curve[t]
                observed_residual = actual_pv[t] - forecast_curve[t]
                scale = float(np.median(np.abs(residuals - np.median(residuals))) + 1e-3)
                log_weights += -0.5 * ((residuals - observed_residual) / scale) ** 2
            if scenario_price is not None:
                residuals = scenario_price[:, t] - p[0]
                observed_residual = observed_price[t] - p[0]
                scale = float(np.median(np.abs(residuals - np.median(residuals))) + 1e-4)
                log_weights += -0.5 * ((residuals - observed_residual) / scale) ** 2
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= weights.sum()
        if scenario_pv is None:
            pv_for_horizon = forecast_curve[t:].copy()
        else:
            pv_for_horizon = np.average(scenario_pv[:, t:], axis=0, weights=weights)
        pv_for_horizon[0] = actual_pv[t]
        # LP layout c,d,w_pv,w_grid,e,s. Future forecast is only a control aid.
        c0,d0,wpv0,wgrid0,e0,s0 = 0,rem,2*rem,3*rem,4*rem,5*rem
        terminal_deficit = 6 * rem + 1
        if tracking_enabled:
            tracking_dev_pos = terminal_deficit + 1
            tracking_dev_neg = tracking_dev_pos + rem
            count = tracking_dev_neg + rem
        else:
            count = 6*rem + 2
        # Keep the release-specific robust price as the MPC objective.  The
        # paired price paths still update scenario likelihoods above, but a
        # weighted mean would erase the calibrated upper-tail protection.
        control_price = p.copy()
        # The first remaining interval is observable at execution time; later
        # intervals retain the release-specific causal forecast.
        control_price[0] = observed_price[t]
        obj = np.zeros(count); obj[e0:e0+rem] = EMERGENCY_MULTIPLIER*control_price
        obj[c0:c0+rem] = THROUGHPUT_PENALTY_YUAN_PER_KWH
        obj[d0:d0+rem] = THROUGHPUT_PENALTY_YUAN_PER_KWH
        obj[terminal_deficit] = terminal_penalty
        if tracking_enabled:
            obj[tracking_dev_pos:tracking_dev_pos + rem] = (
                trajectory_tracking_penalty_yuan_per_kwh
            )
            obj[tracking_dev_neg:tracking_dev_neg + rem] = (
                trajectory_tracking_penalty_yuan_per_kwh
            )
        eq_rows = 2 * rem + 1 + (rem if tracking_enabled else 0)
        eq_row = []
        eq_col = []
        eq_value = []
        beq = np.zeros(eq_rows)
        for k in range(rem):
            r = 2 * k
            eq_row.extend((r, r, r, r, r))
            eq_col.extend((c0 + k, d0 + k, wpv0 + k, wgrid0 + k, e0 + k))
            eq_value.extend((-1.0, 1.0, -1.0, -1.0, 1.0))
            beq[r] = l[k] - schedule[t + k] - pv_for_horizon[k]
            eq_row.extend((r + 1, r + 1, r + 1, r + 1))
            eq_col.extend((s0 + k + 1, s0 + k, c0 + k, d0 + k))
            eq_value.extend((1.0, -1.0, -ETA, 1.0 / ETA))
        terminal_row = 2 * rem
        eq_row.append(terminal_row); eq_col.append(s0); eq_value.append(1.0)
        beq[terminal_row] = soc_current
        if tracking_enabled:
            row = terminal_row + 1
            for k in range(rem):
                eq_row.extend((row, row, row, row))
                eq_col.extend((c0 + k, d0 + k, tracking_dev_pos + k, tracking_dev_neg + k))
                eq_value.extend((ETA, -1.0 / ETA, -1.0, 1.0))
                beq[row] = reference_net_soc_change_kwh[t + k]
                row += 1
        Aeq = coo_matrix(
            (eq_value, (eq_row, eq_col)), shape=(eq_rows, count)
        ).tocsr()
        baseline = schedule[t:] + pv_for_horizon
        # Emergency electricity is reserved for the load.  Charging is
        # allowed only when scheduled electricity plus forecast PV already
        # covers the load, matching Question 2's source-allocation rule.
        bounds = [
            (0.0, BATTERY_POWER_KWH)
            if baseline[k] >= l[k] - 1e-9
            else (0.0, 0.0)
            for k in range(rem)
        ]
        bounds += [(0.0, BATTERY_POWER_KWH)] * rem
        bounds += [(0.0, float(max(pv_for_horizon[k], 0.0))) for k in range(rem)]
        bounds += [(0.0, float(max(schedule[t + k], 0.0))) for k in range(rem)]
        bounds += [
            (0.0, float(max(l[k] - baseline[k], 0.0))) if baseline[k] < l[k] - 1e-8 else (0.0, 0.0)
            for k in range(rem)
        ]
        bounds += [(BATTERY_MIN_KWH,BATTERY_MAX_KWH)]*(rem+1)
        bounds += [(0.0, None)]
        if tracking_enabled:
            bounds += [(0.0, None)] * (2 * rem)
        ub_row = []
        ub_col = []
        ub_value = []
        for k in range(rem):
            ub_row.extend((k, k)); ub_col.extend((c0 + k, d0 + k)); ub_value.extend((1.0, 1.0))
        ub_row.extend((rem, rem)); ub_col.extend((s0 + rem, terminal_deficit)); ub_value.extend((-1.0, -1.0))
        power_ub = coo_matrix(
            (ub_value, (ub_row, ub_col)), shape=(rem + 1, count)
        ).tocsr()
        upper_rhs = np.concatenate((np.full(rem, BATTERY_POWER_KWH), [-terminal_target]))
        sol = linprog(
            obj, A_ub=power_ub, b_ub=upper_rhs,
            A_eq=Aeq, b_eq=beq, bounds=bounds, method=LP_METHOD
        )
        if not sol.success:
            raise RuntimeError(f"滚动执行LP失败(t={t}): {sol.message}")
        # The continuous LP may use a simultaneous charge/discharge basis
        # when the day-ahead tracking reference asks for both directions.
        # Only the current control is committed, so resolve this step with
        # each mutually exclusive mode and keep the cheaper feasible result.
        if sol.x[c0] > 1e-7 and sol.x[d0] > 1e-7:
            candidates = []
            for blocked in (c0, d0):
                mode_bounds = list(bounds)
                mode_bounds[blocked] = (0.0, 0.0)
                mode_sol = linprog(
                    obj, A_ub=power_ub, b_ub=upper_rhs,
                    A_eq=Aeq, b_eq=beq, bounds=mode_bounds, method=LP_METHOD
                )
                if mode_sol.success:
                    candidates.append(mode_sol)
            if candidates:
                sol = min(candidates, key=lambda item: float(item.fun))
        x=sol.x; charge[t]=x[c0]; discharge[t]=x[d0]
        curtail[t]=x[wpv0]; unused_grid[t]=x[wgrid0]
        emergency[t]=x[e0]; soc[t+1]=x[s0+1]
        soc_current = soc[t+1]
    balance = (schedule[:execute_periods] + actual_pv[:execute_periods]
               + discharge + emergency - load[:execute_periods] - charge
               - curtail - unused_grid)
    return {"charge":charge,"discharge":discharge,"emergency":emergency,"curtailment":curtail,
            "unused_grid":unused_grid,"soc":soc,
            "terminal_shortfall":float(max(terminal_target - soc[-1], 0.0)),
            "max_balance_error":float(np.max(np.abs(balance))),
            "trajectory_tracking_enabled":bool(tracking_enabled)}


def solve_day(price, load, actual_pv, releases, initial_soc, margins=None,
              enabled_releases: tuple[int, ...] = (0, 6, 12, 18),
              initial_plan: dict | None = None,
              scenario_curves: dict[int, np.ndarray] | None = None,
              scenario_probabilities: dict[int, np.ndarray] | None = None,
              expansion_method: str = DEFAULT_FORECAST_EXPANSION,
              scenario_reduction_mode: str = DEFAULT_SCENARIO_REDUCTION_MODE,
              terminal_soc_target_kwh: float | None = None,
              terminal_shortfall_penalty_yuan_per_kwh: float | None = None,
              cvar_alpha: float | None = None,
              risk_aversion_lambda: float | None = None,
              settlement_mode: str = DEFAULT_SETTLEMENT_MODE,
              price_by_release: dict[int, np.ndarray] | None = None,
              scenario_price_curves: dict[int, np.ndarray] | None = None,
              settlement_price: np.ndarray | None = None,
              observed_price: np.ndarray | None = None):
    price = np.asarray(price, dtype=float)
    if price.shape != (PERIODS,) or not np.isfinite(price).all() or np.any(price < 0.0):
        raise ValueError("问题三决策价格必须为长度144的有限非负向量")
    if settlement_price is None:
        settlement_price = price.copy()
    else:
        settlement_price = np.asarray(settlement_price, dtype=float)
    if settlement_price.shape != (PERIODS,) or not np.isfinite(settlement_price).all() or np.any(settlement_price < 0.0):
        raise ValueError("问题三结算价格必须为长度144的有限非负向量")
    if observed_price is None:
        observed_price = settlement_price.copy()
    else:
        observed_price = np.asarray(observed_price, dtype=float)
    if observed_price.shape != (PERIODS,) or not np.isfinite(observed_price).all() or np.any(observed_price < 0.0):
        raise ValueError("问题三已观测价格必须为长度144的有限非负向量")
    release_prices = {0: price.copy()}
    if price_by_release is not None:
        for release, curve in price_by_release.items():
            if release not in (0, 6, 12, 18):
                raise ValueError("price_by_release只能包含0、6、12、18时")
            curve = np.asarray(curve, dtype=float)
            if curve.shape != (PERIODS,) or not np.isfinite(curve).all() or np.any(curve < 0.0):
                raise ValueError("每个发布时间的决策价格必须为长度144的有限非负向量")
            release_prices[int(release)] = curve.copy()
    raw_curves = build_release_curves(releases, expansion_method=expansion_method)
    adjusted_curves = build_release_curves(
        releases, margins, expansion_method=expansion_method
    )
    settlement_mode = _validate_settlement_mode(settlement_mode)
    # 00:00 initial plan, based only on the 00:00 forecast.
    if initial_plan is None:
        plan_lp = _solve_contract_lp(
            release_prices[0], load, raw_curves[0], initial_soc,
            terminal_soc_target_kwh=terminal_soc_target_kwh,
            terminal_shortfall_penalty_yuan_per_kwh=(
                terminal_shortfall_penalty_yuan_per_kwh
            ),
            settlement_mode=settlement_mode,
        )
        plan = plan_lp["purchase"]
    else:
        plan = np.asarray(initial_plan["purchase_kwh"], dtype=float).copy()
        if plan.shape != (PERIODS,) or np.any(plan < -1e-8):
            raise ValueError("问题三的初始风险计划必须是长度为144的非负向量")
    if terminal_soc_target_kwh is None:
        if initial_plan is not None and "expected_first_day_terminal_soc_kwh" in initial_plan:
            terminal_soc_target_kwh = float(
                initial_plan["expected_first_day_terminal_soc_kwh"]
            )
        elif initial_plan is not None and "expected_terminal_soc_kwh" in initial_plan:
            terminal_soc_target_kwh = float(initial_plan["expected_terminal_soc_kwh"])
        else:
            terminal_soc_target_kwh = float(initial_soc)
    terminal_soc_target_kwh = _terminal_target(
        terminal_soc_target_kwh, initial_soc
    )
    if terminal_shortfall_penalty_yuan_per_kwh is None:
        terminal_shortfall_penalty_yuan_per_kwh = _terminal_penalty(
            np.asarray(release_prices[0], dtype=float)
        )
    schedule = plan.copy(); current_soc = initial_soc; executed = {"charge":np.zeros(PERIODS),"discharge":np.zeros(PERIODS),
        "emergency":np.zeros(PERIODS),"curtailment":np.zeros(PERIODS),
        "unused_grid":np.zeros(PERIODS),"soc":np.zeros(PERIODS+1)}
    executed["soc"][0] = initial_soc
    update_log = []
    enabled = tuple(sorted(set(enabled_releases)))
    if not enabled or enabled[0] != 0 or any(x not in (0, 6, 12, 18) for x in enabled):
        raise ValueError("enabled_releases必须从(0,6,12,18)中选择且包含0")
    allowed_releases = {0, 6, 12, 18}
    scenario_arrays: dict[int, np.ndarray] = {}
    price_scenario_arrays: dict[int, np.ndarray] = {}
    probability_arrays: dict[int, np.ndarray] = {}
    if scenario_curves is not None:
        if not isinstance(scenario_curves, dict):
            raise ValueError("scenario_curves必须是按发布时间索引的字典")
        if set(scenario_curves) - allowed_releases:
            raise ValueError("scenario_curves只能包含0、6、12、18时")
        for release, values in scenario_curves.items():
            array = np.asarray(values, dtype=float)
            if (
                array.ndim != 2
                or array.shape[1] != PERIODS
                or array.shape[0] == 0
                or not np.isfinite(array).all()
                or np.any(array < 0.0)
            ):
                raise ValueError("每个发布时间的光伏场景必须为(m,144)有限非负矩阵")
            scenario_arrays[int(release)] = array
    if scenario_price_curves is not None:
        if not isinstance(scenario_price_curves, dict):
            raise ValueError("scenario_price_curves必须是按发布时间索引的字典")
        if set(scenario_price_curves) - allowed_releases:
            raise ValueError("scenario_price_curves只能包含0、6、12、18时")
        if scenario_curves is None:
            raise ValueError("场景价格必须与场景光伏同时提供")
        for release, values in scenario_price_curves.items():
            array = np.asarray(values, dtype=float)
            if (
                array.ndim != 2
                or array.shape[1] != PERIODS
                or array.shape[0] == 0
                or not np.isfinite(array).all()
                or np.any(array < 0.0)
            ):
                raise ValueError("每个发布时间的价格场景必须为(m,144)有限非负矩阵")
            price_scenario_arrays[int(release)] = array
        if set(price_scenario_arrays) - set(scenario_arrays):
            raise ValueError("场景价格必须与对应发布时间的光伏场景配对")
    if scenario_probabilities is not None:
        if not isinstance(scenario_probabilities, dict):
            raise ValueError("scenario_probabilities必须是按发布时间索引的字典")
        if set(scenario_probabilities) - allowed_releases:
            raise ValueError("scenario_probabilities只能包含0、6、12、18时")
        if scenario_curves is None:
            raise ValueError("场景概率必须与场景光伏同时提供")
        for release, values in scenario_probabilities.items():
            array = np.asarray(values, dtype=float)
            if (
                array.ndim != 1
                or array.size == 0
                or not np.isfinite(array).all()
                or np.any(array < 0.0)
                or float(array.sum()) <= 0.0
            ):
                raise ValueError("每个发布时间的场景概率必须为有限非负向量且总和为正")
            probability_arrays[int(release)] = array
        if set(probability_arrays) - set(scenario_arrays):
            raise ValueError("场景概率必须与对应发布时间的光伏场景配对")
    for release in set(scenario_arrays) & set(price_scenario_arrays):
        if scenario_arrays[release].shape[0] != price_scenario_arrays[release].shape[0]:
            raise ValueError("对应发布时间的光伏与价格场景数量不一致")
    for release in set(scenario_arrays) & set(probability_arrays):
        if scenario_arrays[release].shape[0] != probability_arrays[release].shape[0]:
            raise ValueError("对应发布时间的光伏场景与概率数量不一致")
    for pos, release in enumerate(enabled):
        # The source row labelled h:00 is the first row covered by an h:00
        # update; its zero-based index is 6*h-1 for later releases.
        start = release_first_index(release)
        if release == 0:
            schedule[:] = plan
        else:
            adj = _solve_contract_lp(
                release_prices.get(release, release_prices[0]),
                load, adjusted_curves[release], current_soc, plan, start,
                scenario_pv=(scenario_curves[release]
                             if scenario_curves is not None and release in scenario_curves
                             else None),
                scenario_probabilities=(scenario_probabilities[release]
                                        if scenario_probabilities is not None and release in scenario_probabilities
                             else None),
                terminal_soc_target_kwh=terminal_soc_target_kwh,
                terminal_shortfall_penalty_yuan_per_kwh=(
                    terminal_shortfall_penalty_yuan_per_kwh
                ),
                cvar_alpha=cvar_alpha,
                risk_aversion_lambda=risk_aversion_lambda,
                settlement_mode=settlement_mode,
                scenario_price=(
                    scenario_price_curves[release]
                    if scenario_price_curves is not None and release in scenario_price_curves
                    else None
                ),
            )
            schedule[start:] = adj["purchase"]
        # Execute until next update, while optimizing over the whole
        # remaining day.  The 00:00 controller uses the raw published curve;
        # later releases use their causal bias-corrected curves.
        # ``end`` is an exclusive bound.  The next release takes effect at
        # its first interval, so the current block includes every interval
        # up to (but not including) that release start.
        end = (
            PERIODS
            if pos == len(enabled) - 1
            else release_first_index(enabled[pos + 1])
        )
        execution_curve = raw_curves[release] if release == 0 else adjusted_curves[release]
        execution_price = release_prices.get(release, release_prices[0])
        part = _rolling_execute(
            execution_price[start:], load[start:], actual_pv[start:], schedule[start:],
            execution_curve[start:], current_soc,
            terminal_soc_target_kwh=terminal_soc_target_kwh,
            terminal_shortfall_penalty_yuan_per_kwh=(
                terminal_shortfall_penalty_yuan_per_kwh
            ),
            execute_periods=end - start,
            scenario_pv=(
                scenario_curves[release][:, start:]
                if scenario_curves is not None and release in scenario_curves
                else None
            ),
            scenario_probabilities=(
                scenario_probabilities[release]
                if scenario_probabilities is not None and release in scenario_probabilities
                else None
            ),
            observed_price=observed_price[start:],
            scenario_price=(
                scenario_price_curves[release][:, start:]
                if scenario_price_curves is not None and release in scenario_price_curves
                else None
            ),
            reference_charge_kwh=(
                np.asarray(initial_plan["expected_first_day_charge_kwh"], dtype=float)[start:]
                if initial_plan is not None
                and "expected_first_day_charge_kwh" in initial_plan
                else None
            ),
            reference_discharge_kwh=(
                np.asarray(initial_plan["expected_first_day_discharge_kwh"], dtype=float)[start:]
                if initial_plan is not None
                and "expected_first_day_discharge_kwh" in initial_plan
                else None
            ),
        ) if end > start else None
        if part is not None:
            executed["charge"][start:end] = part["charge"]; executed["discharge"][start:end] = part["discharge"]
            executed["emergency"][start:end] = part["emergency"]; executed["curtailment"][start:end] = part["curtailment"]
            executed["unused_grid"][start:end] = part["unused_grid"]
            executed["soc"][start:end+1] = part["soc"]; current_soc = float(part["soc"][-1])
        update_log.append({
            "release_time": f"{release:02d}:00",
            "updated_from_index": int(start),
            # Persist the already-issued contract prefix so an independent
            # audit can verify that later releases only replace the suffix.
            "contract_prefix_purchase_kwh": schedule[:start].tolist(),
            "execution_scenario_count": int(
                len(scenario_probabilities[release])
                if scenario_probabilities is not None and release in scenario_probabilities
                else 0
            ),
        })
    # All reported settlement costs are ex-post values under Attachment 4's
    # realised tariff.  The forecast curves are used only inside the LP/MPC.
    planned_cost = float(np.dot(settlement_price, plan))
    diff = schedule - plan
    down = np.maximum(-diff, 0.0)
    up = np.maximum(diff, 0.0)
    base_settlement = float(np.dot(settlement_price, np.minimum(plan, schedule)))
    adjustment_cost = float(np.dot(0.5*settlement_price, down) + np.dot(1.5*settlement_price, up))
    main_settlement = base_settlement + adjustment_cost
    alternative_settlement = float(
        planned_cost + np.dot(0.5 * settlement_price, down) + np.dot(1.5 * settlement_price, up)
    )
    emergency_cost = float(np.dot(EMERGENCY_MULTIPLIER*settlement_price, executed["emergency"]))
    selected_settlement = (
        main_settlement
        if settlement_mode == "min_base"
        else alternative_settlement
    )
    return {"plan":plan,"adjusted":schedule,"charge":executed["charge"],"discharge":executed["discharge"],
            "soc":executed["soc"],"emergency":executed["emergency"],"curtailment":executed["curtailment"],
            "unused_grid":executed["unused_grid"],
            "planned_cost":planned_cost,"base_settlement_cost":base_settlement,
            "adjustment_cost":adjustment_cost,"normal_settlement_cost":selected_settlement,
            "main_settlement_cost":main_settlement,
            "alternative_settlement_cost":alternative_settlement,
            "penalty_only_settlement_cost":alternative_settlement,
            "emergency_cost":emergency_cost,
            "total_cost":selected_settlement+emergency_cost,
            "main_total_cost":main_settlement+emergency_cost,
            "alternative_total_cost":alternative_settlement+emergency_cost,
            "settlement_mode":settlement_mode,
            "terminal_soc_target_kwh":terminal_soc_target_kwh,
            "terminal_soc_shortfall_kwh":float(
                max(terminal_soc_target_kwh - executed["soc"][-1], 0.0)
            ),
            "trajectory_tracking_enabled": bool(
                initial_plan is not None
                and "expected_first_day_charge_kwh" in initial_plan
                and "expected_first_day_discharge_kwh" in initial_plan
            ),
            "trajectory_tracking_penalty_yuan_per_kwh": (
                TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH
            ),
            "max_balance_error":float(np.max(np.abs(
                schedule+actual_pv+executed["discharge"]+executed["emergency"]-load-executed["charge"]
                -executed["curtailment"]-executed["unused_grid"]))),
            "update_log":update_log}


def contiguous_intervals(values: np.ndarray) -> list[dict[str, float | str]]:
    labels = interval_labels(); out=[]; start=None; amount=0.0
    for i,v in enumerate(values):
        if v >= REPORTING_TOLERANCE_KWH:
            if start is None: start=i; amount=0.0
            amount += float(v)
        elif start is not None:
            out.append({"time_period":f"{labels[start].split('-')[0]}-{labels[i-1].split('-')[1]}","purchase_kwh":amount}); start=None
    if start is not None: out.append({"time_period":f"{labels[start].split('-')[0]}-{labels[-1].split('-')[1]}","purchase_kwh":amount})
    return out


def solve_all(
    root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    enabled_releases: tuple[int, ...] = (0, 6, 12, 18),
    seasonal_margins: bool = False,
    scenario_reduction_mode: str = DEFAULT_SCENARIO_REDUCTION_MODE,
    forecast_expansion: str = DEFAULT_FORECAST_EXPANSION,
    cvar_alpha: float | None = None,
    risk_aversion_lambda: float | None = None,
    stress_probability: float | None = None,
    settlement_mode: str = DEFAULT_SETTLEMENT_MODE,
) -> dict:
    price, dates, load, actual_pv, forecasts = load_inputs(root)
    start_date = start_date or OUTPUT_START; end_date = end_date or OUTPUT_END
    date_to_idx={d:i for i,d in enumerate(dates)}; days=[]; soc=INITIAL_SOC_KWH
    alpha = solve_q2.CVAR_ALPHA if cvar_alpha is None else float(cvar_alpha)
    risk_lambda = (
        solve_q2.RISK_AVERSION_LAMBDA
        if risk_aversion_lambda is None
        else float(risk_aversion_lambda)
    )
    effective_stress_probability = (
        solve_q2.CVAR_STRESS_PROBABILITY
        if stress_probability is None
        else float(stress_probability)
    )
    settlement_mode = _validate_settlement_mode(settlement_mode)
    margin_history = {
        d: causal_error_margins(
            d, dates, actual_pv, forecasts, quantile=0.30, shrink=1.0,
            seasonal=seasonal_margins,
        )
        for d in dates
    }
    for d in dates:
        index = date_to_idx[d]
        margins = margin_history[d]
        # The Q2-derived 00:00 risk plan uses the raw attachment-3 forecast
        # together with causal historical residual and low-PV stress scenarios.
        # Quantile bias correction is reserved for later forecast updates, so
        # the same forecast error is not counted twice in the base plan.
        forecast_00_raw = build_release_curves(
            forecasts[d], expansion_method=forecast_expansion
        )[0]
        forecast_00_adjusted = build_release_curves(
            forecasts[d], margins, expansion_method=forecast_expansion
        )[0]
        base_error_history = build_release0_error_history(
            index, dates, actual_pv, forecasts,
            expansion_method=forecast_expansion,
        )
        base_plan = solve_q2_base_plan(
            price, load[index], forecast_00_raw, soc, base_error_history,
            scenario_reduction_mode=scenario_reduction_mode,
            cvar_alpha=alpha,
            risk_aversion_lambda=risk_lambda,
            stress_probability=effective_stress_probability,
        )
        enabled = tuple(sorted(set(enabled_releases)))
        scenario_curves = {}
        scenario_probabilities = {}
        for release in enabled:
            current_curve = (
                build_release_curves(
                    forecasts[d], expansion_method=forecast_expansion
                )[release]
                if release == 0
                else build_release_curves(
                    forecasts[d], margins,
                    expansion_method=forecast_expansion,
                )[release]
            )
            curves, probabilities = build_release_scenario_curves(
                index,
                release,
                dates,
                actual_pv,
                forecasts,
                current_curve,
                price,
                margin_history,
                expansion_method=forecast_expansion,
                scenario_reduction_mode=scenario_reduction_mode,
                stress_probability=effective_stress_probability,
            )
            scenario_curves[release] = curves
            scenario_probabilities[release] = probabilities
        result=solve_day(
            price,load[index],actual_pv[index],forecasts[d],soc,margins,
            enabled_releases, initial_plan=base_plan,
            scenario_curves=scenario_curves,
            scenario_probabilities=scenario_probabilities,
            expansion_method=forecast_expansion,
            scenario_reduction_mode=scenario_reduction_mode,
            terminal_soc_target_kwh=float(
                base_plan["expected_first_day_terminal_soc_kwh"]
            ),
            terminal_shortfall_penalty_yuan_per_kwh=_terminal_penalty(price),
            cvar_alpha=alpha,
            risk_aversion_lambda=risk_lambda,
            settlement_mode=settlement_mode,
        )
        soc=float(result["soc"][-1])
        if start_date <= d <= end_date:
            p=result["plan"]; a=result["adjusted"]
            days.append({"date":d.isoformat(),"planned_purchase_kwh":p.tolist(),"adjusted_purchase_kwh":a.tolist(),
                "charge_kwh":result["charge"].tolist(),"discharge_kwh":result["discharge"].tolist(),"soc_kwh":result["soc"].tolist(),
                "emergency_kwh":result["emergency"].tolist(),"emergency_intervals":contiguous_intervals(result["emergency"]),
                "curtailment_kwh":result["curtailment"].tolist(),
                "unused_grid_kwh":result["unused_grid"].tolist(),
                "planned_cost_yuan":result["planned_cost"],"base_settlement_cost_yuan":result["base_settlement_cost"],
                "adjustment_cost_yuan":result["adjustment_cost"],"normal_settlement_cost_yuan":result["normal_settlement_cost"],
                "penalty_only_settlement_cost_yuan":result["penalty_only_settlement_cost"],
                "main_settlement_cost_yuan":result["main_settlement_cost"],
                "alternative_settlement_cost_yuan":result["alternative_settlement_cost"],
                "emergency_cost_yuan":result["emergency_cost"],"total_cost_yuan":result["total_cost"],
                "main_total_cost_yuan":result["main_total_cost"],
                "alternative_total_cost_yuan":result["alternative_total_cost"],
                "forecast_00_kwh":forecast_00_adjusted.tolist(),
                "base_plan_forecast_00_kwh":forecast_00_raw.tolist(),
                "forecast_06_kwh":build_release_curves(
                    forecasts[d], margins, expansion_method=forecast_expansion
                )[6].tolist(),
                "forecast_12_kwh":build_release_curves(
                    forecasts[d], margins, expansion_method=forecast_expansion
                )[12].tolist(),
                "forecast_18_kwh":build_release_curves(
                    forecasts[d], margins, expansion_method=forecast_expansion
                )[18].tolist(),
                "day_ahead_reference_charge_kwh": base_plan[
                    "expected_first_day_charge_kwh"
                ].tolist(),
                "day_ahead_reference_discharge_kwh": base_plan[
                    "expected_first_day_discharge_kwh"
                ].tolist(),
                "forecast_margin_00_kwh":margins[0].tolist(),"forecast_margin_06_kwh":margins[6].tolist(),
                "forecast_margin_12_kwh":margins[12].tolist(),"forecast_margin_18_kwh":margins[18].tolist(),
                "base_plan_scenario_count":int(base_plan["scenario_count"]),
                "base_plan_expected_emergency_cost_yuan":float(base_plan["expected_emergency_cost_yuan"]),
                "base_plan_cvar_emergency_cost_yuan":float(base_plan["cvar_emergency_cost_yuan"]),
                "base_plan_expected_terminal_soc_kwh":float(base_plan["expected_terminal_soc_kwh"]),
                "base_plan_expected_terminal_shortfall_kwh":float(base_plan["expected_terminal_shortfall_kwh"]),
                "base_plan_nonanticipativity_constraint_count":int(base_plan["nonanticipativity_constraint_count"]),
                "base_plan_planning_horizon_days":int(base_plan["planning_horizon_days"]),
                "base_plan_objective_value_yuan":float(base_plan["objective_value_yuan"]),
                "base_plan_raw_error_count":int(base_error_history.shape[0]),
                "max_balance_error_kwh":result["max_balance_error"],
                "initial_soc_kwh":float(result["soc"][0]),
                "terminal_soc_kwh":float(result["soc"][-1]),
                "terminal_soc_target_kwh":float(result["terminal_soc_target_kwh"]),
                "terminal_soc_shortfall_kwh":float(result["terminal_soc_shortfall_kwh"]),
                "trajectory_tracking_enabled": bool(result["trajectory_tracking_enabled"]),
                "trajectory_tracking_penalty_yuan_per_kwh": float(
                    result["trajectory_tracking_penalty_yuan_per_kwh"]
                ),
                "update_log":result["update_log"]})
    summary={"day_count":len(days),"planned_contract_cost_yuan":sum(x["planned_cost_yuan"] for x in days),
        "base_settlement_cost_yuan":sum(x["base_settlement_cost_yuan"] for x in days),
        "adjustment_cost_yuan":sum(x["adjustment_cost_yuan"] for x in days),
         "normal_settlement_cost_yuan":sum(x["normal_settlement_cost_yuan"] for x in days),
         "penalty_only_settlement_cost_yuan":sum(x["penalty_only_settlement_cost_yuan"] for x in days),
         "main_settlement_cost_yuan":sum(x["main_settlement_cost_yuan"] for x in days),
         "alternative_settlement_cost_yuan":sum(x["alternative_settlement_cost_yuan"] for x in days),
         "emergency_cost_yuan":sum(x["emergency_cost_yuan"] for x in days),
         "total_cost_yuan":sum(x["total_cost_yuan"] for x in days),
         "main_total_cost_yuan":sum(x["main_total_cost_yuan"] for x in days),
         "alternative_total_cost_yuan":sum(x["alternative_total_cost_yuan"] for x in days),
         "emergency_energy_kwh":sum(sum(x["emergency_kwh"]) for x in days),
        "emergency_day_count":sum(any(v>=REPORTING_TOLERANCE_KWH for v in x["emergency_kwh"]) for x in days),
        "max_balance_error_kwh":max(x["max_balance_error_kwh"] for x in days) if days else 0.0,
        "curtailment_energy_kwh":sum(sum(x["curtailment_kwh"]) for x in days),
        "unused_grid_energy_kwh":sum(sum(x["unused_grid_kwh"]) for x in days),
        "simultaneous_charge_discharge_intervals":sum(
            int(np.sum((np.asarray(x["charge_kwh"]) > 1e-7) &
                      (np.asarray(x["discharge_kwh"]) > 1e-7)))
            for x in days
        )}
    return {"model":{"name":"问题二风险感知日前计划 + 四阶段光伏预测更新 + 计划/调整购电合同 + 因果十分钟储能执行",
        "forecast_releases":"0:00, 6:00, 12:00, 18:00","enabled_releases":list(enabled_releases),"forecast_expansion":forecast_expansion,
        "base_plan":"沿用问题二风险感知日前LP的单日特例；0:00使用附件3原始预报、历史误差场景和低光伏压力场景，问题三再叠加四阶段预报更新和合同调整",
        "base_plan_solver":"调用solve_q2.solve_multiday_risk_aware_plan，设置horizon_days=1；保留CVaR、压力场景、非预见性约束和末端库存缺口惩罚",
        "base_plan_error_scenarios":f"目标日前附件3的0:00发布原始预测误差按{scenario_reduction_mode}评分分组并保留组概率，再加入与问题二相同的低光伏压力场景；不与更新阶段的分位数修正重复叠加",
        "forecast_error_margin":"仅用于6:00、12:00、18:00调整合同的目标日前历史附件3误差0.30分位数修正，不收缩并加到各发布时间预测",
        "margin_quantile":0.30,"margin_shrink":1.0,"margin_seasonal":seasonal_margins,"minimum_margin_history_days":7,
        "adjustment_risk_model":"调整合同为所有场景共享的A；储能和紧急购电为场景recourse，包含期望紧急费用与CVaR，并加入非预见性约束",
        "cvar_alpha":alpha,"risk_aversion_lambda":risk_lambda,"stress_probability":effective_stress_probability,
        "scenario_reduction_mode":scenario_reduction_mode,
        "settlement": "主口径为p·min(P,A)+0.5p·(P-A)^++1.5p·(A-P)^+；备用口径为pP+0.5p·(P-A)^++1.5p·(A-P)^+；紧急购电按5p",
        "settlement_mode": settlement_mode,
        "battery_efficiency":ETA,"battery_min_kwh":BATTERY_MIN_KWH,"battery_max_kwh":BATTERY_MAX_KWH,
        "battery_power_kw":5000.0,"time_step_hours":DT_HOURS,"initial_soc_kwh":INITIAL_SOC_KWH,
        "terminal_soc_policy":"以00:00风险日前计划的预期日末SOC为目标，滚动执行和调整LP统一使用软缺口约束",
        "terminal_soc_shortfall_penalty_factor":TERMINAL_SOC_SHORTFALL_PENALTY_FACTOR,
        "throughput_penalty_yuan_per_kwh":THROUGHPUT_PENALTY_YUAN_PER_KWH,
        "trajectory_tracking_penalty_yuan_per_kwh":TRAJECTORY_TRACKING_PENALTY_YUAN_PER_KWH,
        "trajectory_tracking":"执行层以L1软惩罚跟踪00:00风险日前计划的概率加权第一日充放电轨迹；实际光伏偏差时允许偏离，物理约束和紧急购电优先",
        "emergency_reporting_tolerance_kwh":REPORTING_TOLERANCE_KWH,
        "causality":"0:00初始计划只使用当时可得的历史误差场景和0:00预报；每个更新时刻只修改未来合同；每十分钟开始时将当前可测光伏视为已知，未来区间仍只使用预测与场景",
        "execution_mpc_horizon":"每个十分钟时刻优化从当前时刻到24:00的剩余日，并只提交当前时刻控制量",
        "execution_scenario_update":"按当前实际光伏与历史同发布时间误差场景及低光伏压力场景的似然更新权重，使用加权剩余预测",
        "execution_output":"逐十分钟充电、放电、PV弃光、未利用计划购电、紧急购电和SOC；四小时汇总仅用于官方模板展示"},
        "interval_labels":interval_labels(),"days":days,"summary":summary}


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/"结果"/"q3_solution.json"
    result=solve_all(root)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result["summary"],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
