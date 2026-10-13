"""Question 1: deterministic daily dispatch for the microgrid.

The input contains one 10-minute record for each of 144 ordered periods.
All power observations are converted to kWh before optimization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from data_processing import (
    DT_HOURS,
    PERIODS_PER_DAY,
    build_interval_labels,
    load_processed_attachment1,
)


ETA = 0.90
BATTERY_MIN_KWH = 1200.0
BATTERY_MAX_KWH = 10800.0
BATTERY_POWER_KW = 5000.0
INITIAL_SOC_KWH = 6000.0


@dataclass(frozen=True)
class ModelResult:
    interval_labels: list[str]
    purchase_kwh: list[float]
    charge_kwh: list[float]
    discharge_kwh: list[float]
    soc_kwh: list[float]
    curtailed_kwh: list[float]
    total_purchase_kwh: float
    total_cost_yuan: float
    max_balance_error_kwh: float
    min_soc_kwh: float
    max_soc_kwh: float
    max_charge_kwh: float
    max_discharge_kwh: float


def read_input(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_processed_attachment1(path)
    return (
        data["price"].to_numpy(dtype=float),
        data["load_kwh"].to_numpy(dtype=float),
        data["pv_kwh"].to_numpy(dtype=float),
    )


def solve_dispatch(
    price: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
) -> ModelResult:
    n = len(price)
    if not (len(load_kwh) == len(pv_kwh) == n == PERIODS_PER_DAY):
        raise ValueError("优化输入必须包含144个时段")

    # Variable blocks: grid purchase, charge, discharge, curtailment, and SOC.
    g = slice(0, n)
    c = slice(n, 2 * n)
    d = slice(2 * n, 3 * n)
    w = slice(3 * n, 4 * n)
    s = slice(4 * n, 4 * n + n + 1)
    variable_count = 5 * n + 1

    objective = np.zeros(variable_count)
    objective[g] = price
    # A tiny secondary penalty removes otherwise irrelevant charge/discharge ties.
    objective[c] = 1e-9
    objective[d] = 1e-9

    equality_rows: list[np.ndarray] = []
    equality_rhs: list[float] = []

    for t in range(n):
        row = np.zeros(variable_count)
        row[g.start + t] = 1.0
        row[c.start + t] = -1.0
        row[d.start + t] = 1.0
        row[w.start + t] = -1.0
        equality_rows.append(row)
        equality_rhs.append(float(load_kwh[t] - pv_kwh[t]))

        soc_row = np.zeros(variable_count)
        soc_row[s.start + t + 1] = 1.0
        soc_row[s.start + t] = -1.0
        soc_row[c.start + t] = -ETA
        soc_row[d.start + t] = 1.0 / ETA
        equality_rows.append(soc_row)
        equality_rhs.append(0.0)

    initial_row = np.zeros(variable_count)
    initial_row[s.start] = 1.0
    equality_rows.append(initial_row)
    equality_rhs.append(INITIAL_SOC_KWH)

    terminal_row = np.zeros(variable_count)
    terminal_row[s.start + n] = 1.0
    equality_rows.append(terminal_row)
    equality_rhs.append(INITIAL_SOC_KWH)

    max_battery_energy = BATTERY_POWER_KW * DT_HOURS
    bounds = (
        [(0.0, None)] * n
        + [(0.0, max_battery_energy)] * n
        + [(0.0, max_battery_energy)] * n
        + [(0.0, None)] * n
        + [(BATTERY_MIN_KWH, BATTERY_MAX_KWH)] * (n + 1)
    )

    solution = linprog(
        objective,
        A_eq=np.vstack(equality_rows),
        b_eq=np.asarray(equality_rhs),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"线性规划求解失败: {solution.message}")

    x = solution.x
    purchase = x[g]
    charge = x[c]
    discharge = x[d]
    curtailment = x[w]
    soc = x[s]
    balance_error = purchase + pv_kwh + discharge - load_kwh - charge - curtailment

    return ModelResult(
        interval_labels=build_interval_labels(),
        purchase_kwh=purchase.tolist(),
        charge_kwh=charge.tolist(),
        discharge_kwh=discharge.tolist(),
        soc_kwh=soc.tolist(),
        curtailed_kwh=curtailment.tolist(),
        total_purchase_kwh=float(purchase.sum()),
        total_cost_yuan=float(np.dot(price, purchase)),
        max_balance_error_kwh=float(np.max(np.abs(balance_error))),
        min_soc_kwh=float(soc.min()),
        max_soc_kwh=float(soc.max()),
        max_charge_kwh=float(charge.max()),
        max_discharge_kwh=float(discharge.max()),
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "C题" / "数据" / "processed_attachment1.csv"
    output_path = project_root / "C题" / "结果" / "q1_solution.json"
    price, load_kwh, pv_kwh = read_input(input_path)
    result = solve_dispatch(price, load_kwh, pv_kwh)

    if result.max_balance_error_kwh > 1e-7:
        raise AssertionError("能量平衡校验失败")
    if result.min_soc_kwh < BATTERY_MIN_KWH - 1e-7:
        raise AssertionError("储能下限校验失败")
    if result.max_soc_kwh > BATTERY_MAX_KWH + 1e-7:
        raise AssertionError("储能上限校验失败")

    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
