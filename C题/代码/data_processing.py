"""Shared data contract and readers for Problem C.

All optimization code uses the same 10-minute indexing convention:

* ``t = 1, ..., 144``;
* source label ``0:10`` is the start of interval ``0:10-0:20``;
* source label ``0:00+1`` is the start of the final interval
  ``0:00+1-0:10+1``;
* power in kW is multiplied by ``1/6`` before entering an energy balance.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INTERVAL_MINUTES = 10
PERIODS_PER_DAY = 144
DT_HOURS = INTERVAL_MINUTES / 60.0
EXPECTED_DAYS = 365
ATTACHMENT1_COLUMNS = ["时间", "电价", "小区负载", "光伏发电预测功率"]
ATTACHMENT2_SHEETS = ("小区负载", "光伏发电实际功率")
MODEL_COLUMNS = [
    "t",
    "price",
    "load_kw",
    "pv_kw",
    "load_kwh",
    "pv_kwh",
    "net_load_kw",
    "net_load_kwh",
]


@dataclass(frozen=True)
class AnnualPowerData:
    """Validated annual power observations from attachment 2."""

    dates: tuple[dt.date, ...]
    load_kw: np.ndarray
    pv_actual_kw: np.ndarray

    @property
    def load_kwh(self) -> np.ndarray:
        return self.load_kw * DT_HOURS

    @property
    def pv_actual_kwh(self) -> np.ndarray:
        return self.pv_actual_kw * DT_HOURS


def parse_source_time(value: object) -> int:
    """Convert an attachment time label to minutes after the day's 00:00."""
    if isinstance(value, dt.datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, dt.time):
        return value.hour * 60 + value.minute

    text = str(value).strip().replace("：", ":")
    day_offset = 0
    if text.endswith("+1"):
        text = text[:-2]
        day_offset = 24 * 60

    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"无法解析时间标签：{value!r}")
    hour, minute = (int(part) for part in parts)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"时间标签超出范围：{value!r}")
    return day_offset + hour * 60 + minute


def expected_source_minutes() -> list[int]:
    """Return the required source time sequence: 10, 20, ..., 1440."""
    return [INTERVAL_MINUTES * t for t in range(1, PERIODS_PER_DAY + 1)]


def validate_time_axis(values: list[object], source_name: str) -> list[int]:
    """Validate a complete attachment time axis and return minute offsets."""
    if len(values) != PERIODS_PER_DAY:
        raise ValueError(
            f"{source_name}应包含{PERIODS_PER_DAY}个十分钟时段，"
            f"实际为{len(values)}个"
        )
    actual = [parse_source_time(value) for value in values]
    expected = expected_source_minutes()
    if actual != expected:
        mismatches = [
            (index + 1, observed, required)
            for index, (observed, required) in enumerate(zip(actual, expected))
            if observed != required
        ]
        raise ValueError(
            f"{source_name}时间轴不符合10,20,...,1440分钟的统一规范。"
            f"前几处不一致：{mismatches[:5]}"
        )
    return actual


def format_clock(minutes: int) -> str:
    """Format a minute offset using the competition's ``+1`` suffix."""
    day_offset, minute_of_day = divmod(minutes, 24 * 60)
    hour, minute = divmod(minute_of_day, 60)
    suffix = f"+{day_offset}" if day_offset else ""
    return f"{hour}:{minute:02d}{suffix}"


def build_interval_labels() -> list[str]:
    """Build the 144 result labels corresponding to the source timestamps."""
    return [
        f"{format_clock(start)}-{format_clock(start + INTERVAL_MINUTES)}"
        for start in expected_source_minutes()
    ]


def _numeric_frame(frame: pd.DataFrame, columns: list[str], source_name: str) -> pd.DataFrame:
    result = frame[columns].copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.isna().any().any():
        raise ValueError(f"{source_name}存在空值或非数值：{result.isna().sum().to_dict()}")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{source_name}存在NaN或无穷值")
    return result


def prepare_attachment1(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate attachment 1 and return the standard eight-column model table."""
    columns = [str(column).strip() for column in raw.columns]
    if columns != ATTACHMENT1_COLUMNS:
        raise ValueError(
            "附件1列名不符合统一规范。\n"
            f"实际列名：{columns}\n预期列名：{ATTACHMENT1_COLUMNS}"
        )
    if len(raw) != PERIODS_PER_DAY:
        raise ValueError(
            f"附件1应包含{PERIODS_PER_DAY}行，实际读取到{len(raw)}行"
        )
    validate_time_axis(raw["时间"].tolist(), "附件1")

    data = _numeric_frame(
        raw,
        ["电价", "小区负载", "光伏发电预测功率"],
        "附件1",
    )
    data.columns = ["price", "load_kw", "pv_kw"]
    negative_load = int((data["load_kw"] < 0).sum())
    negative_pv = int((data["pv_kw"] < 0).sum())
    if negative_load or negative_pv:
        raise ValueError(
            f"附件1负载负值时段数={negative_load}，"
            f"光伏负值时段数={negative_pv}"
        )

    data.insert(0, "t", np.arange(1, PERIODS_PER_DAY + 1, dtype=int))
    data["load_kwh"] = data["load_kw"] * DT_HOURS
    data["pv_kwh"] = data["pv_kw"] * DT_HOURS
    data["net_load_kw"] = data["load_kw"] - data["pv_kw"]
    data["net_load_kwh"] = data["net_load_kw"] * DT_HOURS
    return data


def load_attachment1(path: Path) -> pd.DataFrame:
    """Read and validate attachment 1."""
    return prepare_attachment1(pd.read_excel(path))


def load_processed_attachment1(path: Path) -> pd.DataFrame:
    """Read the canonical attachment 1 model table and verify its derivations."""
    data = pd.read_csv(path, encoding="utf-8-sig")
    if list(data.columns) != MODEL_COLUMNS:
        raise ValueError(
            "processed_attachment1.csv字段不符合统一规范。\n"
            f"实际字段：{list(data.columns)}\n预期字段：{MODEL_COLUMNS}"
        )
    if data.shape != (PERIODS_PER_DAY, len(MODEL_COLUMNS)):
        raise ValueError(
            "processed_attachment1.csv应为"
            f"{PERIODS_PER_DAY}行、{len(MODEL_COLUMNS)}列，实际为{data.shape}"
        )
    if not np.array_equal(data["t"].to_numpy(), np.arange(1, PERIODS_PER_DAY + 1)):
        raise ValueError("processed_attachment1.csv的t必须严格为1,...,144")
    numeric = data.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("processed_attachment1.csv存在空值、非数值、NaN或无穷值")
    if np.any(data[["load_kw", "pv_kw"]].to_numpy(dtype=float) < 0):
        raise ValueError("processed_attachment1.csv的负荷和光伏不能为负")

    checks = {
        "load_kwh": data["load_kw"] * DT_HOURS,
        "pv_kwh": data["pv_kw"] * DT_HOURS,
        "net_load_kw": data["load_kw"] - data["pv_kw"],
        "net_load_kwh": data["load_kwh"] - data["pv_kwh"],
    }
    for column, expected in checks.items():
        if not np.allclose(data[column], expected, rtol=0.0, atol=1e-9):
            raise ValueError(f"processed_attachment1.csv的{column}计算不一致")
    return data


def _read_annual_sheet(
    excel: pd.ExcelFile,
    sheet_name: str,
) -> tuple[tuple[dt.date, ...], np.ndarray]:
    frame = pd.read_excel(excel, sheet_name=sheet_name)
    expected_shape = (EXPECTED_DAYS, PERIODS_PER_DAY + 1)
    if frame.shape != expected_shape:
        raise ValueError(
            f"附件2工作表“{sheet_name}”应为{expected_shape[0]}行、"
            f"{expected_shape[1]}列，实际为{frame.shape}"
        )
    validate_time_axis(list(frame.columns[1:]), f"附件2工作表“{sheet_name}”")

    parsed_dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    if parsed_dates.isna().any():
        bad_rows = (np.flatnonzero(parsed_dates.isna().to_numpy()) + 2).tolist()
        raise ValueError(f"附件2工作表“{sheet_name}”日期无效，行号：{bad_rows[:5]}")
    dates = tuple(value.date() for value in parsed_dates)
    expected_dates = tuple(dates[0] + dt.timedelta(days=i) for i in range(len(dates)))
    if dates != expected_dates:
        raise ValueError(f"附件2工作表“{sheet_name}”日期必须连续、升序且不重复")

    numeric = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"附件2工作表“{sheet_name}”存在空值或非数值")
    matrix = numeric.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"附件2工作表“{sheet_name}”存在NaN或无穷值")
    if np.any(matrix < 0):
        raise ValueError(f"附件2工作表“{sheet_name}”不能包含负功率")
    return dates, matrix


def load_attachment2(path: Path) -> AnnualPowerData:
    """Read attachment 2 using the same time and unit contract as attachment 1."""
    with pd.ExcelFile(path) as excel:
        missing = [name for name in ATTACHMENT2_SHEETS if name not in excel.sheet_names]
        if missing:
            raise ValueError(f"附件2缺少工作表：{missing}")
        load_dates, load_kw = _read_annual_sheet(excel, ATTACHMENT2_SHEETS[0])
        pv_dates, pv_kw = _read_annual_sheet(excel, ATTACHMENT2_SHEETS[1])
    if load_dates != pv_dates:
        raise ValueError("附件2负荷与光伏工作表的日期不一致")
    return AnnualPowerData(load_dates, load_kw, pv_kw)


def write_attachment1_outputs(data: pd.DataFrame, out_dir: Path, source_path: Path) -> None:
    """Write the standardized model table, summary statistics and audit report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data[MODEL_COLUMNS].to_csv(
        out_dir / "processed_attachment1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stat_columns = [
        "price",
        "load_kw",
        "pv_kw",
        "net_load_kw",
        "load_kwh",
        "pv_kwh",
        "net_load_kwh",
    ]
    stats = pd.DataFrame(
        {
            "mean": data[stat_columns].mean(),
            "min": data[stat_columns].min(),
            "max": data[stat_columns].max(),
            "std_population": data[stat_columns].std(ddof=0),
            "sum": data[stat_columns].sum(),
        }
    )
    stats.round(6).to_csv(out_dir / "summary_stats.csv", encoding="utf-8-sig")

    labels = build_interval_labels()
    min_index = int(data["net_load_kwh"].idxmin())
    max_index = int(data["net_load_kwh"].idxmax())
    positive_net = float(data["net_load_kwh"].clip(lower=0).sum())
    surplus = float((-data["net_load_kwh"].clip(upper=0)).sum())
    report = f"""数据质量报告
================
数据来源：{source_path.resolve()}
处理后数据形状：{len(data)} 行 × {len(MODEL_COLUMNS)} 列

时间约定
----------------
时段总数：{len(data)}
时间间隔：{INTERVAL_MINUTES} 分钟
第1个时段：t=1，{labels[0]}
第144个时段：t=144，{labels[-1]}
源时间序列：10,20,...,1440 分钟
模型使用：t=1,...,144；时间标签仅用于校验和结果标注

完整性检查
----------------
缺失值和非数值：0
负载负值时段数：0
光伏负值时段数：0
负电价时段数：{int((data['price'] < 0).sum())}
光伏为0的时段数：{int((data['pv_kw'] == 0).sum())}
净负荷为负的时段数：{int((data['net_load_kw'] < 0).sum())}

能量统计
----------------
全天负载电量：{data['load_kwh'].sum():.6f} kWh
全天光伏预测电量：{data['pv_kwh'].sum():.6f} kWh
全天净负荷电量：{data['net_load_kwh'].sum():.6f} kWh
净负荷为正的总电量：{positive_net:.6f} kWh
光伏富余总电量：{surplus:.6f} kWh

净负荷极值
----------------
最小净负荷：{data.loc[min_index, 'net_load_kwh']:.6f} kWh，对应 t={min_index + 1}
最大净负荷：{data.loc[max_index, 'net_load_kwh']:.6f} kWh，对应 t={max_index + 1}

单位说明
----------------
price：元/kWh
load_kw、pv_kw、net_load_kw：kW
load_kwh、pv_kwh、net_load_kwh：kWh
每个十分钟时段的功率转电量系数：1/6
"""
    (out_dir / "data_quality_report.txt").write_text(report, encoding="utf-8")


def write_attachment2_report(data: AnnualPowerData, out_dir: Path, source_path: Path) -> None:
    """Write a compact audit report without duplicating the full annual workbook."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = build_interval_labels()
    report = f"""附件2数据质量报告
==================
数据来源：{source_path.resolve()}
日期范围：{data.dates[0].isoformat()} 至 {data.dates[-1].isoformat()}
日期数：{len(data.dates)}
每日时段数：{PERIODS_PER_DAY}
第1个时段：t=1，{labels[0]}
第144个时段：t=144，{labels[-1]}

完整性检查
------------------
负荷与光伏日期一致：是
日期连续、升序且不重复：是
两张工作表时间轴一致：是
缺失值、非数值、NaN或无穷值：0
负荷负值记录数：0
光伏负值记录数：0

单位换算
------------------
原始负荷与光伏：kW
模型负荷与光伏：kWh/10分钟
换算系数：1/6

全年统计
------------------
负荷总电量：{data.load_kwh.sum():.6f} kWh
光伏总电量：{data.pv_actual_kwh.sum():.6f} kWh
负荷功率范围：{data.load_kw.min():.6f} 至 {data.load_kw.max():.6f} kW
光伏功率范围：{data.pv_actual_kw.min():.6f} 至 {data.pv_actual_kw.max():.6f} kW
"""
    (out_dir / "attachment2_quality_report.txt").write_text(report, encoding="utf-8")
