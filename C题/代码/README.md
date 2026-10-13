# 代码说明

微网电力调控优化全部求解代码。所有问题共用同一物理约束框架，
仅通过逐步扩大的信息集区分。

## 模块

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 统一预处理入口：读取附件 1、2，生成模型输入数据与质量校验 |
| `data_processing.py` | 全篇共用的数据契约、时间口径与校验 |
| `forecast_q2.py` | 问题二光伏因果预测 |
| `forecast_price_q4.py` | 问题四价格预测与发布时刻索引 |
| `solve_q1.py` | 问题一：确定性线性规划调度 |
| `solve_q2.py` | 问题二：滚动修正 + CVaR 风险调度 |
| `solve_q3.py` | 问题三：发布时刻滚动修正 |
| `solve_q4.py` | 问题四：联合场景风险管理（q4_2 / q4_3） |
| `export_results.py` | 问题一、二结果写入官方模板（result1/2.xlsx） |
| `export_q3.py` | 问题三结果写入官方模板（result3 / result3_final.xlsx） |
| `export_q4.py` | 问题四结果写入官方模板（result4-2 / result4-3.xlsx） |

## 运行顺序

```bash
cd C题/代码
python prepare_data.py     # 1. 预处理（生成模型输入数据）
python solve_q1.py         # 2. 问题一求解
python solve_q2.py         # 3. 问题二求解（全年 334 天）
python solve_q3.py         # 4. 问题三求解
python solve_q4.py         # 5. 问题四求解
python export_results.py   # 6. 导出 result1/2.xlsx
python export_q3.py        # 7. 导出 result3 / result3_final.xlsx
python export_q4.py        # 8. 导出 result4-2 / result4-3.xlsx
```

依赖 Python 3.8+，需要 numpy、pandas、scipy、openpyxl。

## 时间口径

附件时间标签为区间起点（如第一行 `0:10` 表示 `0:10-0:20` 时段），
每天 144 个 10 分钟时段；评估期 2025-02-01 至 2025-12-31 共 334 天，
1 月仅用于暖机传递储能状态。该口径由 `data_processing.py` 统一实现。
