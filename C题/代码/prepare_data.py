"""Validate attachments 1 and 2 and write reproducible data-audit outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_processing import (
    load_attachment1,
    load_attachment2,
    write_attachment1_outputs,
    write_attachment2_report,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    c_root = project_root / "C题"
    parser = argparse.ArgumentParser(description="按统一十分钟口径处理C题附件数据")
    parser.add_argument("--attachment1", type=Path, default=c_root / "附件" / "附件1.xlsx")
    parser.add_argument("--attachment2", type=Path, default=c_root / "附件" / "附件2.xlsx")
    parser.add_argument("--out-dir", type=Path, default=c_root / "数据")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attachment1 = load_attachment1(args.attachment1)
    attachment2 = load_attachment2(args.attachment2)
    write_attachment1_outputs(attachment1, args.out_dir, args.attachment1)
    write_attachment2_report(attachment2, args.out_dir, args.attachment2)
    print("数据处理完成。")
    print(f"输出目录：{args.out_dir.resolve()}")
    print("统一时段：t=1,...,144；0:10对应0:10-0:20")


if __name__ == "__main__":
    main()
