"""筛选设计与田口设计的专项回归测试。"""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import FactorItem, ProjectData, ResponseItem
import doe_engine as doe
from pages.optimization_page import OptimizationPage
from PyQt6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP = QApplication.instance() or QApplication([])


def test_screening_is_sensitivity_analysis():
    factors = [("X1", 0.0, 1.0, None), ("X2", 0.0, 1.0, None)]
    rows, meta = doe.build_screening(factors, "full", seed=3, randomize=False)
    assert meta["total_runs"] == 4
    values = [{**row, "Y": 3.0 * row["X1"] + row["X2"]} for row in rows]
    effects = doe.screening_from_responses(
        [FactorItem(name="X1", param1="0", param2="1"),
         FactorItem(name="X2", param1="0", param2="1")],
        values,
        "Y",
    )
    assert abs(effects["X1"] - 3.0) < 1e-12
    assert abs(effects["X2"] - 1.0) < 1e-12
    assert abs(effects["X1"]) > abs(effects["X2"])


def test_taguchi_cross_table_and_snr():
    controls = [("X1", 0.0, 1.0, None), ("X2", 0.0, 1.0, None)]
    noise = [("N1", -1.0, 1.0, None)]
    rows, meta = doe.build_taguchi(controls, noise, "L4", "L4", 2, seed=4)
    assert len(rows) == 16
    assert meta["inner_runs"] == 4 and meta["outer_runs"] == 4
    assert len(meta["groups"]) == len(rows)
    assert {item["inner"] for item in meta["groups"]} == {0, 1, 2, 3}
    assert {item["outer"] for item in meta["groups"]} == {0, 1, 2, 3}

    project = ProjectData(
        factors=[FactorItem(name="X1", source="设计", param1="0", param2="1"),
                 FactorItem(name="X2", source="设计", param1="0", param2="1"),
                 FactorItem(name="N1", source="环境", param1="-1", param2="1")],
        responses=[ResponseItem(name="Y", kind="目标", feature="望大")],
        design_method="田口稳健设计（内表×外表）",
        design_params={"meta": meta, "snr_mode": "按各响应目标特征自动", "levels": 2},
        doe_matrix=[
            {**row, "Y": 10.0 + 2.0 * row["X1"] + 0.2 * row["X2"] + 0.5 * row["N1"]}
            for row in rows
        ],
        taguchi_result={
            "array_name": meta["inner_label"], "outer_label": meta["outer_label"],
            "inner_runs": meta["inner_runs"], "outer_runs": meta["outer_runs"],
            "has_outer": True, "levels": 2,
            "factor_levels": meta["factor_levels"], "groups": meta["groups"],
        },
    )
    page = OptimizationPage(project)
    page._run_taguchi()
    assert project.taguchi_result["signal_to_noise"]
    assert project.taguchi_result["analysis_status"].startswith("S/N 计算完成")
    assert "X1" in project.taguchi_result["best_levels"]


if __name__ == "__main__":
    test_screening_is_sensitivity_analysis()
    test_taguchi_cross_table_and_snr()
    print("screening/taguchi regression passed")
