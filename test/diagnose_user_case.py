"""Read-only reproduction of the user's 100-point Sobol workbook and UI settings."""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openpyxl import load_workbook
from PyQt6.QtWidgets import QApplication
from models import ProjectData, FactorItem, ResponseItem
from pages.optimization_page import OptimizationPage
import optimizer_engine as oe

APP = QApplication.instance() or QApplication([])


def make_user_page():
    workbook = load_workbook(ROOT / "test/ADAD_with_responses.xlsx", read_only=True, data_only=True)
    rows = list(workbook.active.values)[1:]
    workbook.close()
    matrix = [{"Run_ID": r[0], "X1": r[1], "X2": r[2], "Y1": r[3], "Y2": r[4]} for r in rows]
    project = ProjectData(
        factors=[FactorItem(name=n, param1="0", param2="4") for n in ("X1", "X2")],
        responses=[ResponseItem(name="Y1", kind="目标", feature="望小"),
                   ResponseItem(name="Y2", kind="约束", feature="望小", robust_limit="0")],
        design_method="代理模型建模 · Kriging",
        surrogate_config={"model": "Kriging", "sample_count": 100,
                          "doe_method": "低差异序列 SOBOL",
                          "bounds": {"X1": [0, 4], "X2": [0, 4]},
                          "model_params": {"correlation": "Gaussian", "trend": "常数（Ordinary）"}},
        doe_matrix=matrix,
    )
    return OptimizationPage(project)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    page = make_user_page()
    page._run_surrogate()
    context, errors = page._build_surrogate_context()
    assert context and not errors, errors
    report = {"samples": len(page.project_data.doe_matrix), "models": {}, "evaluations": {}}
    for name, m in context["models"].items():
        report["models"][name] = {"rmse": m["rmse"], "scale": oe._response_scale(m),
                                  "params": m["validation"]["params"], "r2_test": m["validation"]["r2_test"]}
    for kd in (1.0, 6.0):
        for name, xy in (("screen_point", (0, .559369)), ("broad", (2, 2)), ("narrow", (3.9, 3.9))):
            x = dict(zip(context["names"], xy))
            report["evaluations"][f"k{kd}_{name}"] = oe.evaluate_design(context, x, kd, .5, use_constraints=True, k_constraint=1)
    if args.optimize:
        report["optimization"] = {}
        for kd in (1.0, 6.0):
            result = oe.robust_optimize(context, mode="multi", pop=80, gen=160,
                                        k_design=kd, k_constraint=1, weight=1, seed=7)
            report["optimization"][str(kd)] = {"best": result["best"], "front_count": len(result["front"])}
            print(json.dumps({"k_design": kd, "best": result["best"]}), flush=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["models"]), flush=True)


if __name__ == "__main__":
    main()
