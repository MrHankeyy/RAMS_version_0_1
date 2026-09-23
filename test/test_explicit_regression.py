"""显式函数端到端回归测试。

测试链路：DOE 生成 -> 显式函数回填响应 -> 响应面/代理模型建模 -> 鲁棒优化。
运行：python test/test_explicit_regression.py
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import doe_engine as doe
import optimizer_engine as optimizer
from models import FactorItem, ProjectData, ResponseItem


def explicit_objective(x1, x2):
    """用于建模精度测试的显式二次目标函数。"""
    return 4.0 + (x1 - 2.0) ** 2 + 1.5 * (x2 - 1.5) ** 2 + 0.4 * x1 * x2


def explicit_constraint(x1, x2):
    """约束 g(x)<=0；最优可行点应满足 x1+x2<=3.5。"""
    return x1 + x2 - 3.5


def bimodal_objective(x):
    """双盆地目标：A 深而窄，B 浅而宽。目标是最小化。"""
    x1, x2 = x
    background = 20.0 + 0.1 * ((x1 - 5.0) ** 2 + (x2 - 5.0) ** 2)
    narrow = -12.0 * math.exp(-0.5 * ((x1 - 1.5) ** 2 + (x2 - 1.5) ** 2) / 0.35 ** 2)
    broad = -5.0 * math.exp(-0.5 * ((x1 - 3.6) ** 2 + (x2 - 3.6) ** 2) / 1.8 ** 2)
    return background + narrow + broad


def bimodal_gradient(x):
    x1, x2 = x
    points = np.array([x1, x2], dtype=float)
    gradient = 0.2 * (points - 5.0)
    for center, amplitude, width in (
        (np.array([1.5, 1.5]), 12.0, 0.35),
        (np.array([3.6, 3.6]), 5.0, 1.8),
    ):
        delta = points - center
        gaussian = math.exp(-0.5 * float(delta @ delta) / width ** 2)
        gradient += amplitude * gaussian * delta / width ** 2
    return gradient


def bimodal_hessian_diag(x):
    x1, x2 = x
    points = np.array([x1, x2], dtype=float)
    hessian = np.full(2, 0.2)
    for center, amplitude, width in (
        (np.array([1.5, 1.5]), 12.0, 0.35),
        (np.array([3.6, 3.6]), 5.0, 1.8),
    ):
        delta = points - center
        gaussian = math.exp(-0.5 * float(delta @ delta) / width ** 2)
        hessian += amplitude * gaussian * (1.0 / width ** 2 - delta ** 2 / width ** 4)
    return hessian


def make_analytic_model(predict_fn, grad_fn, hess_fn, bounds):
    grid = np.linspace(bounds[0], bounds[1], 61)
    predicted = [predict_fn(np.array([x1, x2])) for x1 in grid for x2 in grid]
    return {
        "predict_fn": predict_fn,
        "grad_fn": grad_fn,
        "hess_diag_fn": hess_fn,
        "predicted": predicted,
        "rmse": 1.0,
        "residual_std": 0.0,
    }


def make_factors():
    return [
        FactorItem(name="X1", source="设计", param1="0", param2="4"),
        FactorItem(name="X2", source="设计", param1="0", param2="4"),
    ]


def test_response_surface():
    factors = [("X1", 0.0, 4.0, None), ("X2", 0.0, 4.0, None)]
    rows, meta = doe.build_response_surface(
        factors, "CCD", "face", 1.0, centers=3, seed=7, randomize=False
    )
    assert meta["total_runs"] == 11
    assert len(rows) == 11

    x = np.array([[row["X1"], row["X2"]] for row in rows], dtype=float)
    y = np.array([explicit_objective(*point) for point in x])
    center = np.array([2.0, 2.0])
    span = np.array([2.0, 2.0])
    fit = doe.fit_response_surface((x - center) / span, y)
    assert fit["r2"] > 0.999999, fit
    assert fit["rmse"] < 1e-8, fit
    optimum = doe.quadratic_optimum(fit, [-1, -1], [1, 1], "min", samples=20000)
    assert optimum is not None
    # 含交互项时解析驻点为 x1≈1.742、x2≈1.268，换算到 [-1, 1] 编码空间。
    assert abs(optimum["x"][0] + 0.129) < 0.15, optimum
    assert abs(optimum["x"][1] + 0.366) < 0.15, optimum
    return {"runs": len(rows), "r2": fit["r2"], "rmse": fit["rmse"]}


def test_surrogate_models():
    factors = [("X1", 0.0, 4.0, None), ("X2", 0.0, 4.0, None)]
    rows = doe.build_samples(factors, 48, "最优 LHS", seed=19)
    x = [[row["X1"] / 4.0, row["X2"] / 4.0] for row in rows]
    y = [explicit_objective(row["X1"], row["X2"]) for row in rows]

    configs = {
        "Kriging": {"kriging": {"correlation": "Gaussian", "trend": "一次趋势"}},
        "SVR": {
            "svr": {
                "kernel": "多项式核", "poly_degree": 2,
                "c_min": 1.0, "c_max": 1.0, "c_step": 1.0,
                "g_min": 0.5, "g_max": 0.5, "g_step": 1.0,
            }
        },
        "ANN": {
            "ann": {
                "hidden_layers": 1, "hidden_nodes": 8, "activation": "tanh",
                "learning_rate": 0.02, "epochs": 80,
            }
        },
    }
    results = {}
    for model, params in configs.items():
        result = doe.train_surrogate(x, y, model=model, model_params=params)
        assert "error" not in result, (model, result)
        assert len(result["predicted"]) == result["test_samples"]
        assert math.isfinite(result["r2_test"])
        results[model] = result["params"]
    assert results["Kriging"]["trend"] == "一次趋势"
    assert results["SVR"]["poly_degree"] == 2
    return results


def test_robust_optimization():
    factors = make_factors()
    project = ProjectData(
        factors=factors,
        responses=[
            ResponseItem(name="Y", kind="目标", feature="望小", robust_limit="10"),
            ResponseItem(name="G", kind="约束", feature="望小", robust_limit="0"),
        ],
    )
    rows, _ = doe.build_response_surface(
        [("X1", 0.0, 4.0, None), ("X2", 0.0, 4.0, None)],
        "CCD", "face", 1.0, centers=5, seed=3, randomize=False,
    )
    project.doe_matrix = []
    for index, row in enumerate(rows):
        project.doe_matrix.append({
            "Run_ID": index + 1, "X1": row["X1"], "X2": row["X2"],
            "Y": explicit_objective(row["X1"], row["X2"]),
            "G": explicit_constraint(row["X1"], row["X2"]),
        })

    context, errors = optimizer.build_models(project)
    assert context and context["models"], errors
    assert context["models"]["Y"]["r2"] > 0.999999
    assert context["models"]["G"]["r2"] > 0.999999

    results = {}
    for mode in ("multi", "weighted", "constraint"):
        result = optimizer.robust_optimize(
            context, mode=mode, pop=30, gen=50, weight=0.7,
            k_design=6.0, k_constraint=1.0, seed=11,
        )
        assert "best" in result and result["best"]["x"], (mode, result)
        assert all(0.0 <= value <= 4.0 for value in result["best"]["x"].values())
        assert math.isfinite(result["best"]["mean_perf"])
        assert math.isfinite(result["best"]["std_norm"])
        if mode == "multi":
            assert len(result["front"]) >= 2, result
        if mode == "weighted":
            assert len(result["best"]["obj"]) == 1, result
        if mode == "constraint":
            assert result["best"]["infeas"] <= 1e-6, result
        results[mode] = result["best"]
    return results


def test_bimodal_robust_front():
    """用解析导数严格验证：窄深名义最优点与宽浅鲁棒点应同时出现。"""
    factors = make_factors()
    project = ProjectData(
        factors=factors,
        responses=[
            ResponseItem(name="Y", kind="目标", feature="望小", robust_limit="20"),
            ResponseItem(name="G", kind="约束", feature="望小", robust_limit="8"),
        ],
    )
    model_y = make_analytic_model(
        bimodal_objective, bimodal_gradient, bimodal_hessian_diag, (0.0, 5.0)
    )
    model_g = make_analytic_model(
        lambda x: float(x[0] + x[1] - 8.0),
        lambda _x: np.array([1.0, 1.0]),
        lambda _x: np.array([0.0, 0.0]),
        (0.0, 5.0),
    )
    context = {
        "inputs": factors, "names": ["X1", "X2"],
        "models": {"Y": model_y, "G": model_g},
        "_responses": project.responses, "_project": project,
    }

    narrow = np.array([1.5, 1.5])
    broad = np.array([3.6, 3.6])
    narrow_moments = optimizer.robust_moments(
        model_y, {"X1": narrow[0], "X2": narrow[1]}, context,
        k_design=6.0, include_residual=False
    )
    broad_moments = optimizer.robust_moments(
        model_y, {"X1": broad[0], "X2": broad[1]}, context,
        k_design=6.0, include_residual=False
    )
    assert bimodal_objective(narrow) < bimodal_objective(broad)
    # 鲁棒均值包含曲率修正：窄深点的名义值更低，但扰动后的均值反而更差。
    assert narrow_moments[0] > broad_moments[0]
    assert narrow_moments[1] > broad_moments[1] * 3.0

    multi = optimizer.robust_optimize(
        context, mode="multi", pop=80, gen=140, k_design=6.0,
        k_constraint=1.0, seed=23,
    )
    assert len(multi["front"]) >= 2, multi
    front_points = [np.array([item["x"]["X1"], item["x"]["X2"]])
                    for item in multi["front"]]
    assert any(np.linalg.norm(point - narrow) < 0.65 for point in front_points), multi
    assert any(np.linalg.norm(point - broad) < 1.0 for point in front_points), multi

    stable = optimizer.robust_optimize(
        context, mode="weighted", pop=80, gen=140, weight=0.2,
        k_design=6.0, k_constraint=1.0, seed=23,
    )
    stable_point = np.array([stable["best"]["x"]["X1"], stable["best"]["x"]["X2"]])
    assert np.linalg.norm(stable_point - broad) < 1.0, stable
    assert stable["best"]["std_norm"] < narrow_moments[1]

    constrained = optimizer.robust_optimize(
        context, mode="constraint", pop=80, gen=140, k_design=6.0,
        k_constraint=1.0, seed=23,
    )
    assert constrained["best"]["infeas"] <= 1e-8, constrained
    return {
        "narrow": {"mean": narrow_moments[0], "sigma": narrow_moments[1]},
        "broad": {"mean": broad_moments[0], "sigma": broad_moments[1]},
        "pareto_size": len(multi["front"]),
        "stable_point": stable["best"]["x"],
        "constraint_point": constrained["best"]["x"],
    }


def main():
    results = {
        "RSM": test_response_surface(),
        "surrogates": test_surrogate_models(),
        "optimization": test_robust_optimization(),
        "bimodal_robust_front": test_bimodal_robust_front(),
    }
    print("explicit regression passed")
    print(results)


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    main()
