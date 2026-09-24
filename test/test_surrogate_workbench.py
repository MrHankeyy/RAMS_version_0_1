"""Exercise the complete Qt workbench, including post-optimization diagnostics."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication
from models import ProjectData, FactorItem, ResponseItem
from pages.optimization_page import OptimizationPage
import optimizer_engine as oe
import doe_engine as de
import numpy as np

APP = QApplication.instance() or QApplication([])


class SurrogateWorkbenchTest(unittest.TestCase):
    def make_page(self, model="Kriging"):
        project = ProjectData(
            factors=[FactorItem(name="X", param1="0", param2="4")],
            responses=[ResponseItem(name="Y", feature="望小")],
            design_method=f"代理模型建模 · {model}",
            surrogate_config={"model": model, "bounds": {"X": [0, 4]}},
            doe_matrix=[{"X": i / 5, "Y": 1 + (i / 5 - 2) ** 2} for i in range(21)],
        )
        page = OptimizationPage(project)
        page.pop_spin.setValue(20)
        page.gen_spin.setValue(10)
        return page

    def test_complete_button_flow_and_model_reuse(self):
        for model in ("Kriging", "SVR", "ANN"):
            with self.subTest(model=model):
                page = self.make_page(model)
                page._run_surrogate()
                trained = page.project_data.surrogate_result["responses"]["Y"]["predict_fn"]
                with patch.object(page, "_run_surrogate", side_effect=AssertionError("unexpected retraining")), \
                     patch.object(oe, "build_models", side_effect=AssertionError("quadratic fallback")), \
                     patch("pages.optimization_page.QMessageBox.information"), \
                     patch("pages.optimization_page.QMessageBox.warning") as warning:
                    page.run_all_btn.click()
                    warning.assert_not_called()
                self.assertIn("分析完成", page.status_label.text())
                self.assertIn("留出集", page.cv_text.toPlainText())
                self.assertIn(model, page.model_text.toPlainText())
                self.assertAlmostEqual(oe.model_predict(page._oe_context["models"]["Y"], [1.3]), trained([1.3 / 4]))
                self.assertTrue(page.run_all_btn.isEnabled())
                page.close()

    def test_data_change_invalidates_training(self):
        page = self.make_page()
        page._run_surrogate()
        page.project_data.doe_matrix[0]["Y"] += 1
        with patch.object(page, "_run_surrogate", wraps=page._run_surrogate) as train, \
             patch("pages.optimization_page.QMessageBox.information"):
            page._execute_workbench()
            train.assert_called_once()
        page.close()

    def test_exception_does_not_escape_button(self):
        page = self.make_page()
        with patch.object(page, "_execute_workbench", side_effect=ValueError("invalid data")), \
             patch("pages.optimization_page.QMessageBox.warning") as warning, \
             self.assertLogs(level="ERROR"):
            page.run_all_btn.click()
            warning.assert_called_once()
        self.assertIn("分析失败", page.status_label.text())
        self.assertTrue(page.run_all_btn.isEnabled())
        page.close()

    def test_kriging_prediction_is_not_clipped_to_training_range(self):
        x = np.array([[v] for v in (-1, -.8, -.6, -.4, .4, .6, .8, 1)])
        result = de.train_kriging(x, x[:, 0] ** 2, [[0]], trend="二次趋势")
        self.assertAlmostEqual(result["predict_fn"]([0]), 0, places=6)
        self.assertAlmostEqual(result["predicted"][0], 0, places=6)
        model = {"predict_fn": result["predict_fn"]}
        self.assertAlmostEqual(oe.model_hess_diag_x(model, [0])[0], 2, places=4)

    def test_failed_response_blocks_partial_optimization(self):
        page = self.make_page()
        page.project_data.responses.append(ResponseItem(name="missing", kind="约束"))
        page._run_surrogate()
        context, errors = page._build_surrogate_context()
        self.assertIsNone(context)
        self.assertTrue(any("missing" in error for error in errors))
        page.close()

    def test_equal_front_scores_prefer_lower_variance(self):
        unstable = (None, (0.0, 10.0), None)
        stable = (None, (0.02, 0.4), None)
        self.assertIs(oe._knee_point([unstable, stable]), stable)
        self.assertIs(oe._knee_point([stable, unstable]), stable)

    def test_svr_large_training_set_keeps_real_solver(self):
        x = np.linspace(0, 1, 80).reshape(-1, 1)
        result = de.train_svr(x, x[:, 0] ** 2, x[:7],
                              c_min=1, c_max=1, g_min=1, g_max=1)
        self.assertEqual(result["params"]["solver"], "scikit-learn SVR")
        np.testing.assert_allclose([result["predict_fn"](v) for v in x[:7]], result["predicted"])

    def test_svr_fallback_predictor_matches_selected_model(self):
        x = np.linspace(0, 1, 12).reshape(-1, 1)
        with patch.dict(sys.modules, {"sklearn.svm": None}):
            result = de.train_svr(x, np.sin(x[:, 0] * 6), x[:4],
                                  c_min=1, c_max=2, c_step=1,
                                  g_min=.1, g_max=2.1, g_step=1)
        np.testing.assert_allclose([result["predict_fn"](v) for v in x[:4]], result["predicted"])

    def constraint_context(self, role="约束"):
        factors = [FactorItem(name="X", param1="0", param2="4")]
        responses = [ResponseItem(name="Y", kind="目标", feature="望小"),
                     ResponseItem(name="G", kind=role, feature="望小", robust_limit="0")]
        y = {"predict_fn": lambda x: (x[0] - 3) ** 2,
             "grad_fn": lambda x: np.array([2 * (x[0] - 3)]),
             "hess_diag_fn": lambda x: np.array([2.0]),
             "predicted": [0, 9], "rmse": 1e-12, "residual_std": 0}
        g = {"predict_fn": lambda x: x[0] - 1,
             "grad_fn": lambda x: np.array([1.0]),
             "hess_diag_fn": lambda x: np.array([0.0]),
             "predicted": [-1, 3], "rmse": 1e-14, "residual_std": 0}
        return {"inputs": factors, "names": ["X"], "models": {"Y": y, "G": g},
                "_responses": responses, "_project": ProjectData(factors=factors, responses=responses)}

    def test_all_modes_honor_constraint_roles(self):
        context = self.constraint_context()
        for mode in ("multi", "weighted", "constraint"):
            result = oe.robust_optimize(context, mode=mode, pop=20, gen=30,
                                        k_design=6, k_constraint=1, seed=7)
            self.assertLessEqual(result["best"]["x"]["X"], 2 / 3 + 1e-5, mode)
            self.assertEqual(result["objective_names"], ["Y"])
            self.assertEqual(result["constraint_names"], ["G"])
            self.assertLess(result["best"]["std_norm"], 1)

    def test_combined_role_keeps_both_objective_and_constraint(self):
        context = self.constraint_context("目标+约束")
        result = oe.evaluate_design(context, {"X": 2}, 6, .5, use_constraints=True, k_constraint=1)
        self.assertEqual([r["name"] for r in result["objectives"]], ["Y", "G"])
        self.assertEqual([r["name"] for r in result["constraints"]], ["G"])
        self.assertGreater(result["infeas"], 0)

    def test_scale_does_not_depend_on_fitting_error(self):
        context = self.constraint_context()
        first = oe.evaluate_design(context, {"X": .5}, 6, .5, use_constraints=True)
        for m in context["models"].values():
            m["rmse"] = 100
        second = oe.evaluate_design(context, {"X": .5}, 6, .5, use_constraints=True)
        self.assertEqual(first["obj"], second["obj"])
        self.assertEqual(oe._response_scale({"predicted": [0, 0], "rmse": 0}), 1)

    def test_dialog_flat_parameters_reach_training(self):
        x = np.linspace(0, 1, 20).reshape(-1, 1)
        result = de.train_surrogate(x, x[:, 0] * 3, model="Kriging",
                                     model_params={"trend": "一次趋势", "correlation": "Gaussian"})
        self.assertNotIn("error", result)
        self.assertEqual(result["params"]["trend"], "一次趋势")

    def test_input_k_means_standard_deviation_not_search_bound(self):
        factor = FactorItem(name="X", param1="0", param2="4")
        self.assertEqual(oe._sigma_of_factor(factor, 1), 2)
        self.assertAlmostEqual(oe._sigma_of_factor(factor, 6), 1 / 3)


if __name__ == "__main__":
    unittest.main()
