import os
import sys
import unittest
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication
from models import ProjectData, FactorItem, ResponseItem
import optimizer_engine as oe
import workbench_analysis as wa
from notifications import Notice, bus
from main_window import MainWindow
from pages.optimization_page import OptimizationPage

APP = QApplication.instance() or QApplication([])

def quadratic():
    p = ProjectData(factors=[FactorItem(name='X', param1='0', param2='1')],
                    responses=[ResponseItem(name='Y', feature='望大')],
                    design_method='响应曲面设计',
                    doe_matrix=[{'X':float(x), 'Y':float(x*x)} for x in np.linspace(0,1,12)])
    context, errors = oe.build_models(p)
    assert not errors, errors
    return p, context

class ConstraintTests(unittest.TestCase):
    def test_analytic_boundary_and_mean_only_objective(self):
        _, c = quadratic()
        r = oe.robust_optimize(c, mode='constraint', sigma_limits={'Y':.1}, pop=24, gen=45, seed=8)
        sigma = 1/12
        expected = np.sqrt((.1**2 - 2*sigma**4)/(4*sigma**2))
        self.assertAlmostEqual(r['best']['x']['X'], expected, delta=.005)
        self.assertEqual(len(r['best']['obj']), 1)
        self.assertIsNone(r['knee'])
        self.assertLessEqual(r['best']['responses'][0]['sigma'], .1+1e-9)
        self.assertNotIn('sigma_limits', c)

    def test_impossible_limit_does_not_claim_feasible(self):
        _, c = quadratic()
        r = oe.robust_optimize(c, mode='constraint', sigma_limits={'Y':.001}, pop=20, gen=20)
        self.assertEqual(r['feasible_count'], 0)
        self.assertGreater(r['best']['infeas'], 0)
        self.assertFalse(r['front'])

    def test_mean_keeps_improving_beyond_sampled_range(self):
        _, c = quadratic()
        c['_responses'][0].feature = '望小'
        c['models']['Y']['predicted'] = [.5, 1.0]
        r = oe.robust_optimize(c, mode='constraint', sigma_limits={'Y':1}, pop=20, gen=30)
        self.assertLess(r['best']['x']['X'], .01)

    def test_invalid_bound_is_nonmodal_and_leaves_page_usable(self):
        p, _ = quadratic()
        page = OptimizationPage(p)
        page.robust_mode_combo.setCurrentText('约束型')
        page.run_workbench()
        self.assertIsNone(getattr(page, '_worker', None))
        self.assertIn('标准差', page.status_label.text())
        self.assertTrue(page.isEnabled())
        self.assertIn('sigma_limits', page._analysis_settings())
        page.run_statistics()
        self.assertTrue(p.workbench_result)
        page.close()

    def test_missing_nonfinite_and_wrong_response_limits_rejected(self):
        _, c = quadratic()
        for limits in ({}, {'Y':-1}, {'Y':float('nan')}, {'Z':1}):
            with self.assertRaises(ValueError):
                oe.robust_optimize(c, mode='constraint', sigma_limits=limits)

    def test_point_check_uses_same_sigma_limit(self):
        _, c = quadratic()
        c['sigma_limits'] = {'Y':.1}
        r = wa.point_analysis(c, {'X':.9}, 6, 6)
        self.assertFalse(r['feasible'])
        self.assertEqual(r['responses'][0]['判定'], '违反')
        self.assertLess(r['responses'][0]['波动约束余量'], 0)

    def test_message_panel_and_no_project_browser(self):
        window = MainWindow()
        Notice.information(window, '测试保存', '操作完成')
        self.assertIn('测试保存', window.message_view.toPlainText())
        self.assertIsNone(window.findChild(type(window.centralWidget()), 'project_explorer'))
        window.message_filter.setCurrentText('错误')
        self.assertNotIn('测试保存', window.message_view.toPlainText())
        window._clear_messages()
        self.assertEqual(window.message_view.toPlainText(), '')
        window.close()

    def test_async_solver_progress_and_publication(self):
        p, _ = quadratic()
        page = OptimizationPage(p)
        page.pop_spin.setValue(20)
        page.gen_spin.setValue(30)
        page.robust_mode_combo.setCurrentText('约束型')
        page.sigma_limits_table.item(0,1).setText('0.1')
        messages = []
        def receive(*args): messages.append(args)
        bus.posted.connect(receive)
        try:
            page.run_workbench()
            deadline = time.monotonic()+30
            while getattr(page, '_worker', None) is not None and time.monotonic() < deadline:
                APP.processEvents()
                time.sleep(.01)
            self.assertIsNone(page._worker)
            self.assertTrue(p.workbench_result['optimization']['best'])
            self.assertTrue(any('代' in m[2] for m in messages))
            self.assertTrue(page.isEnabled())
        finally:
            bus.posted.disconnect(receive)
            if getattr(page, '_worker', None):
                page._worker.wait()
                APP.processEvents()
            page.close()

if __name__ == '__main__':
    unittest.main()
