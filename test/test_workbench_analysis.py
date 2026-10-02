import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from scipy import stats
from PyQt6.QtWidgets import QApplication,QMessageBox,QFileDialog
import optimizer_engine as oe
import workbench_analysis as wa
from models import ProjectData,FactorItem,ResponseItem
from pages.optimization_page import OptimizationPage

APP=QApplication.instance() or QApplication([])


def make_context(noise=True,limit=None):
    rng=np.random.default_rng(31)
    x=rng.uniform(-1,1,(35,2));eps=rng.normal(0,.08,len(x)) if noise else np.zeros(len(x))
    rows=[{'A':float(a),'B':float(b),'Y':float(2-a+2*b*b+.3*a*b+e),'G':float(a)} for (a,b),e in zip(x,eps)]
    responses=[ResponseItem(name='Y',feature='望小')]
    if limit is not None:responses.append(ResponseItem(name='G',kind='约束',feature='望小',robust_limit=str(limit)))
    p=ProjectData(factors=[FactorItem(name=n,param1='-1',param2='1') for n in ('A','B')],responses=responses,
                  design_method='响应曲面设计',doe_matrix=rows)
    context,errors=oe.build_models(p)
    assert not errors,errors
    return p,context


class WorkbenchAnalysisTests(unittest.TestCase):
    def test_descriptive_finite_and_sample_sd(self):
        p=ProjectData(responses=[ResponseItem(name='Y')],doe_matrix=[{'Y':v} for v in [1,2,3,'',float('nan'),float('inf')]])
        r=wa.descriptive(p)[0]
        self.assertEqual(r['有效数'],3);self.assertEqual(r['缺失或无效'],3)
        self.assertEqual(r['样本标准差'],1);self.assertEqual(r['中位数'],2)
        self.assertEqual(oe.level_stats(ProjectData())['rows'],[])

    def test_ols_inference_against_independent_matrix(self):
        p,c=make_context();r=wa.diagnostics(c,'Y');m=c['models']['Y']
        x=np.array([[row['A'],row['B']] for row in p.doe_matrix]);y=np.array([row['Y'] for row in p.doe_matrix])
        z=(x-m['center'])/m['half']
        a=np.column_stack([np.ones(len(x)),z[:,0],z[:,0]**2,z[:,1],z[:,1]**2,z[:,0]*z[:,1]])
        beta=np.linalg.solve(a.T@a,a.T@y);res=y-a@beta;mse=res@res/(len(x)-6)
        se=np.sqrt(np.diag(np.linalg.inv(a.T@a))*mse)
        np.testing.assert_allclose([row['估计值'] for row in r['coefficients']],beta)
        np.testing.assert_allclose([row['标准误'] for row in r['coefficients']],se)
        for i,row in enumerate(r['coefficients']):
            self.assertAlmostEqual(row['未校正 p'],2*stats.t.sf(abs(beta[i]/se[i]),29))
            self.assertAlmostEqual(row['95%下限'],beta[i]-stats.t.ppf(.975,29)*se[i])
        self.assertAlmostEqual(sum(row['杠杆值'] for row in r['observations']),6)

    def test_press_matches_explicit_leave_one_out(self):
        p,c=make_context();d={row['统计量']:row['值'] for row in wa.diagnostics(c,'Y')['summary']}
        x=np.array([[row['A'],row['B']] for row in p.doe_matrix]);y=np.array([row['Y'] for row in p.doe_matrix])
        press=0
        for i in range(len(x)):
            m=oe._fit_model(np.delete(x,i,0),np.delete(y,i),full_quadratic=True)
            press+=(y[i]-oe.model_predict(m,x[i]))**2
        self.assertAlmostEqual(d['LOO PRESS'],press,places=10)

    def test_exact_polynomial_has_no_fake_p_values(self):
        p,c=make_context(False);r=wa.diagnostics(c,'Y')
        self.assertTrue(all(row['未校正 p'] is None for row in r['coefficients']))
        self.assertTrue(all(row['Cook距离'] is None for row in r['observations']))

    def test_lof_uses_repeated_point_pure_error(self):
        p,c=make_context()
        for row in list(p.doe_matrix[:6]):
            p.doe_matrix.extend([{**row,'Y':row['Y']-.1},{**row,'Y':row['Y']+.1}])
        c,errors=oe.build_models(p);self.assertFalse(errors)
        measures={r['统计量']:r['值'] for r in wa.diagnostics(c,'Y')['summary']}
        self.assertEqual(measures['纯误差自由度'],12)
        self.assertAlmostEqual(measures['纯误差平方和'],.12)
        expected=((measures['残差平方和']-.12)/measures['失拟自由度'])/(.12/12)
        self.assertAlmostEqual(measures['失拟 F'],expected)

    def test_data_only_ui_with_missing_values_and_one_level(self):
        p,c=make_context()
        for row in p.doe_matrix:row['A']=0
        p.doe_matrix[0]['Y']='';p.doe_matrix[1]['Y']=float('nan')
        page=OptimizationPage(p)
        with patch.object(oe,'build_models',side_effect=AssertionError('data-only view fitted model')):
            page._refresh_data_analysis()
        self.assertEqual(page.data_stats_table.rowCount(),1)
        self.assertTrue(all(np.isnan(r['F']) for r in oe.level_stats(p)['rows'] if r['factor']=='A'))
        page.close()

    def test_surrogate_holdout_never_claims_ols_inference(self):
        p,c=make_context();c['models']['Y']={'predict_fn':lambda x:0,'validation':{'actual':[1,2,3],'predicted':[1,2.5,3]}}
        r=wa.diagnostics(c,'Y')
        self.assertFalse(r['coefficients']);self.assertNotIn('杠杆值',r['observations'][0])
        self.assertIn('留出',r['source'])

    def test_local_search_feasibility_and_monotone_acceptance(self):
        p,c=make_context(False,limit=.5)
        for method in ('最陡上升','AOFAT','EVOP'):
            r=oe.local_search(c,{'A':0,'B':0},6,2,.8,method,16,.2)
            self.assertTrue(all(v['feasible'] for v in r['path']))
            self.assertTrue(all(a['score']>=b['score'] for a,b in zip(r['path'],r['path'][1:])))
            self.assertTrue(all(-1<=v<=1 for row in r['trials'] for v in row['x'].values()))

    def test_stability_search_does_not_return_infeasible_point(self):
        p,c=make_context(False,limit=-100)
        r=oe.parameter_design(c,{},6,2)
        self.assertIsNone(r['stable'])
        self.assertFalse(oe.local_search(c,{'A':0,'B':0},6,2)['best']['feasible'])

    def test_tolerance_reconciles_with_optimizer_at_same_point(self):
        p,c=make_context();point={'A':.4,'B':-.2}
        text,detail=oe.tolerance_contribution(c,6,point)
        mu,sigma=oe.robust_moments(c['models']['Y'],point,c,6)
        self.assertAlmostEqual(sum(r['variance'] for r in detail['Y']),sigma**2)
        self.assertAlmostEqual(sum(r['share'] for r in detail['Y']),100)

    def test_point_margin_uses_selected_k_and_bounds(self):
        p,c=make_context(False,limit=.5)
        a=wa.point_analysis(c,{'A':0,'B':0},6,2)
        b=wa.point_analysis(c,{'A':0,'B':0},6,6)
        self.assertTrue(a['feasible']);self.assertFalse(b['feasible'])
        self.assertAlmostEqual(a['responses'][1]['约束余量'],.5-2/6,places=9)
        with self.assertRaises(ValueError):wa.point_analysis(c,{'A':2,'B':0},6,2)

    def test_statistics_ui_export_and_invalidation(self):
        p,c=make_context();page=OptimizationPage(p)
        with patch.object(oe,'robust_optimize',side_effect=AssertionError('statistics ran optimizer')),patch.object(QMessageBox,'warning') as warning:
            page.run_statistics();warning.assert_not_called()
        panel=page.diagnostics_panel
        self.assertEqual(panel.tables['observations'].rowCount(),35)
        self.assertEqual(panel.tables['coefficients'].rowCount(),6)
        panel.evaluate_point();self.assertEqual(panel.point_table.rowCount(),1)
        with tempfile.TemporaryDirectory() as d:
            json_path=str(Path(d)/'statistics.json');csv_path=str(Path(d)/'statistics.csv')
            with patch.object(QFileDialog,'getSaveFileName',return_value=(json_path,'')):panel.export_json()
            self.assertEqual(len(json.loads(Path(json_path).read_text(encoding='utf-8'))['responses']['Y']['observations']),35)
            panel.tabs.setCurrentWidget(panel.tables['observations'])
            with patch.object(QFileDialog,'getSaveFileName',return_value=(csv_path,'')):panel.export_csv()
            self.assertEqual(len(Path(csv_path).read_text(encoding='utf-8-sig').splitlines()),36)
        p.doe_matrix[0]['Y']+=1;page._check_inputs()
        self.assertIsNone(panel.context);self.assertEqual(panel.tables['observations'].rowCount(),0)
        page.close()


if __name__=='__main__':unittest.main()
