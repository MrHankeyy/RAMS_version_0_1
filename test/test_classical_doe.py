"""Analytical checks and Qt integration for the three classical DOE workflows."""
import os
import sys
import math
import unittest
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import doe_engine as de
import design_analysis as da
import optimizer_engine as oe
from models import ProjectData, FactorItem, ResponseItem
from PyQt6.QtWidgets import QApplication, QMessageBox
from pages.optimization_page import OptimizationPage
from pages.configuration_page import ConfigurationPage
from pages.analysis_page import AnalysisPage

APP = QApplication.instance() or QApplication([])


def factors(k):
    return [FactorItem(name=f'X{i}', param1='-1', param2='1') for i in range(k)]


def taguchi(feature='望小', outer=True, repeats=1):
    fs = factors(2)
    noise = [FactorItem(name='N', source='环境', param1='-1', param2='1')] if outer else []
    rows, meta = de.build_taguchi(de.simple_factors(fs), de.simple_factors(noise), 'L4', 'L4', 2,
                                  seed=4, replicates=repeats)
    for row, group in zip(rows, meta['groups']):
        row.update({'内表组号':group['inner']+1, '外表组号':group['outer']+1})
        row['Y'] = 10 + 2*row['X0'] + row['X1'] + .1*row.get('N', 0)
    return ProjectData(factors=fs+noise, responses=[ResponseItem(name='Y', feature=feature, lower='9', upper='11')],
        design_method='田口稳健设计（内表×外表）', design_params={'meta':meta}, doe_matrix=rows)


class ClassicalDOE(unittest.TestCase):
    def test_screening_all_columns(self):
        for kind, ks in [('full', range(1,8)), ('half',range(3,11)), ('quarter',range(5,11)), ('pb',range(1,20))]:
            for k in ks:
                with self.subTest(kind=kind,k=k):
                    x = np.array(de.screening_coded_runs(k,kind)[0])
                    np.testing.assert_equal(x.sum(axis=0),np.zeros(k))
                    np.testing.assert_equal(x.T@x,len(x)*np.eye(k))
        with self.assertRaises(ValueError): de.screening_coded_runs(4,'quarter')

    def test_screening_real_effects_missing_and_scale(self):
        fs=factors(3)
        rows,meta=de.build_screening(de.simple_factors(fs),'full',seed=1,centers=2,replicates=2)
        for row in rows:
            row['Y']=5+3*row['X0']-row['X1']+.5*row['X2']+2*row['X0']*row['X1']
            row['Z']=1000*row['Y']
        p=ProjectData(factors=fs,responses=[ResponseItem(name='Y'),ResponseItem(name='Z')],doe_matrix=rows,design_params={'meta':meta,'top_n':1})
        r=da.screening(p)
        self.assertEqual(len(r['main_effects']),1)
        self.assertEqual([e['effect'] for e in r['response_analysis']['Y']['effects']],[6,-2,1])
        self.assertEqual(r['response_analysis']['Y']['pure_error_df'],11)
        self.assertAlmostEqual(r['response_analysis']['Y']['effects'][0]['share'],r['response_analysis']['Z']['effects'][0]['share'])
        rows[0]['Y']=''
        self.assertNotIn('Y',da.screening(p)['response_analysis'])

    def test_oa_balance(self):
        for label,k in [('L4',3),('L8',7),('L12',11),('L16',15),('L9',4),('L27',13)]:
            _,levels,n,rows=de.taguchi_arrays(label,k)
            for i,j in itertools.combinations(range(k),2):
                self.assertEqual(set(Counter((r[i],r[j]) for r in rows).values()),{n//levels**2})

    def test_taguchi_mixed_levels_repeats(self):
        rows,meta=de.build_taguchi(de.simple_factors(factors(2)),[('N',-1,1,None)],'L9','L4',3,replicates=2)
        self.assertEqual(len(rows),72)
        self.assertEqual(meta['noise_level_count'],2)
        self.assertEqual(set(Counter((g['inner'],g['outer']) for g in meta['groups']).values()),{2})

    def test_snr_formulas(self):
        self.assertAlmostEqual(da.snr([1,2],'望小'),-10*math.log10(2.5))
        self.assertAlmostEqual(da.snr([1,2],'望大'),-10*math.log10(.625))
        self.assertAlmostEqual(da.snr([8,12],'望目',10),-10*math.log10(4))
        self.assertEqual(da.snr([10,10],'望目',10),float('inf'))
        with self.assertRaises(ValueError): da.snr([0,1],'望大')

    def test_taguchi_directions_and_conflicts(self):
        for feature,value in [('望小',-1),('望大',1)]:
            p=taguchi(feature,False)
            self.assertEqual(da.taguchi(p)['best_levels'],{'X0':value,'X1':value})
            self.assertFalse(da.taguchi(p)['response_analysis']['Y']['has_replication'])
        p=taguchi('望目',False)
        self.assertEqual(da.taguchi(p)['response_analysis']['Y']['feature'],'望目')
        p.responses.append(ResponseItem(name='Z',feature='望大'))
        for row in p.doe_matrix: row['Z']=row['Y']
        p.responses[0].feature='望小'
        self.assertEqual(da.taguchi(p)['best_levels'],{})

    def test_taguchi_sorted_and_corrupt_groups(self):
        p=taguchi(repeats=2)
        before=da.taguchi(p)['best_levels']
        p.doe_matrix.reverse()
        self.assertEqual(before,da.taguchi(p)['best_levels'])
        for idx,row in enumerate(p.doe_matrix): row['Run_ID']=len(p.doe_matrix)-idx
        for row in p.doe_matrix:
            del row['内表组号'];del row['外表组号']
        self.assertEqual(before,da.taguchi(p)['best_levels'])
        p=taguchi()
        p.doe_matrix[0]['内表组号']=1.5
        with self.assertRaises(ValueError): da.taguchi(p)
        p=taguchi();p.doe_matrix.pop()
        with self.assertRaises(ValueError): da.taguchi(p)

    def test_rsm_polynomial_and_derivatives(self):
        fs=factors(3)
        for kind,mode in [('CCD','face'),('CCD','rotatable'),('CCD','custom'),('BBD','face')]:
            rows,meta=de.build_response_surface(de.simple_factors(fs),kind,mode,1.5,centers=3,replicates=1,seed=1)
            fun=lambda x: 4+2*x[0]+3*x[0]**2-2*x[1]**2+.5*x[2]**2+4*x[0]*x[1]
            for row in rows: row['Y']=fun([row[f.name] for f in fs])
            p=ProjectData(factors=fs,responses=[ResponseItem(name='Y')],design_method='响应曲面设计',doe_matrix=rows,design_params={'meta':meta})
            context,errors=oe.build_models(p)
            self.assertFalse(errors)
            model=context['models']['Y'];x=[.2,.3,.4]
            self.assertAlmostEqual(oe.model_predict(model,x),fun(x),places=8)
            np.testing.assert_allclose(oe.model_grad_x(model,x),[4.4,-.4,.4],atol=1e-8)
            np.testing.assert_allclose(oe.model_hess_diag_x(model,x),[6,-4,1],atol=1e-8)
            page=OptimizationPage(p);context=page._run_rsm()
            np.testing.assert_allclose(p.rsm_result['fit']['Y']['predicted'],context['models']['Y']['predicted'])
            page.close()
        p.doe_matrix[0]['Y']=''
        self.assertTrue(oe.build_models(p)[1])
        with self.assertRaises(ValueError): de.build_response_surface(de.simple_factors(fs),'BBD','face',1,centers=0,replicates=1,seed=1)

    def test_qt_generation_dispatch_report_and_invalidation(self):
        for method in ['筛选设计','田口稳健设计（内表×外表）','响应曲面设计']:
            p=ProjectData(factors=factors(3),responses=[ResponseItem(name='Y')],design_method=method)
            cfg=ConfigurationPage(p);cfg._generate_by_method(method)
            for row in p.doe_matrix: row['Y']=10+row['X0']+2*row['X1']+row['X2']**2
            page=OptimizationPage(p)
            with patch.object(QMessageBox,'information'),patch.object(QMessageBox,'critical') as error,patch.object(oe,'robust_optimize',side_effect=AssertionError('classical dispatch invoked continuous optimizer')):
                if "响应曲面" in method:
                    page.run_optimization()
                else:
                    page._run_primary_action()
                error.assert_not_called()
            report=AnalysisPage(p);report.run_analysis()
            if '筛选' in method:
                self.assertEqual(page.screening_panel.table.rowCount(),3)
                self.assertEqual(report.screening_panel.table.rowCount(),3)
            elif '田口' in method:
                self.assertEqual(page.taguchi_panel.table.rowCount(),6)
                self.assertEqual(report.taguchi_panel.table.rowCount(),6)
            else: self.assertTrue(p.rsm_result['fit'])
            p.doe_matrix[0]['Y']+=1
            page._check_inputs()
            self.assertFalse(p.screening_result or p.taguchi_result or p.rsm_result)
            self.assertIsNone(page._oe_context)
            cfg.close();page.close();report.close()

    def test_rsm_actual_workbench(self):
        fs=factors(2)
        rows,meta=de.build_response_surface(de.simple_factors(fs),'CCD','face',1,centers=3,replicates=1,seed=2)
        for row in rows: row['Y']=(row['X0']-.2)**2+2*(row['X1']+.3)**2
        p=ProjectData(factors=fs,responses=[ResponseItem(name='Y')],design_method='响应曲面设计',doe_matrix=rows,design_params={'meta':meta})
        page=OptimizationPage(p);page.pop_spin.setValue(20);page.gen_spin.setValue(10)
        with patch.object(QMessageBox,'information'), patch.object(QMessageBox,'warning') as warning:
            page.run_workbench()
            warning.assert_not_called()
        self.assertTrue(page._robust_res.get('best'))
        np.testing.assert_allclose(page._oe_context['models']['Y']['predicted'],p.rsm_result['fit']['Y']['predicted'])
        self.assertNotIn('模型拟合充分',oe.anova_and_lof(page._oe_context,p))
        page.close()

    def test_fixed_factors_and_seed(self):
        fs=[('X0',-1,1,None),('X1',-1,1,None),('FIX',0,1,.4)]
        a,meta=de.build_screening(fs,'full',seed=3,replicates=2,centers=1)
        b,_=de.build_screening(fs,'full',seed=3,replicates=2,centers=1)
        self.assertEqual(a,b);self.assertEqual(len(a),10)
        self.assertTrue(all(row['FIX']==.4 for row in a))
        rows,meta=de.build_taguchi(fs,[('N',-1,1,0)],'L4','无',2)
        self.assertEqual(meta['outer_runs'],1)
        self.assertTrue(all(row['N']==0 for row in rows))

    def test_missing_responses_never_invent_rankings(self):
        p=taguchi();p.doe_matrix[0]['Y']=''
        r=da.taguchi(p)
        self.assertFalse(r['best_levels']);self.assertFalse(r['response_analysis'])
        p=ProjectData(factors=factors(2),design_method='筛选设计')
        report=AnalysisPage(p);report.run_analysis()
        self.assertEqual(report.rank_table.rowCount(),0)
        self.assertEqual(report.screening_panel.table.rowCount(),0)
        report.close()


if __name__=='__main__': unittest.main()
