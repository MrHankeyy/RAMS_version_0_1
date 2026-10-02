import os
import sys
import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt6.QtWidgets import QApplication,QMessageBox,QFileDialog
from PyQt6.QtCore import Qt
from models import ProjectData,FactorItem,ResponseItem
from pages.optimization_page import OptimizationPage
from pages.analysis_page import AnalysisPage
import report_engine as re
from test_workbench_analysis import make_context

APP=QApplication.instance() or QApplication([])


class ReportTests(unittest.TestCase):
    def test_operations_emit_messages_without_history(self):
        from notifications import bus
        p,_=make_context();messages=[]
        def receive(*args):messages.append(args)
        bus.posted.connect(receive)
        try:
            p.record_operation('响应保存')
            p.record_operation('优化', '失败', message='不可行')
            self.assertFalse(p.report_history)
            self.assertEqual(messages[-1], ('错误','优化','失败：不可行'))
        finally:bus.posted.disconnect(receive)

    def test_report_does_not_refit_and_has_current_sections(self):
        p,_=make_context();page=OptimizationPage(p)
        page.run_statistics()
        self.assertTrue(p.workbench_result)
        with patch('optimizer_engine.build_models',side_effect=AssertionError('report refitted')):
            report=re.build_current_report(re.bundle(p))
        self.assertEqual([s['id'] for s in report['sections']],['summary','purpose','materials','methods','results','settings','risks'])
        self.assertTrue(any(b['type']=='chart' for s in report['sections'] for b in s['blocks']))
        self.assertFalse(p.report_history)
        page.close()

    def test_untrained_and_empty_projects(self):
        for p in [ProjectData(),ProjectData(design_method='响应曲面设计',rsm_result={'fit':None})]:
            report=re.build_report(re.bundle(p));content=re.render_html(report)
            self.assertIn('待完善',content);self.assertIn('尚无分析结果',content)

    def test_html_escapes_user_text(self):
        p=ProjectData(report_metadata={'project_name':'<script>alert(1)</script>','purpose':'<img src="file:///private">'})
        content=re.render_html(re.build_report(re.bundle(p)))
        self.assertNotIn('<script>',content);self.assertIn('&lt;script&gt;',content)

    def test_serialization_preserves_arbitrary_data_column_names(self):
        data={'signature':1.2,'predict_fn':3.4,'model':{'predict_fn':lambda x:0}}
        clean=re.serializable(data)
        self.assertEqual(clean['signature'],1.2);self.assertEqual(clean['predict_fn'],3.4)
        self.assertEqual(clean['model'],{})

    def test_stale_results_removed(self):
        p,_=make_context();p.ensure_results_current();p.workbench_result={'optimization':{'best':{'x':{'A':.7}}}}
        p.record_operation('鲁棒优化')
        p.doe_matrix[0]['Y']+=1
        data=re.bundle(p)
        self.assertFalse(data['results']['workbench']);self.assertFalse(data['history'])

    def test_qt_single_result_and_minimal_metadata(self):
        p,_=make_context();opt=OptimizationPage(p);opt.run_statistics()
        page=AnalysisPage(p);page.run_analysis()
        self.assertIsNotNone(page.report)
        self.assertEqual(set(page.fields), {'project_name','author'})
        self.assertFalse(hasattr(page, 'history_table'))
        self.assertNotIn('试验记录与版本追溯', page.report_html)
        p.doe_matrix[0]['Y']+=1
        page.run_analysis()
        self.assertIsNone(page.report)
        self.assertEqual(page.report_html, '')
        page.close();opt.close()

    def test_pdf_export_and_gui_html(self):
        p,_=make_context();p.report_metadata={'project_name':'报告导出测试'}
        opt=OptimizationPage(p);opt.run_statistics()
        page=AnalysisPage(p);page.run_analysis()
        with tempfile.TemporaryDirectory() as directory:
            pdf=Path(directory)/'report.pdf';html=Path(directory)/'report.html'
            with patch.object(QFileDialog,'getSaveFileName',return_value=(str(pdf),'')),patch.object(QMessageBox,'warning') as warning:
                page.export_pdf();warning.assert_not_called()
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))
            with patch.object(QFileDialog,'getSaveFileName',return_value=(str(html),'')):page.export_html()
            self.assertIn('报告导出测试',html.read_text(encoding='utf-8'))
        page.close()

    def test_optimization_failure_is_message_and_invalidates_old_result(self):
        from notifications import Notice
        p,_=make_context();page=OptimizationPage(p)
        p.workbench_result={'old':True}
        with patch.object(page,'_prepare_context',side_effect=ValueError('模型不可识别')),patch.object(Notice,'warning') as warning:
            page.run_workbench()
            self.assertIn('模型不可识别', warning.call_args.args[2])
        self.assertFalse(p.workbench_result)
        self.assertFalse(p.report_history)
        page.close()


if __name__=='__main__':unittest.main()
