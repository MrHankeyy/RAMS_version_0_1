"""Create explicitly labelled synthetic reports for export and visual QA."""
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_workbench_analysis import APP,make_context
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QFont,QFontDatabase
from pages.optimization_page import OptimizationPage
from pages.analysis_page import AnalysisPage
import report_engine as reports
import workbench_analysis as wa

font=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
if font>=0:APP.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0],10))
p,c=make_context(noise=True,limit=.5)
p.report_metadata={'project_name':'演示数据：稳定性设计报告（非用户工程结论）',
    'author':'软件验证示例','reviewer':'待复核','purpose':'用已知二次函数和固定种子噪声验证报告的数据追溯、约束判断和导出。',
    'materials':'合成数值试验，35 个固定随机种子生成的验证点；未使用真实样机或用户试验数据。',
    'conclusion':'本文件仅为软件功能演示，不能作为产品验收结论。',
    'confirmation':'实际工程使用时，应对候选点补充独立确认试验并复核输入扰动假设。'}
p.record_operation('业务模型保存');p.record_operation('试验响应保存',message='固定种子合成响应，用于软件验证')
page=OptimizationPage(p);page.pop_spin.setValue(20);page.gen_spin.setValue(30);page.k_constraint_spin.setValue(2)
with patch.object(QMessageBox,'information'),patch.object(QMessageBox,'warning') as warning:
    page._execute_workbench();assert not warning.called
evaluation=wa.point_analysis(page._oe_context,{'A':1,'B':1},6,2)
assert not evaluation['feasible']
data=reports.bundle(p)
out=Path('output/pdf');out.mkdir(parents=True,exist_ok=True)
report=reports.build_current_report(data)
document=reports.render_html(report)
reports.export_pdf(document,out/'current-result-demo.pdf')
ui=AnalysisPage(p);ui.resize(1450,960);ui.show();ui.run_analysis();APP.processEvents()
artifacts=Path('test/report_artifacts');artifacts.mkdir(exist_ok=True)
ui.grab().save(str(artifacts/'report-page.png'))

print('Synthetic demo reports and previews generated.')
ui.close();page.close()
