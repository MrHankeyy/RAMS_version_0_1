"""Synthetic-data screenshots for desktop layout QA (no user files)."""
import os
import sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PyQt6.QtWidgets import QApplication, QDoubleSpinBox
from PyQt6.QtGui import QFont, QFontDatabase
from main_window import MainWindow
from test_workbench_analysis import make_context
from test_classical_doe import taguchi

app = QApplication.instance() or QApplication([])
fid = QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
font = QFont(QFontDatabase.applicationFontFamilies(fid)[0], 10)
w = MainWindow()
app.setFont(font)
out = Path('test/workbench_artifacts/desktop')
out.mkdir(parents=True, exist_ok=True)
w.show()
for _ in range(2): w.page_MOD.add_response_row()
for _ in range(4): w.page_MOD.add_design_factor_row()
w.page_MOD.add_environment_factor_row()

def capture(name):
    for spin in w.findChildren(QDoubleSpinBox):
        spin.setFont(font)
        spin.lineEdit().setFont(font)
    for width, height in [(1440, 940), (1100, 760)]:
        w.resize(width, height)
        app.processEvents()
        w.grab().save(str(out / f'{name}-{width}.png'))

capture('01-model')
w.on_switch_module('方案配置', 1)
p, _ = make_context()
w.project_data.__dict__.update(p.__dict__)
w.page_CFG.update_info_display()
capture('02-config')
for i in (1, 2, 3):
    w.page_CFG.method_combo.setCurrentIndex(i)
    capture(f'02-config-method{i}')
w.on_switch_module('数据管理', 2)
w.page_DM.load_doe_matrix()
capture('03-data')
w.on_switch_module('设计优化', 3)
w.page_OPT.pop_spin.setValue(20)
w.page_OPT.gen_spin.setValue(30)
w.page_OPT._execute_workbench()
w.page_OPT.tabs.setCurrentIndex(1)
capture('04-optimize')
w.page_OPT.tabs.setCurrentIndex(11)
capture('04-diagnostics')
for index in (0, 2, 3, 4, 5, 6, 7, 8):
    w.page_OPT.tabs.setCurrentIndex(index)
    capture(f'04-result-{index}')
from pages.surrogate_dialogs import SurrogateDialog
for key in ('Kriging', 'SVR', 'ANN'):
    dialog = SurrogateDialog(key, w.project_data.factors, w)
    dialog.show()
    app.processEvents()
    dialog.grab().save(str(out / f'dialog-{key}.png'))
    dialog.close()
w.on_switch_module('分析报告', 4)
capture('05-report')
t = taguchi()
w.project_data.__dict__.update(t.__dict__)
w.on_switch_module('设计优化', 3)
w.page_OPT.run_optimization()
w.page_OPT.tabs.setCurrentIndex(10)
capture('04-taguchi')
w.close()
print('Desktop layout previews saved:', out)
