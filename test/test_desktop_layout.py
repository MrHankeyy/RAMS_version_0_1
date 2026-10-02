"""Navigation and resizing contracts; computation regression lives in engine tests."""
import os
import sys
import unittest
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from main_window import MainWindow

APP = QApplication.instance() or QApplication([])


class DesktopLayoutTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        APP.processEvents()

    def tearDown(self):
        self.window.close()

    def test_workflow_heading_and_small_window(self):
        self.window.resize(1100, 760)
        for index in range(5):
            self.window.btns_switch[index].click()
            APP.processEvents()
            self.assertEqual(self.window.stacked_widget.currentIndex(), index)
            self.assertEqual(self.window.page_title.text(), self.window.PAGE_INFO[index][0])
            self.assertEqual(self.window.width(), 1100)
            self.assertEqual(sum(b.property('active') for b in self.window.btns_switch), 1)

    def test_result_navigation_tracks_visible_method_pages(self):
        page = self.window.page_OPT
        for method, wanted, unwanted in [('筛选设计', 9, 10), ('田口设计', 10, 9), ('响应曲面设计', 1, 9)]:
            page.project_data.design_method = method
            page._update_method_actions()
            navigation = page.results_workspace.navigation
            indices = [navigation.item(i).data(Qt.ItemDataRole.UserRole) for i in range(navigation.count())]
            self.assertIn(wanted, indices)
            self.assertNotIn(unwanted, indices)
            navigation.setCurrentRow(indices.index(wanted))
            self.assertEqual(page.tabs.currentIndex(), wanted)
            page.tabs.setCurrentIndex(0)
            self.assertEqual(navigation.currentItem().data(Qt.ItemDataRole.UserRole), 0)

    def test_factor_tabs_preserve_rows_and_show_counts(self):
        page = self.window.page_MOD
        page.add_design_factor_row()
        page.add_environment_factor_row()
        page.environment_factor_table.cellWidget(0, 0).setText('温度')
        page.factor_tabs.setCurrentIndex(1)
        self.assertEqual(page.factor_tabs.tabText(1), '环境因子  1')
        page.factor_tabs.setCurrentIndex(0)
        self.assertEqual(page.environment_factor_table.cellWidget(0, 0).text(), '温度')

    def test_message_collapse_preserves_content(self):
        self.window.fold_messages.click()
        self.assertTrue(self.window.message_view.isHidden())
        self.window._post_message('成功', '保存', '已完成')
        self.window.fold_messages.click()
        self.assertFalse(self.window.message_view.isHidden())
        self.assertIn('已完成', self.window.message_view.toPlainText())


if __name__ == '__main__':
    unittest.main()
