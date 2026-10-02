"""Single current-result report; no operation archive UI."""
from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QLineEdit, QFileDialog
import report_engine as reports
from notifications import Notice

class AnalysisPage(QWidget):
    def __init__(self, project_data):
        super().__init__()
        self.project_data = project_data
        self.report = None
        self.report_html = ""
        layout = QVBoxLayout(self)
        title = QLabel("设计结果报告")
        title.setStyleSheet("font-size:22px;font-weight:600;color:#17385e;padding:8px;")
        title.hide()
        layout.setContentsMargins(16,12,16,12)
        self.status_label = QLabel("完成一次分析或优化后，在这里预览并导出当前结果。")
        self.status_label.setProperty("role", "muted")
        layout.addWidget(self.status_label)
        bar = QHBoxLayout()
        self.fields = {}
        for key, label in [("project_name", "项目名称"), ("author", "编制人")]:
            field = QLineEdit(project_data.report_metadata.get(key, ""))
            field.setPlaceholderText(label + "（选填）")
            self.fields[key] = field
            bar.addWidget(field)
        for label, slot in [("更新预览", self.run_analysis), ("导出 PDF", self.export_pdf), ("导出 HTML", self.export_html)]:
            button = QPushButton(label)
            button.clicked.connect(slot)
            if label == "导出 PDF":
                from ui_theme import primary
                primary(button)
            bar.addWidget(button)
        layout.addLayout(bar)
        self.preview = QTextBrowser()
        layout.addWidget(self.preview, 1)

    def showEvent(self, event):
        super().showEvent(event)
        self.run_analysis()

    def run_analysis(self, *_args):
        self.report = None
        self.report_html = ""
        try:
            self.project_data.report_metadata = {k:w.text().strip() for k,w in self.fields.items()}
            data = reports.bundle(self.project_data)
            if not any(data["results"].values()):
                self.preview.setPlainText("暂无当前结果。请先完成筛选分析、田口分析或模型优化。输入改变后需要重新计算。")
                self.status_label.setText("等待当前输入的分析结果")
                return
            self.report = reports.build_current_report(data)
            self.report_html = reports.render_html(self.report)
            self.preview.setHtml(self.report_html)
            self.status_label.setText("当前计算结果 · 再次计算后替换本报告 · 不保存操作履历")
        except Exception as exc:
            self.preview.setPlainText("报告生成失败：" + str(exc))
            Notice.warning(self, "报告生成失败", str(exc))

    def export_pdf(self):
        self._export("pdf")

    def export_html(self):
        self._export("html")

    def _export(self, kind):
        self.run_analysis()
        if not self.report_html:
            Notice.warning(self, "暂无报告", "请先完成当前输入的分析或优化。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出当前结果", "设计结果报告." + kind, kind.upper() + " (*." + kind + ")")
        if not path:
            return
        try:
            if kind == "pdf":
                reports.export_pdf(self.report_html, path)
            else:
                Path(path).write_text(self.report_html, encoding="utf-8")
            Notice.information(self, "报告已导出", path)
        except Exception as exc:
            Notice.warning(self, "导出失败", str(exc))
