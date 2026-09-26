"""Read-only, response-specific result panels for classical DOE methods."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class DesignResultsPanel(QWidget):
    def __init__(self, family, parent=None):
        super().__init__(parent)
        self.family = family
        self.result = {}
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("响应："))
        self.response_combo = QComboBox()
        row.addWidget(self.response_combo)
        factor_label = QLabel("绘图因子：")
        factor_label.setVisible(family == "taguchi")
        row.addWidget(factor_label)
        self.factor_combo = QComboBox()
        row.addWidget(self.factor_combo)
        self.factor_combo.setVisible(family == "taguchi")
        layout.addLayout(row)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(125)
        layout.addWidget(self.summary)
        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.runs_table = QTableWidget()
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.setVisible(family == "taguchi")
        layout.addWidget(self.runs_table)
        self.fig = Figure(figsize=(6, 2.5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        self.response_combo.currentTextChanged.connect(self.render)
        self.factor_combo.currentTextChanged.connect(self.render)

    @staticmethod
    def fill(table, headers, rows):
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                text = "—" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)
                table.setItem(i, j, QTableWidgetItem(text))

    def set_result(self, result):
        self.result = result or {}
        names = list(self.result.get("response_analysis", {})) + list(self.result.get("errors", {}))
        old = self.response_combo.currentText()
        self.response_combo.blockSignals(True)
        self.response_combo.clear()
        self.response_combo.addItems(list(dict.fromkeys(names)))
        if old in names:
            self.response_combo.setCurrentText(old)
        self.response_combo.blockSignals(False)
        self.render()

    def render(self, *_args):
        name = self.response_combo.currentText()
        item = self.result.get("response_analysis", {}).get(name, {})
        self.fig.clear()
        self.table.setRowCount(0)
        self.runs_table.setRowCount(0)
        lines = [self.result.get("analysis_status", "等待响应数据与后端分析。")]
        error = self.result.get("errors", {}).get(name)
        if error:
            lines.append(f"{name}：{error}")
        elif self.family == "screening" and item:
            self.fill(self.table, ["排序", "因子", "高−低效应", "低水平均值", "高水平均值", "平方效应份额 %", "标准误", "未校正 p"],
                      [[e["rank"], e["name"], e["effect"], e["low_mean"], e["high_mean"], e["share"], e["se"], e["p"]] for e in item["effects"]])
            lines += [item["note"], f"纯误差自由度：{item['pure_error_df']}；中心−角点均值差：{item['curvature']}",
                      self.result.get("diagnostic_summary", "")]
            aliases = self.result.get("aliases", [])
            lines += ["主效应与二因子交互混杂：\n" + "\n".join(aliases)] if aliases else ["未发现主效应与二因子交互混杂；高阶混杂仍取决于设计分辨率。"]
            ax = self.fig.add_subplot(111)
            effects = item["effects"][:len(self.result.get("main_effects", [])) or 5]
            ax.bar([e["name"] for e in effects], [e["effect"] for e in effects])
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set_title(f"{name}：重点主效应（保留方向）")
            ax.set_ylabel("高水平均值 − 低水平均值")
        elif item:
            names = list(dict.fromkeys(r["factor"] for r in item["levels"]))
            if [self.factor_combo.itemText(i) for i in range(self.factor_combo.count())] != names:
                self.factor_combo.blockSignals(True)
                self.factor_combo.clear()
                self.factor_combo.addItems(names)
                self.factor_combo.blockSignals(False)
            self.fill(self.table, ["因子", "实际水平", "均值", "平均组内标准差", item["metric_label"], "内表组数", "候选水平"],
                      [[r["factor"], r["level_value"], r["mean"], r["std"], r["metric"], r["n"], "是" if r["recommended"] else ""] for r in item["levels"]])
            self.fill(self.runs_table, ["内表组", "控制因子设置", "观测数", "均值", "标准差", item["metric_label"]],
                      [[r["inner"], str(r["settings"]), r["n"], r["mean"], r["std"], r["metric"]] for r in item["runs"]])
            lines += [f"{name}（{item['role']}）：{item['metric_label']}", item["note"],
                      f"该响应候选水平：{item['best_levels'] or '无（约束仅描述）'}", self.result.get("recommendation", "")]
            rows = [r for r in item["levels"] if r["factor"] == self.factor_combo.currentText()]
            for i, (key, label) in enumerate((("mean", "均值"), ("metric", item["metric_label"]))):
                ax = self.fig.add_subplot(1, 2, i+1)
                ax.plot([r["level_value"] for r in rows], [r[key] if math.isfinite(r[key]) else float("nan") for r in rows], "o-")
                if any(not math.isfinite(r[key]) for r in rows):
                    ax.text(.02, .98, "零损失 S/N=+∞，详见表格", transform=ax.transAxes, va="top")
                ax.set_title(label)
                ax.set_xlabel(self.factor_combo.currentText())
        self.summary.setPlainText("\n".join(lines))
        self.canvas.draw()
