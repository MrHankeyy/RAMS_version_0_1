from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt
from models import ProjectData

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class AnalysisPage(QWidget):
    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)

        top_layout = QHBoxLayout()
        self.info_label = QLabel("主效应筛选与稳健设计分析")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.calc_btn = QPushButton("运行分析")
        self.calc_btn.setStyleSheet("background-color: #17a2b8; color: white;")
        self.calc_btn.setMinimumWidth(150)
        self.calc_btn.clicked.connect(self.run_analysis)

        top_layout.addWidget(self.info_label)
        top_layout.addStretch()
        top_layout.addWidget(self.calc_btn)
        main_layout.addLayout(top_layout)

        self.tabs = QTabWidget()

        self.tab_rank = QWidget(); self.setup_rank_tab(); self.tabs.addTab(self.tab_rank, "因子影响排序")
        self.tab_plot = QWidget(); self.setup_plot_tab(); self.tabs.addTab(self.tab_plot, "主效应/因子图")
        self.tab_stability = QWidget(); self.setup_stability_tab(); self.tabs.addTab(self.tab_stability, "稳定性影响")
        self.tab_diag = QWidget(); self.setup_diag_tab(); self.tabs.addTab(self.tab_diag, "设计诊断")
        self.tab_surrogate = QWidget(); self.setup_surrogate_tab(); self.tabs.addTab(self.tab_surrogate, "代理模型预测 vs 实测")

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def setup_rank_tab(self):
        layout = QVBoxLayout(self.tab_rank)
        self.rank_table = QTableWidget(0, 3)
        self.rank_table.setHorizontalHeaderLabels(["排序", "因子", "影响度 (%)"])
        self.rank_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.rank_table)

    def setup_plot_tab(self):
        layout = QVBoxLayout(self.tab_plot)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def setup_stability_tab(self):
        layout = QVBoxLayout(self.tab_stability)
        self.stability_table = QTableWidget(0, 3)
        self.stability_table.setHorizontalHeaderLabels(["因子", "稳定性影响", "说明"])
        self.stability_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.stability_table)

    def setup_diag_tab(self):
        layout = QVBoxLayout(self.tab_diag)
        self.diag_text = QLabel()
        self.diag_text.setWordWrap(True)
        self.diag_text.setStyleSheet("background-color: #f4f6f8; padding: 12px; border: 1px solid #d9dee5;")
        layout.addWidget(self.diag_text)

    def setup_surrogate_tab(self):
        layout = QVBoxLayout(self.tab_surrogate)
        self.surrogate_fig = Figure(figsize=(5, 4), dpi=100)
        self.surrogate_canvas = FigureCanvas(self.surrogate_fig)
        layout.addWidget(self.surrogate_canvas)

    def run_analysis(self):
        result = self.project_data.screening_result
        if not result or not result.get("factor_rankings"):
            result = {
                "factor_rankings": [{"rank": idx + 1, "name": f.name, "share": 10 + (len([x for x in self.project_data.factors if x.name]) - idx) * 2} for idx, f in enumerate([f for f in self.project_data.factors if f.name][:5])],
                "main_effects": [],
                "stability_impact": [],
                "diagnostic_summary": "系统未检测到有效筛选结果，请先在方案配置中生成筛选方案。",
                "conclusion": "未生成筛选结论。",
                "plot_points": [],
            }

        self.rank_table.setRowCount(len(result["factor_rankings"]))
        for idx, item in enumerate(result["factor_rankings"]):
            self.rank_table.setItem(idx, 0, QTableWidgetItem(str(item.get("rank", idx + 1))))
            self.rank_table.setItem(idx, 1, QTableWidgetItem(str(item.get("name", ""))))
            self.rank_table.setItem(idx, 2, QTableWidgetItem(f"{item.get('share', 0):.2f}%"))

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        plot_points = result.get("plot_points", [])
        if plot_points:
            names = [point["name"] for point in plot_points]
            values = [float(point["value"]) for point in plot_points]
            ax.bar(names, values, color="#17a2b8")
            ax.set_title("主效应图 / 因子图")
            ax.set_ylabel("影响强度")
            ax.tick_params(axis='x', rotation=30)
        else:
            ax.text(0.5, 0.5, "待筛选结果", ha='center', va='center')
        self.canvas.draw()

        stability = result.get("stability_impact") or []
        self.stability_table.setRowCount(len(stability))
        for idx, item in enumerate(stability):
            self.stability_table.setItem(idx, 0, QTableWidgetItem(str(item.get("name", ""))))
            self.stability_table.setItem(idx, 1, QTableWidgetItem(f"{item.get('impact', 0):.2f}%"))
            self.stability_table.setItem(idx, 2, QTableWidgetItem(f"Rank {item.get('rank', idx + 1)}"))

        self.diag_text.setText(
            f"【筛选结论】\n{result.get('conclusion', '无筛选结论。')}\n\n"
            f"【诊断摘要】\n{result.get('diagnostic_summary', '暂无诊断信息。')}"
        )

        # 代理模型预测 vs 实测
        self.surrogate_fig.clear()
        ax = self.surrogate_fig.add_subplot(111)
        per_resp = (self.project_data.surrogate_result or {}).get("responses") or {}
        if per_resp:
            for resp_name, item in per_resp.items():
                actual = item.get("actual") or []
                predicted = item.get("predicted") or []
                if actual and len(actual) == len(predicted):
                    ax.scatter(actual, predicted, label=f"{resp_name} (R²={item.get('r2_test', '')})")
            if ax.has_data():
                lo_v, hi_v = ax.get_xlim()
                ax.plot([lo_v, hi_v], [lo_v, hi_v], "--", color="gray", label="理想线 y=x")
                ax.set_xlabel("实测值")
                ax.set_ylabel("模型预测值")
                ax.set_title("代理模型测试集：预测 vs 实测")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "暂无代理模型训练数据", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "暂无代理模型训练数据", ha="center", va="center")
        self.surrogate_canvas.draw()

        QMessageBox.information(self, "分析完成", "已更新筛选与主效应分析结果。")
