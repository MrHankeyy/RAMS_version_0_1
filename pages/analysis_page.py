from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt
from models import ProjectData
from pages.design_results import DesignResultsPanel

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

        self.calc_btn = QPushButton("刷新分析报告")
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

        self.screening_panel = DesignResultsPanel("screening")
        self.taguchi_panel = DesignResultsPanel("taguchi")
        self.tabs.addTab(self.screening_panel, "筛选响应分析")
        self.tabs.addTab(self.taguchi_panel, "田口水平分析")
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
        self.project_data.ensure_results_current()
        method = self.project_data.design_method
        screening, taguchi = "筛选" in method, "田口" in method
        for i in range(self.tabs.count()):
            self.tabs.setTabVisible(i, i == (5 if screening else 6 if taguchi else 3) or (i == 4 and "代理模型" in method))
        self.screening_panel.set_result(self.project_data.screening_result)
        self.taguchi_panel.set_result(self.project_data.taguchi_result)
        self.tabs.setCurrentIndex(5 if screening else 6 if taguchi else 3)
        fit = (self.project_data.rsm_result or {}).get("fit", {})
        self.diag_text.setText("\n".join(f"{name}：R²={item['r2']}，RMSE={item['rmse']}，残差自由度={item['df_residual']}" for name, item in fit.items()) if "响应曲面" in method and fit else "请先在优化设计模块执行当前方案分析；本页仅展示已计算结果。")

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

    def showEvent(self, event):
        super().showEvent(event)
        self.run_analysis()
