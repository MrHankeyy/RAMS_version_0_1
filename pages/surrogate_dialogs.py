"""代理模型建模设置窗口（Kriging / SVR / BP 人工神经网络）。

每个模型窗口都包含两部分：
1. 试验设计（DOE）：不确定性变量维度/上下界、样本点数量、试验设计方法；
2. 模型自身设置：相关函数与回归函数（Kriging）、核函数与 C/g 搜索（SVR）、
   网络参数设置（BPANN：手动/自动寻优）。
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QWidget
)
from PyQt6.QtCore import Qt

from doe_engine import factor_limits


MODEL_TITLES = {
    "Kriging": "Kriging 代理模型建模",
    "SVR": "SVR（支持向量回归）建模",
    "ANN": "BP 人工神经网络建模",
}


class SurrogateDialog(QDialog):
    """代理模型设置窗口。factors 为业务建模中的设计因子对象列表。"""

    def __init__(self, model_key, factors, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.factors = [f for f in factors if getattr(f, "name", "").strip()]
        self.setWindowTitle(MODEL_TITLES[model_key])
        self.resize(760, 620)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(self._build_doe_group())
        root.addWidget(self._build_model_group())

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("确定（保存设置）")
        ok_btn.setStyleSheet("background-color:#1d4ed8;color:white;font-weight:bold;")
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        root.addLayout(btn_row)

    def _build_doe_group(self):
        group = QGroupBox("试验设计（DOE）设置")
        layout = QVBoxLayout(group)

        if not self.factors:
            layout.addWidget(QLabel("当前业务模型没有可用的设计因子，请先在【业务建模】中添加设计因子。"))
            return group

        desc = QLabel(
            f"输入（不确定性）变量维度：{len(self.factors)} 个 "
            f"（来自业务建模设计因子）。下方可调整各变量抽样上下界。"
        )
        layout.addWidget(desc)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("样本点数量："))
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(6, 20000)
        self.sample_count_spin.setValue(50)
        row1.addWidget(self.sample_count_spin)
        row1.addSpacing(20)
        row1.addWidget(QLabel("试验设计方法："))
        self.doe_method_combo = QComboBox()
        self.doe_method_combo.addItems([
            "Latin Hypercube (LHS)",
            "均匀网格抽样",
            "随机抽样",
        ])
        row1.addWidget(self.doe_method_combo)
        row1.addStretch()
        layout.addLayout(row1)

        self.bounds_table = QTableWidget(len(self.factors), 3)
        self.bounds_table.setHorizontalHeaderLabels(["不确定性变量", "下限", "上限"])
        header = self.bounds_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.bounds_table.verticalHeader().setVisible(False)
        for r, f in enumerate(self.factors):
            lo, hi = factor_limits(f)
            self.bounds_table.setItem(r, 0, QTableWidgetItem(f.name))
            self.bounds_table.item(r, 0).setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.bounds_table.setItem(r, 1, QTableWidgetItem(str(lo)))
            self.bounds_table.setItem(r, 2, QTableWidgetItem(str(hi)))
        layout.addWidget(self.bounds_table)
        return group

    def _build_model_group(self):
        group = QGroupBox("模型设置")
        layout = QVBoxLayout(group)
        key = self.model_key

        if key == "Kriging":
            row = QHBoxLayout()
            row.addWidget(QLabel("相关函数："))
            self.corr_combo = QComboBox()
            self.corr_combo.addItems(["Gaussian（高斯）", "指数 Exponential", "Matern 3/2"])
            row.addWidget(self.corr_combo)
            row.addSpacing(20)
            row.addWidget(QLabel("回归函数（趋势项）："))
            self.regress_combo = QComboBox()
            self.regress_combo.addItems(["常数（Ordinary）", "一次趋势", "二次趋势"])
            row.addWidget(self.regress_combo)
            row.addStretch()
            layout.addLayout(row)
            layout.addWidget(QLabel("说明：θ 相关参数由最大似然在候选网格上自动寻优；"
                                    "回归函数影响 Kriging 漂移项的复杂度。"))

        elif key == "SVR":
            row = QHBoxLayout()
            row.addWidget(QLabel("核函数："))
            self.kernel_combo = QComboBox()
            self.kernel_combo.addItems(["RBF 核", "多项式核", "Sigmoid 核"])
            row.addWidget(self.kernel_combo)
            row.addSpacing(20)
            row.addWidget(QLabel("多项式次数："))
            self.poly_degree_spin = QSpinBox()
            self.poly_degree_spin.setRange(1, 5)
            self.poly_degree_spin.setValue(3)
            self.poly_degree_spin.setEnabled(False)
            row.addWidget(self.poly_degree_spin)
            row.addStretch()
            layout.addLayout(row)
            self.kernel_combo.currentTextChanged.connect(
                lambda t: self.poly_degree_spin.setEnabled(t == "多项式核"))

            grid = QVBoxLayout()
            c_row = QHBoxLayout()
            c_row.addWidget(QLabel("惩罚系数 C 搜索范围："))
            self.c_min = self._dspin(0.01, 100000.0, 0.1)
            self.c_max = self._dspin(0.1, 1000000.0, 100.0)
            self.c_step = self._dspin(0.01, 10000.0, 10.0)
            c_row.addWidget(self.c_min)
            c_row.addWidget(QLabel("~"))
            c_row.addWidget(self.c_max)
            c_row.addWidget(QLabel("搜索步长："))
            c_row.addWidget(self.c_step)
            c_row.addStretch()
            grid.addLayout(c_row)

            g_row = QHBoxLayout()
            g_row.addWidget(QLabel("核参数 g（γ）搜索范围："))
            self.g_min = self._dspin(0.0001, 100.0, 0.01)
            self.g_max = self._dspin(0.01, 10000.0, 10.0)
            self.g_step = self._dspin(0.0001, 100.0, 1.0)
            g_row.addWidget(self.g_min)
            g_row.addWidget(QLabel("~"))
            g_row.addWidget(self.g_max)
            g_row.addWidget(QLabel("搜索步长："))
            g_row.addWidget(self.g_step)
            g_row.addStretch()
            grid.addLayout(g_row)
            layout.addLayout(grid)
            layout.addWidget(QLabel("说明：C 与 g 按给定范围/步长网格寻优，ε-SVR 误差带取 0.05；"
                                    "未安装 scikit-learn 时自动退回核岭回归近似。"))

        else:  # ANN
            row = QHBoxLayout()
            row.addWidget(QLabel("网络参数设置方法："))
            self.net_mode_combo = QComboBox()
            self.net_mode_combo.addItems(["手动设置", "自动寻优"])
            row.addWidget(self.net_mode_combo)
            row.addStretch()
            layout.addLayout(row)

            self.manual_widget = QWidget()
            manual_row = QHBoxLayout(self.manual_widget)
            manual_row.setContentsMargins(0, 0, 0, 0)
            manual_row.addWidget(QLabel("隐含层层数："))
            self.hidden_layers_spin = QSpinBox()
            self.hidden_layers_spin.setRange(1, 5)
            self.hidden_layers_spin.setValue(1)
            manual_row.addWidget(self.hidden_layers_spin)
            manual_row.addSpacing(20)
            manual_row.addWidget(QLabel("隐含层节点数："))
            self.hidden_nodes_spin = QSpinBox()
            self.hidden_nodes_spin.setRange(1, 500)
            self.hidden_nodes_spin.setValue(8)
            manual_row.addWidget(self.hidden_nodes_spin)
            manual_row.addStretch()
            layout.addWidget(self.manual_widget)

            self.auto_widget = QWidget()
            auto_row = QHBoxLayout(self.auto_widget)
            auto_row.setContentsMargins(0, 0, 0, 0)
            auto_row.addWidget(QLabel("隐含层节点搜索区间："))
            self.node_min_spin = QSpinBox()
            self.node_min_spin.setRange(1, 200)
            self.node_min_spin.setValue(2)
            self.node_max_spin = QSpinBox()
            self.node_max_spin.setRange(2, 500)
            self.node_max_spin.setValue(30)
            auto_row.addWidget(self.node_min_spin)
            auto_row.addWidget(QLabel("~"))
            auto_row.addWidget(self.node_max_spin)
            auto_row.addSpacing(15)
            auto_row.addWidget(QLabel("遍历步长："))
            self.node_step_spin = QSpinBox()
            self.node_step_spin.setRange(1, 50)
            self.node_step_spin.setValue(2)
            auto_row.addWidget(self.node_step_spin)
            auto_row.addStretch()
            layout.addWidget(self.auto_widget)
            self.auto_widget.setVisible(False)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel("激活函数："))
            self.act_combo = QComboBox()
            self.act_combo.addItems(["sigmoid", "tanh", "relu"])
            row2.addWidget(self.act_combo)
            row2.addSpacing(20)
            row2.addWidget(QLabel("学习率："))
            self.lr_spin = self._dspin(0.0001, 1.0, 0.05)
            row2.addWidget(self.lr_spin)
            row2.addSpacing(20)
            row2.addWidget(QLabel("训练代数："))
            self.epoch_spin = QSpinBox()
            self.epoch_spin.setRange(10, 20000)
            self.epoch_spin.setValue(500)
            row2.addWidget(self.epoch_spin)
            row2.addStretch()
            layout.addLayout(row2)
            self.net_mode_combo.currentTextChanged.connect(self._on_net_mode)
        return group

    def _dspin(self, lo, hi, val):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(4 if lo < 0.01 else 3)
        spin.setValue(val)
        spin.setSingleStep(max((hi - lo) / 100, spin.decimals() * 0.001))
        return spin

    def _on_net_mode(self, text):
        manual = text == "手动设置"
        self.manual_widget.setVisible(manual)
        self.auto_widget.setVisible(not manual)

    # ------------------------------------------------------------------ 数据
    def _bounds(self):
        result = {}
        for r, f in enumerate(self.factors):
            try:
                lo = float(self.bounds_table.item(r, 1).text())
                hi = float(self.bounds_table.item(r, 2).text())
            except (TypeError, ValueError):
                lo, hi = factor_limits(f)
            if hi < lo:
                lo, hi = hi, lo
            result[f.name] = [lo, hi]
        return result

    def collect_config(self):
        model_cfg = {}
        if self.model_key == "Kriging":
            model_cfg = {
                "correlation": self.corr_combo.currentText(),
                "trend": self.regress_combo.currentText(),
            }
        elif self.model_key == "SVR":
            model_cfg = {
                "kernel": self.kernel_combo.currentText(),
                "poly_degree": self.poly_degree_spin.value(),
                "c_min": self.c_min.value(), "c_max": self.c_max.value(),
                "c_step": self.c_step.value(), "g_min": self.g_min.value(),
                "g_max": self.g_max.value(), "g_step": self.g_step.value(),
            }
        else:
            node_search = None
            if self.net_mode_combo.currentText() == "自动寻优":
                node_search = [self.node_min_spin.value(), self.node_max_spin.value(),
                               self.node_step_spin.value()]
            model_cfg = {
                "hidden_layers": self.hidden_layers_spin.value(),
                "hidden_nodes": self.hidden_nodes_spin.value(),
                "activation": self.act_combo.currentText(),
                "learning_rate": self.lr_spin.value(),
                "epochs": self.epoch_spin.value(),
                "node_search": node_search,
                "net_mode": self.net_mode_combo.currentText(),
            }
        return {
            "model": self.model_key,
            "model_params": model_cfg,
            "sample_count": self.sample_count_spin.value(),
            "doe_method": self.doe_method_combo.currentText(),
            "bounds": self._bounds(),
        }

    def _on_ok(self):
        bounds = self._bounds()
        for name, (lo, hi) in bounds.items():
            if hi <= lo:
                QMessageBox.warning(self, "参数错误", f"变量 {name} 的上下界无效，请检查。")
                return
        if self.sample_count_spin.value() < len(self.factors) + 1:
            QMessageBox.warning(self, "参数错误", "样本点数量过少，建议不小于因子数+1。")
            return
        self.accept()
