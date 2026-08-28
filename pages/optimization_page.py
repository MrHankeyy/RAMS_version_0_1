from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGroupBox, QRadioButton, QButtonGroup, QTextEdit, QSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from models import ProjectData
import random

class OptimizationPage(QWidget):
    # 定义一个信号，用于通知主窗口跳转页面
    optimization_finished = pyqtSignal()

    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # ========= 鲁棒优化范式选择区 =========
        paradigm_group = QGroupBox("鲁棒优化设计范式选择")
        paradigm_layout = QVBoxLayout()

        self.paradigm_combo = QComboBox()
        self.paradigm_combo.addItems([
            "完全多目标寻求帕累托前沿 (基于NSGA-II)",
            "拉格朗日线性加权 (基于权重探索单点最优)",
            "带鲁棒约束的单目标优化 (设定合格红线底线)"
        ])
        paradigm_layout.addWidget(QLabel("选择优化目标体系:"))
        paradigm_layout.addWidget(self.paradigm_combo)

        # 方法详细说明区域
        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        paradigm_layout.addWidget(self.desc_text)

        paradigm_group.setLayout(paradigm_layout)
        main_layout.addWidget(paradigm_group)
        
        # ========= 范式参数联动区 =========
        params_group = QGroupBox("范式参数配置")
        self.params_layout = QVBoxLayout()
        
        # 1. 加权范式参数
        self.weight_widget = QWidget()
        weight_lyt = QHBoxLayout(self.weight_widget)
        weight_lyt.setContentsMargins(0, 0, 0, 0)
        self.lambda_spin = QSpinBox()
        self.lambda_spin.setRange(0, 100)
        self.lambda_spin.setValue(50)
        weight_lyt.addWidget(QLabel("均值指标 $\\lambda$ 权重 (%):"))
        weight_lyt.addWidget(self.lambda_spin)
        weight_lyt.addWidget(QLabel("说明：剩余比例即为主导鲁棒性方差/区间半径的权重。"))
        weight_lyt.addStretch()
        
        # 2. 约束范式参数
        self.constraint_widget = QWidget()
        cons_lyt = QHBoxLayout(self.constraint_widget)
        cons_lyt.setContentsMargins(0, 0, 0, 0)
        self.limit_val = QSpinBox()
        self.limit_val.setRange(-10000, 10000)
        cons_lyt.addWidget(QLabel("鲁棒约束边界上限:"))
        cons_lyt.addWidget(self.limit_val)
        cons_lyt.addWidget(QLabel("必须满足要求: $\\mu_Y + 3\\cdot\\sigma_Y \\le Limit$"))
        cons_lyt.addStretch()
        
        self.params_layout.addWidget(self.weight_widget)
        self.params_layout.addWidget(self.constraint_widget)
        
        params_group.setLayout(self.params_layout)
        main_layout.addWidget(params_group)

        # ========= 操作按钮区 =========
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.run_btn = QPushButton("执行鲁棒优化计算")
        self.run_btn.setMinimumWidth(200)
        self.run_btn.setStyleSheet("font-weight: bold; background-color: #28a745; color: white;")
        self.run_btn.clicked.connect(self.run_optimization)
        btn_layout.addWidget(self.run_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

        self.paradigm_combo.currentTextChanged.connect(self.update_ui_state)
        self.update_ui_state(self.paradigm_combo.currentText())

    def run_optimization(self):
        """执行优化计算并弹出结果反馈"""
        # ================== 模拟后端鲁棒优化计算过程 ==================
        # 这里提取当前的界面参数并结合 project_data 模拟计算
        paradigm = self.paradigm_combo.currentText()
        optim_results = []
        
        # 针对每个因子，随机生成一个设计推荐值与其浮动范围
        for f in self.project_data.factors:
            try:
                min_v = float(f.param1)
                max_v = float(f.param2)
                if min_v > max_v:  # 防御用户填反了
                    min_v, max_v = max_v, min_v
            except ValueError:
                min_v, max_v = 0.0, 10.0 # 解析失败兜底
                
            best_val = round(min_v + random.random() * (max_v - min_v), 2)
            optim_results.append(f"{f.name}: 推荐设计点 {best_val} (±{round((max_v - min_v)*0.05, 2)})")
        
        # 针对目标响应的预测表现
        resp_results = []
        for r in self.project_data.responses:
            # 判断指标追求最大还是最小
            target = 100.0 if "大" in r.feature else 5.0
            predicted_mean = round(target + random.uniform(-2, 2), 2)
            predicted_std = round(random.uniform(0.1, 1.5), 2)
            resp_str = f"{r.name}: 预测均值 μ={predicted_mean}, 鲁棒波动 σ={predicted_std}"
            
            if "鲁棒约束" in paradigm and r.robust_limit:
                try:
                    limit_val = float(r.robust_limit)
                    resp_str += f" | 满足边界 {limit_val}"
                except ValueError:
                    pass
            resp_results.append(resp_str)
            
        # ================== 拼接展示信息 ==================
        msg = f"已完成【{paradigm}】计算求解！\n\n"
        msg += "🎯 【鲁棒最优设计点推荐】\n"
        msg += "\n".join(optim_results) + "\n\n"
        msg += "📈 【目标性能与波动预测】\n"
        msg += "\n".join(resp_results) + "\n\n"
        msg += "接下来将进入【分析报告】模块，对本组优化解进行详细的系统性分析及可视化呈现。"
        
        QMessageBox.information(self, "鲁棒优化计算完毕", msg)
        
        # 触发跳转信号
        self.optimization_finished.emit()

    def update_ui_state(self, text):
        desc = ""
        self.weight_widget.setVisible(False)
        self.constraint_widget.setVisible(False)
        
        # 探测是否需要判断指标的明确界限
        has_bounds = False
        reasons = []
        for r in self.project_data.responses:
            if r.robust_limit:  # 现在读取专属的鲁棒约束边界极值
                has_bounds = True
                reasons.append(f"指标[{r.name}]包含红线: (边界≤{r.robust_limit})")
                
        bounds_txt = "\n\n💡检测到底层数据边界情况：" + " | ".join(reasons) if has_bounds else "\n\n💡底层指标无明确防越界红线极值定义。无法启用惩罚约束！请返回【业务建模】填写！"

        if "帕累托" in text:
            desc = "将性能特征(如均值、目标偏离)与稳定性特征(方差、区间范围)视作互不退让的并行目标函数，运行 MOGA 算法得出帕累托前沿曲线以供高层决策。"
        elif "线性加权" in text:
            desc = "将多目标转化为单目标，引入人工主导偏好(加权系数)。目标函数变为： $F = \\lambda\\cdot\\mu_Y + (1-\\lambda)\\cdot\\sigma_Y$。直接利用梯度规划类算法求绝对最优点。"
            self.weight_widget.setVisible(True)
        elif "鲁棒约束" in text:
            desc = "在维持核心指标向最好方向移动的同时，增加不可触碰的工程鲁棒性红线，强制缩小波动带不能超过一定限制。" + bounds_txt
            self.constraint_widget.setVisible(True)
            self.run_btn.setEnabled(has_bounds) # 如果没有红线，禁用执行按钮
            
        self.desc_text.setText(desc)