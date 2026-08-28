from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QTableWidget, QComboBox, QLineEdit, QHeaderView, QGroupBox
)
from models import ResponseItem, FactorItem, ProjectData


class ModelingPage(QWidget):
    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # ========= 项目类型设置区 =========
        type_group = QGroupBox("实验类型设置")
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("选择实验类型(区分实验成本)："))
        self.exp_type_combo = QComboBox()
        self.exp_type_combo.addItems(["实际实验", "仿真实验"])
        
        # 加载已有数据
        idx = self.exp_type_combo.findText(self.project_data.experiment_type)
        if idx >= 0: self.exp_type_combo.setCurrentIndex(idx)
            
        type_layout.addWidget(self.exp_type_combo)
        type_layout.addStretch()
        type_group.setLayout(type_layout)
        main_layout.addWidget(type_group)

        # ========= 响应区 =========
        response_group = QGroupBox("响应")
        response_layout = QVBoxLayout()

        response_btn_layout = QHBoxLayout()
        self.response_add_btn = QPushButton("+")
        self.response_add_btn.setFixedWidth(35)
        self.response_add_btn.clicked.connect(self.add_response_row)
        response_btn_layout.addWidget(self.response_add_btn)
        response_btn_layout.addStretch()
        response_layout.addLayout(response_btn_layout)

        self.response_table = QTableWidget(0, 8)
        self.response_table.setHorizontalHeaderLabels(
            ["名称", "目标/约束", "目标特征", "目标下限", "目标上限", "鲁棒红线极值", "单位", "操作"]
        )
        self.response_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.response_table.verticalHeader().setVisible(False)
        response_layout.addWidget(self.response_table)

        response_group.setLayout(response_layout)
        main_layout.addWidget(response_group)

        # ========= 因子区 =========
        factor_group = QGroupBox("因子")
        factor_layout = QVBoxLayout()

        factor_btn_layout = QHBoxLayout()
        self.factor_add_btn = QPushButton("+")
        self.factor_add_btn.setFixedWidth(35)
        self.factor_add_btn.clicked.connect(self.add_factor_row)
        factor_btn_layout.addWidget(self.factor_add_btn)
        factor_btn_layout.addStretch()
        factor_layout.addLayout(factor_btn_layout)

        self.factor_table = QTableWidget(0, 9)
        self.factor_table.setHorizontalHeaderLabels(
            ["名称", "连续/离散", "设计/环境", "不确定性", "分布类型", "P1(下限/均值)", "P2(上限/方差)", "单位", "操作"]
        )
        self.factor_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.factor_table.verticalHeader().setVisible(False)
        factor_layout.addWidget(self.factor_table)

        factor_group.setLayout(factor_layout)
        main_layout.addWidget(factor_group)

        # ========= 操作按钮区 =========
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.save_btn = QPushButton("保存业务模型数据")
        self.save_btn.setMinimumWidth(150)
        self.save_btn.clicked.connect(self.on_save_clicked)
        btn_layout.addWidget(self.save_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def on_save_clicked(self):
        """
        触发：将界面上的输入数据传递到后端（同步到数据模型 ProjectData 中）
        并可在此调用后续算法
        """
        self.sync_to_project_data()
        
        # 演示：打印收集到的数据（模拟传递给后端算法）
        print("======== 业务摸建数据已保存 ========")
        print(f"包含 {len(self.project_data.responses)} 个响应指标:")
        for r in self.project_data.responses:
            print(f"  - {r.name}: {r.kind} ({r.feature}), 范围: [{r.lower}, {r.upper}] {r.unit}")
            
        print(f"包含 {len(self.project_data.factors)} 个因子:")
        for f in self.project_data.factors:
            if f.uncertainty == "概率":
                print(f"  - {f.name}: {f.mode} | {f.source} | 概率({f.distribution}), 均值: {f.param1}, 方差: {f.param2} {f.unit}")
            else:
                print(f"  - {f.name}: {f.mode} | {f.source} | 区间, 范围: [{f.param1}, {f.param2}] {f.unit}")
        print("====================================")

    # =========================
    # 响应表：新增一行
    # =========================
    def add_response_row(self):
        row = self.response_table.rowCount()
        self.response_table.insertRow(row)

        self.response_table.setCellWidget(row, 0, QLineEdit())

        obj_constraint_combo = QComboBox()
        obj_constraint_combo.addItems(["目标", "约束"])
        self.response_table.setCellWidget(row, 1, obj_constraint_combo)

        target_feature_combo = QComboBox()
        target_feature_combo.addItems(["望大", "望小", "望目"])
        self.response_table.setCellWidget(row, 2, target_feature_combo)

        self.response_table.setCellWidget(row, 3, QLineEdit())
        self.response_table.setCellWidget(row, 4, QLineEdit())
        
        robust_edit = QLineEdit()
        robust_edit.setPlaceholderText("选填(触发展望约束)")
        self.response_table.setCellWidget(row, 5, robust_edit)
        
        self.response_table.setCellWidget(row, 6, QLineEdit())

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(lambda _, r=row: self.delete_response_row_by_button(r))
        self.response_table.setCellWidget(row, 7, delete_btn)

        self.refresh_response_delete_buttons()

    def delete_response_row_by_button(self, row):
        if 0 <= row < self.response_table.rowCount():
            self.response_table.removeRow(row)
            self.refresh_response_delete_buttons()

    def refresh_response_delete_buttons(self):
        for row in range(self.response_table.rowCount()):
            btn = self.response_table.cellWidget(row, 7)
            if btn is not None:
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(lambda _, r=row: self.delete_response_row_by_button(r))

    # =========================
    # 因子表：新增一行
    # =========================
    def add_factor_row(self):
        row = self.factor_table.rowCount()
        self.factor_table.insertRow(row)

        self.factor_table.setCellWidget(row, 0, QLineEdit())

        continuous_discrete_combo = QComboBox()
        continuous_discrete_combo.addItems(["连续", "离散"])
        self.factor_table.setCellWidget(row, 1, continuous_discrete_combo)

        design_env_combo = QComboBox()
        design_env_combo.addItems(["设计", "环境"])
        self.factor_table.setCellWidget(row, 2, design_env_combo)

        prob_interval_combo = QComboBox()
        prob_interval_combo.addItems(["区间", "概率"])
        self.factor_table.setCellWidget(row, 3, prob_interval_combo)

        dist_combo = QComboBox()
        dist_combo.addItems(["无", "正态分布", "均匀分布", "对数正态分布"])
        dist_combo.setEnabled(False) # 默认区间，关闭分布类型选择
        self.factor_table.setCellWidget(row, 4, dist_combo)

        p1_edit = QLineEdit()
        p1_edit.setPlaceholderText("下限")
        self.factor_table.setCellWidget(row, 5, p1_edit)

        p2_edit = QLineEdit()
        p2_edit.setPlaceholderText("上限")
        self.factor_table.setCellWidget(row, 6, p2_edit)

        self.factor_table.setCellWidget(row, 7, QLineEdit())

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(lambda _, r=row: self.delete_factor_row_by_button(r))
        self.factor_table.setCellWidget(row, 8, delete_btn)

        # 联动信号，当选择“概率”时，启用分布特征，并修改提示词
        def on_uncertainty_changed(text):
            if text == "概率":
                dist_combo.setEnabled(True)
                p1_edit.setPlaceholderText("均值")
                p2_edit.setPlaceholderText("方差")
            else:
                dist_combo.setEnabled(False)
                dist_combo.setCurrentText("无")
                p1_edit.setPlaceholderText("下限")
                p2_edit.setPlaceholderText("上限")
                
        prob_interval_combo.currentTextChanged.connect(on_uncertainty_changed)

        self.refresh_factor_delete_buttons()

    def delete_factor_row_by_button(self, row):
        if 0 <= row < self.factor_table.rowCount():
            self.factor_table.removeRow(row)
            self.refresh_factor_delete_buttons()

    def refresh_factor_delete_buttons(self):
        for row in range(self.factor_table.rowCount()):
            btn = self.factor_table.cellWidget(row, 8)
            if btn is not None:
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(lambda _, r=row: self.delete_factor_row_by_button(r))

    # =========================
    # 获取当前页面数据
    # =========================
    def get_response_data(self):
        data = []
        for row in range(self.response_table.rowCount()):
            row_data = {
                "名称": self.response_table.cellWidget(row, 0).text(),
                "目标/约束": self.response_table.cellWidget(row, 1).currentText(),
                "目标特征": self.response_table.cellWidget(row, 2).currentText(),
                "下限": self.response_table.cellWidget(row, 3).text(),
                "上限": self.response_table.cellWidget(row, 4).text(),
                "鲁棒红线极值": self.response_table.cellWidget(row, 5).text(),
                "单位": self.response_table.cellWidget(row, 6).text(),
            }
            data.append(row_data)
        return data

    def get_factor_data(self):
        data = []
        for row in range(self.factor_table.rowCount()):
            row_data = {
                "名称": self.factor_table.cellWidget(row, 0).text(),
                "连续/离散": self.factor_table.cellWidget(row, 1).currentText(),
                "设计/环境": self.factor_table.cellWidget(row, 2).currentText(),
                "不确定性": self.factor_table.cellWidget(row, 3).currentText(),
                "分布类型": self.factor_table.cellWidget(row, 4).currentText(),
                "P1": self.factor_table.cellWidget(row, 5).text(),
                "P2": self.factor_table.cellWidget(row, 6).text(),
                "单位": self.factor_table.cellWidget(row, 7).text(),
            }
            data.append(row_data)
        return data
    
    def sync_to_project_data(self):
        self.project_data.experiment_type = self.exp_type_combo.currentText()
        
        self.project_data.responses = [
            ResponseItem(
                name=item["名称"],
                kind=item["目标/约束"],
                feature=item["目标特征"],
                lower=item["下限"],
                upper=item["上限"],
                robust_limit=item["鲁棒红线极值"],
                unit=item["单位"],
            )
            for item in self.get_response_data()
        ]

        self.project_data.factors = [
            FactorItem(
                name=item["名称"],
                mode=item["连续/离散"],
                source=item["设计/环境"],
                uncertainty=item["不确定性"],
                distribution=item["分布类型"],
                param1=item["P1"],
                param2=item["P2"],
                unit=item["单位"],
            )
            for item in self.get_factor_data()
        ]
            