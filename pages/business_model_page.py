from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QTableWidget, QComboBox, QLineEdit, QHeaderView, QGroupBox
)
from models import ResponseItem, FactorItem, ProjectData


class BusinessModelPage(QWidget):
    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        title = QLabel("业务建模模块")
        title.setStyleSheet("color: red; font-weight: bold; font-size: 18px;")
        main_layout.addWidget(title)

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

        self.response_table = QTableWidget(0, 7)
        self.response_table.setHorizontalHeaderLabels(
            ["名称", "目标/约束", "目标特征", "下限", "上限", "单位", "操作"]
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

        self.factor_table = QTableWidget(0, 8)
        self.factor_table.setHorizontalHeaderLabels(
            ["名称", "连续/离散", "设计/环境", "概率/区间", "下限", "上限", "单位", "操作"]
        )
        self.factor_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.factor_table.verticalHeader().setVisible(False)
        factor_layout.addWidget(self.factor_table)

        factor_group.setLayout(factor_layout)
        main_layout.addWidget(factor_group)

        self.setLayout(main_layout)

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
        self.response_table.setCellWidget(row, 5, QLineEdit())

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(lambda _, r=row: self.delete_response_row_by_button(r))
        self.response_table.setCellWidget(row, 6, delete_btn)

        self.refresh_response_delete_buttons()

    def delete_response_row_by_button(self, row):
        if 0 <= row < self.response_table.rowCount():
            self.response_table.removeRow(row)
            self.refresh_response_delete_buttons()

    def refresh_response_delete_buttons(self):
        for row in range(self.response_table.rowCount()):
            btn = self.response_table.cellWidget(row, 6)
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
        prob_interval_combo.addItems(["概率", "区间"])
        self.factor_table.setCellWidget(row, 3, prob_interval_combo)

        self.factor_table.setCellWidget(row, 4, QLineEdit())
        self.factor_table.setCellWidget(row, 5, QLineEdit())
        self.factor_table.setCellWidget(row, 6, QLineEdit())

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(lambda _, r=row: self.delete_factor_row_by_button(r))
        self.factor_table.setCellWidget(row, 7, delete_btn)

        self.refresh_factor_delete_buttons()

    def delete_factor_row_by_button(self, row):
        if 0 <= row < self.factor_table.rowCount():
            self.factor_table.removeRow(row)
            self.refresh_factor_delete_buttons()

    def refresh_factor_delete_buttons(self):
        for row in range(self.factor_table.rowCount()):
            btn = self.factor_table.cellWidget(row, 7)
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
                "单位": self.response_table.cellWidget(row, 5).text(),
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
                "概率/区间": self.factor_table.cellWidget(row, 3).currentText(),
                "下限": self.factor_table.cellWidget(row, 4).text(),
                "上限": self.factor_table.cellWidget(row, 5).text(),
                "单位": self.factor_table.cellWidget(row, 6).text(),
            }
            data.append(row_data)
        return data
    
    def sync_to_project_data(self):
        self.project_data.responses = [
            ResponseItem(
                name=item["名称"],
                kind=item["目标/约束"],
                feature=item["目标特征"],
                lower=item["下限"],
                upper=item["上限"],
                unit=item["单位"],
            )
            for item in self.get_response_data()
        ]

        self.project_data.factors = [
            FactorItem(
                name=item["名称"],
                mode=item["连续/离散"],
                source=item["设计/环境"],
                uncertainty=item["概率/区间"],
                lower=item["下限"],
                upper=item["上限"],
                unit=item["单位"],
            )
            for item in self.get_factor_data()
        ]
            