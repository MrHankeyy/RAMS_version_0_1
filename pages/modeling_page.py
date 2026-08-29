from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QPainter, QPen, QColor, QPalette
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
        QTableWidget, QComboBox, QLineEdit, QHeaderView, QGroupBox,
    QCheckBox, QMessageBox
)
from models import ResponseItem, FactorItem, ProjectData


class MultiLevelHeader(QHeaderView):
    """多级表头：支持「分组大栏 → 小栏 → 叶子列」的层级渲染。"""

    def __init__(self, levels, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._leaves = []          # 最底层叶子列: (起始列, 结束列, 文本)
        self._group_bands = []     # 分组带: _group_bands[d] 存放第 d 层的分组
        self._leaf_count = 0
        for node in levels:
            self._build(node, 0)
        self._band_height = 28
        self._header_height = self._band_height * (len(self._group_bands) + 1)
        self.setFixedHeight(self._header_height)
        self.setSectionsClickable(False)
        self.setHighlightSections(False)
        self.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sectionResized.connect(lambda *_: self.viewport().update())
        self.sectionMoved.connect(lambda *_: self.viewport().update())

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self._header_height)

    def _build(self, node, depth):
        """递归解析表头结构：叶子占一列，分组记录其跨列范围。"""
        if isinstance(node, str):
            start = self._leaf_count
            self._leaf_count += 1
            self._leaves.append((start, start, node))
            return
        label, children = node
        start = self._leaf_count
        for child in children:
            self._build(child, depth + 1)
        end = self._leaf_count - 1
        while len(self._group_bands) <= depth:
            self._group_bands.append([])
        self._group_bands[depth].append((start, end, label))

    def paintEvent(self, event):
        painter = QPainter(self.viewport())
        painter.fillRect(event.rect(), self.palette().window())
        self._draw_band(painter, 0, self._leaves, bold=False)
        for band_idx, groups in enumerate(self._group_bands, start=1):
            self._draw_band(painter, band_idx, groups, bold=True)

    def _draw_band(self, painter, band_index, items, bold):
        band_h = self._band_height
        if band_index == 0:
            y = self.height() - band_h
        else:
            y = (band_index - 1) * band_h
        font = self.font()
        font.setBold(bold)
        painter.setFont(font)
        for start, end, label in items:
            x0 = self.sectionViewportPosition(start)
            x1 = self.sectionViewportPosition(end) + self.sectionSize(end)
            if x1 <= 0 or x0 >= self.viewport().width():
                continue
            rect = QRect(x0, y, x1 - x0, band_h)
            painter.fillRect(rect, QColor(245, 245, 247))
            painter.setPen(QPen(QColor(200, 200, 210)))
            painter.drawRect(rect)
            painter.setPen(self.palette().color(QPalette.ColorRole.Text))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)


class ModelingPage(QWidget):
    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ========= 响应区 =========
        response_group = QGroupBox("响应")
        response_group.setObjectName("response_group")
        response_layout = QVBoxLayout()
        response_layout.setContentsMargins(10, 12, 10, 10)

        response_btn_layout = QHBoxLayout()
        self.response_add_btn = QPushButton("+")
        self.response_add_btn.setFixedWidth(35)
        self.response_add_btn.clicked.connect(self.add_response_row)
        response_btn_layout.addWidget(self.response_add_btn)
        response_btn_layout.addStretch()
        response_layout.addLayout(response_btn_layout)

        self.response_table = QTableWidget(0, 7)
        self.response_table.setHorizontalHeader(MultiLevelHeader([
            "名称",
            "目标响应",
            "约束响应",
            "目标特征",
            "稳定性阈值",
            "单位",
            "操作",
        ]))
        self.response_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.response_table.verticalHeader().setVisible(False)
        response_layout.addWidget(self.response_table)

        response_group.setLayout(response_layout)
        main_layout.addWidget(response_group)

        # ========= 因子区 =========
        factor_group = QGroupBox("因子")
        factor_group.setObjectName("factor_group")
        factor_layout = QVBoxLayout()
        factor_layout.setContentsMargins(10, 12, 10, 10)

        factor_hint = QLabel("填写不确定性参数；点击“固定值”后，该因子将按均值或区间中点作为确定值使用。")
        factor_hint.setStyleSheet("color: #888888;")
        factor_layout.addWidget(factor_hint)

        self.design_factor_table = self._create_factor_table("设计因子")
        self.environment_factor_table = self._create_factor_table("环境因子")
        factor_layout.addWidget(self._factor_section("设计因子", self.design_factor_table, self.add_design_factor_row))
        factor_layout.addWidget(self._factor_section("环境因子", self.environment_factor_table, self.add_environment_factor_row))

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
        self.setStyleSheet("""
        QGroupBox {
            margin-top: 12px;
            font-size: 18px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
        }
        #response_group, #factor_group {
            border: 1px solid #c9d2dc;
            border-radius: 5px;
            background: #ffffff;
        }
        QTableWidget {
            gridline-color: #d9e0e7;
            alternate-background-color: #f7f9fb;
        }
        """)

    def on_save_clicked(self):
        """
        触发：将界面上的输入数据传递到后端（同步到数据模型 ProjectData 中）
        并可在此调用后续算法
        """
        errors = self._validate_response_inputs()
        if errors:
            QMessageBox.warning(self, "业务建模数据校验未通过", "\n".join(errors))
            return

        self.sync_to_project_data()

        # 演示：打印收集到的数据（模拟传递给后端算法）
        print("======== 业务建模数据已保存 ========")
        print(f"包含 {len(self.project_data.responses)} 个响应指标:")
        for r in self.project_data.responses:
            if r.feature == "望目":
                print(f"  - {r.name}: {r.kind} ({r.feature}), 稳定性阈值区间: [{r.lower}, {r.upper}] {r.unit}")
            else:
                print(f"  - {r.name}: {r.kind} ({r.feature}), 稳定性阈值: {r.robust_limit} {r.unit}")

        print(f"包含 {len(self.project_data.factors)} 个因子:")
        for f in self.project_data.factors:
            if f.uncertainty == "概率":
                print(f"  - {f.name}: 连续 | {f.source} | 概率({f.distribution}), 均值: {f.param1}, 方差: {f.param2} {f.unit}")
            else:
                print(f"  - {f.name}: 连续 | {f.source} | 区间, 范围: [{f.param1}, {f.param2}] {f.unit}")
        print("====================================")

    def _validate_response_inputs(self):
        """校验响应表：目标/约束至少勾选一项；望目必须填写逗号分隔的上下界。"""
        errors = []
        for row in range(self.response_table.rowCount()):
            label = f"响应「{self._cell_text(self.response_table, row, 0) or f'第{row + 1}行'}」"
            obj_widget = self.response_table.cellWidget(row, 1)
            cons_widget = self.response_table.cellWidget(row, 2)
            is_obj = bool(obj_widget and obj_widget.centered_widget.isChecked())
            is_cons = bool(cons_widget and cons_widget.centered_widget.isChecked())
            if not is_obj and not is_cons:
                errors.append(f"{label}：请勾选「目标」或「约束」。")
            feature = self._cell_text(self.response_table, row, 3)
            threshold = self._cell_text(self.response_table, row, 4).replace("，", ",")
            if feature == "望目":
                parts = [p.strip() for p in threshold.split(",")]
                if not threshold or len(parts) != 2 or not all(self._is_number(p) for p in parts):
                    errors.append(f"{label}：望目需在「稳定性阈值」中填写逗号分隔的下限、上限两个数字（均必填），例如 3,8。")
            elif threshold and not self._is_number(threshold):
                errors.append(f"{label}：望大/望小时「稳定性阈值」需为单个数字。")
        return errors

    @staticmethod
    def _is_number(text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def _cell_text(table, row, col):
        widget = table.cellWidget(row, col)
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return ""

    # =========================
    # 响应表：新增一行
    # =========================
    def add_response_row(self):
        row = self.response_table.rowCount()
        self.response_table.insertRow(row)

        self.response_table.setCellWidget(row, 0, QLineEdit())

        obj_check = QCheckBox()
        obj_check.setToolTip("勾选后该响应作为优化目标")
        cons_check = QCheckBox()
        cons_check.setToolTip("勾选后该响应作为稳定性约束")
        self.response_table.setCellWidget(row, 1, self._make_centered(obj_check))
        self.response_table.setCellWidget(row, 2, self._make_centered(cons_check))

        feature_combo = QComboBox()
        feature_combo.addItems(["望大", "望小", "望目"])
        self.response_table.setCellWidget(row, 3, feature_combo)

        threshold_edit = QLineEdit()
        threshold_edit.setPlaceholderText("稳定要求最小值，如 10")
        self.response_table.setCellWidget(row, 4, threshold_edit)

        self.response_table.setCellWidget(row, 5, QLineEdit())  # 单位

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(lambda _, r=row: self.delete_response_row_by_button(r))
        self.response_table.setCellWidget(row, 6, delete_btn)

        # 稳定性阈值与目标特征联动：望大=稳定要求最小值，望小=稳定要求最大值，望目=下限,上限
        def on_feature_changed(text):
            if text == "望目":
                threshold_edit.setPlaceholderText("下限,上限（逗号分隔，两项均必填）")
            elif text == "望大":
                threshold_edit.setPlaceholderText("稳定要求最小值，如 10")
            else:
                threshold_edit.setPlaceholderText("稳定要求最大值，如 5")

        feature_combo.currentTextChanged.connect(on_feature_changed)
        on_feature_changed(feature_combo.currentText())

        self.refresh_response_delete_buttons()

    @staticmethod
    def _make_centered(widget):
        """将紧凑控件（如复选框）水平居中放入单元格。"""
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(widget)
        wrapper.centered_widget = widget
        return wrapper

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
    def _create_factor_table(self, title):
        table = QTableWidget(0, 8)
        table.setHorizontalHeader(MultiLevelHeader([
            "名称", "不确定类型", "分布类型", "P1(下限/均值)",
            "P2(上限/方差)", "固定值（均值/中值）", "单位", "操作",
        ]))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        return table

    def _factor_section(self, title, table, add_handler):
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        label = QLabel(title)
        label.setStyleSheet("font-size: 16px; font-weight: bold; color: #34495e;")
        add_button = QPushButton("+ 新增因子")
        add_button.clicked.connect(add_handler)
        header.addWidget(label)
        header.addStretch()
        header.addWidget(add_button)
        layout.addLayout(header)
        layout.addWidget(table)
        return section

    def add_design_factor_row(self):
        self._add_factor_row(self.design_factor_table)

    def add_environment_factor_row(self):
        self._add_factor_row(self.environment_factor_table)

    def _add_factor_row(self, table):
        row = table.rowCount()
        table.insertRow(row)
        table.setCellWidget(row, 0, QLineEdit())
        self._create_continuous_block(table, row, 1, 2, 3, 4)

        fixed_btn = QPushButton("固定值")
        fixed_btn.clicked.connect(lambda _, t=table, r=row: self._fix_factor_value(t, r))
        table.setCellWidget(row, 5, fixed_btn)
        table.setCellWidget(row, 6, QLineEdit())

        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(lambda _, t=table, r=row: self._delete_factor_row(t, r))
        table.setCellWidget(row, 7, delete_btn)
        self._refresh_factor_delete_buttons(table)

    def _create_continuous_block(self, table, row, u_col, d_col, p1_col, p2_col):
        """创建连续因子的不确定性参数输入。"""
        uncertainty_combo = QComboBox()
        uncertainty_combo.addItems(["区间", "概率"])
        table.setCellWidget(row, u_col, uncertainty_combo)

        dist_combo = QComboBox()
        dist_combo.addItems(["无", "正态分布", "均匀分布", "对数正态分布"])
        dist_combo.setEnabled(False)  # 默认区间，关闭分布类型选择
        table.setCellWidget(row, d_col, dist_combo)

        p1_edit = QLineEdit()
        p1_edit.setPlaceholderText("下限")
        table.setCellWidget(row, p1_col, p1_edit)

        p2_edit = QLineEdit()
        p2_edit.setPlaceholderText("上限")
        table.setCellWidget(row, p2_col, p2_edit)

        # 选择“概率”时启用分布类型，并切换 P1/P2 提示词
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

        uncertainty_combo.currentTextChanged.connect(on_uncertainty_changed)

    def _fix_factor_value(self, table, row):
        uncertainty = self._cell_text(table, row, 1)
        p1 = self._cell_text(table, row, 3)
        p2 = self._cell_text(table, row, 4)
        try:
            value = (float(p1) + float(p2)) / 2 if uncertainty == "区间" else float(p1)
            fixed_value = str(round(value, 8))
        except ValueError:
            fixed_value = ""
        button = table.cellWidget(row, 5)
        if fixed_value and button:
            button.setText(f"已固定: {fixed_value}")
            button.setProperty("fixed_value", fixed_value)
            button.setStyleSheet("background-color: #d8f3dc; color: #1b4332;")

    def _delete_factor_row(self, table, row):
        if 0 <= row < table.rowCount():
            table.removeRow(row)
            self._refresh_factor_delete_buttons(table)

    def _refresh_factor_delete_buttons(self, table):
        for row in range(table.rowCount()):
            btn = table.cellWidget(row, 7)
            if btn is not None:
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(lambda _, t=table, r=row: self._delete_factor_row(t, r))

    # =========================
    # 获取当前页面数据
    # =========================
    def get_response_data(self):
        data = []
        for row in range(self.response_table.rowCount()):
            obj_widget = self.response_table.cellWidget(row, 1)
            cons_widget = self.response_table.cellWidget(row, 2)
            row_data = {
                "名称": self._cell_text(self.response_table, row, 0),
                "目标": bool(obj_widget and obj_widget.centered_widget.isChecked()),
                "约束": bool(cons_widget and cons_widget.centered_widget.isChecked()),
                "目标特征": self._cell_text(self.response_table, row, 3),
                "稳定性阈值": self._cell_text(self.response_table, row, 4),
                "单位": self._cell_text(self.response_table, row, 5),
            }
            data.append(row_data)
        return data

    def get_factor_data(self):
        data = []
        for table, source in ((self.design_factor_table, "设计"), (self.environment_factor_table, "环境")):
            for row in range(table.rowCount()):
                name = self._cell_text(table, row, 0)
                if not name:
                    continue
                fixed_button = table.cellWidget(row, 5)
                data.append({
                    "名称": name,
                    "设计/环境": source,
                    "不确定性": self._cell_text(table, row, 1),
                    "分布类型": self._cell_text(table, row, 2),
                    "P1": self._cell_text(table, row, 3),
                    "P2": self._cell_text(table, row, 4),
                    "固定值": (fixed_button.property("fixed_value") or "") if fixed_button else "",
                    "单位": self._cell_text(table, row, 6),
                })
        return data

    def sync_to_project_data(self):
        responses = []
        for item in self.get_response_data():
            feature = item["目标特征"]
            threshold = item["稳定性阈值"].replace("，", ",")
            if feature == "望目":
                parts = [p.strip() for p in threshold.split(",")]
                lower = parts[0] if len(parts) >= 1 else ""
                upper = parts[1] if len(parts) >= 2 else ""
                robust_limit = ""
            else:
                lower, upper, robust_limit = "", "", threshold
            kind = "目标" if item["目标"] else ""
            if item["约束"]:
                kind = kind + "+约束" if kind else "约束"
            responses.append(ResponseItem(
                name=item["名称"],
                kind=kind,
                feature=feature,
                lower=lower,
                upper=upper,
                robust_limit=robust_limit,
                unit=item["单位"],
            ))
        self.project_data.responses = responses

        self.project_data.factors = [
            FactorItem(
                name=item["名称"],
                source=item["设计/环境"],
                uncertainty=item["不确定性"],
                distribution=item["分布类型"],
                param1=item["P1"],
                param2=item["P2"],
                fixed_value=item["固定值"],
                is_fixed=bool(item["固定值"]),
                unit=item["单位"],
            )
            for item in self.get_factor_data()
        ]

