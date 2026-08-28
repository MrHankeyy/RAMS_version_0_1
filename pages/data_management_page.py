from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QMessageBox, QFileDialog
)
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
import csv
from models import ProjectData

class ExcelLikeTableWidget(QTableWidget):
    """支持从 Excel/CSV 范围复制、粘贴以及批量删除的增强表格"""
    def keyPressEvent(self, event):
        # Ctrl + C 复制选区
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_to_clipboard()
        # Ctrl + V 粘贴
        elif event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self.paste_from_clipboard()
        # Delete 或 Backspace 批量删除所选项 (仅限可编辑框)
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def copy_to_clipboard(self):
        selected_ranges = self.selectedRanges()
        if not selected_ranges:
            return

        # 找到选区的边界 (支持用户反向框选)
        top_row = min(r.topRow() for r in selected_ranges)
        bottom_row = max(r.bottomRow() for r in selected_ranges)
        left_col = min(r.leftColumn() for r in selected_ranges)
        right_col = max(r.rightColumn() for r in selected_ranges)

        clipboard_text = ""
        for row in range(top_row, bottom_row + 1):
            row_data = []
            for col in range(left_col, right_col + 1):
                item = self.item(row, col)
                row_data.append(item.text() if item else "")
            # 使用 tab (制表符) 分隔同一行数据，应对 Excel 复制规范
            clipboard_text += "\t".join(row_data) + "\n"
        
        QApplication.clipboard().setText(clipboard_text)

    def paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text: return

        # 获取当前选中的起始单元格
        selected_ranges = self.selectedRanges()
        if not selected_ranges:
            start_row, start_col = 0, 0
        else:
            start_row = selected_ranges[0].topRow()
            start_col = selected_ranges[0].leftColumn()

        # 分割行 (去掉末尾可能自带的空换行)
        rows = text.rstrip('\n').split('\n')
        for r_idx, row in enumerate(rows):
            columns = row.split('\t')
            for c_idx, col_text in enumerate(columns):
                target_row = start_row + r_idx
                target_col = start_col + c_idx
                # 检查是否越界
                if target_row < self.rowCount() and target_col < self.columnCount():
                    item = self.item(target_row, target_col)
                    # ★ 核心防线：仅在非锁定状态下（即白色响应列）允许粘贴覆盖改变数据
                    if item and (item.flags() & Qt.ItemFlag.ItemIsEditable):
                        item.setText(col_text.strip())

    def delete_selected(self):
        """批量清空所选且允许编辑的单元格内容"""
        for item in self.selectedItems():
            # 判断这个单元格是否处于可编辑状态
            if item and (item.flags() & Qt.ItemFlag.ItemIsEditable):
                item.setText("")

class DataManagementPage(QWidget):
    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # ========= 顶部控制栏 =========
        ctrl_layout = QHBoxLayout()
        self.title_label = QLabel("当前无加载的数据方案。请先在[方案配置]生成。")
        self.title_label.setStyleSheet("font-weight: bold; color: #333;")
        
        self.load_btn = QPushButton("自动拉取最新 DOE 因子矩阵")
        self.load_btn.setStyleSheet("background-color: #6c757d; color: white;")
        self.load_btn.clicked.connect(self.load_doe_matrix)
        
        self.export_btn = QPushButton("导出为 CSV 表格")
        self.export_btn.clicked.connect(self.export_to_csv)
        
        self.import_btn = QPushButton("从 CSV 导入测试数据")
        self.import_btn.clicked.connect(self.import_from_csv)
        
        ctrl_layout.addWidget(self.title_label)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.load_btn)
        ctrl_layout.addWidget(self.export_btn)
        ctrl_layout.addWidget(self.import_btn)
        main_layout.addLayout(ctrl_layout)

        # ========= 数据表格区 =========
        table_group = QGroupBox("实验测试点交互数据输入 (支持框选并 Ctrl+C / Ctrl+V 与 Excel 互通)")
        table_layout = QVBoxLayout()
        
        self.table = ExcelLikeTableWidget()
        table_layout.addWidget(self.table)
        
        table_group.setLayout(table_layout)
        main_layout.addWidget(table_group)

        # ========= 底部按钮区 =========
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("数据分布初评(未完善)")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.simple_analysis)
        
        self.save_btn = QPushButton("💾 保存测试数据(传递给下一环节)")
        self.save_btn.setMinimumWidth(200)
        self.save_btn.setStyleSheet("font-weight: bold; background-color: #2b78e4; color: white;")
        self.save_btn.clicked.connect(self.save_results)
        
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def load_doe_matrix(self):
        matrix = self.project_data.doe_matrix
        if not matrix:
            QMessageBox.warning(self, "无数据", "后端中未找到DOE点列，请先进入【方案配置】生成DOE实验方案！")
            return
            
        method = self.project_data.design_method
        self.title_label.setText(f"目前方案：{method} | 实验次数：{len(matrix)}")
        
        # 提取表头
        columns = list(matrix[0].keys())
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(matrix))
        self.table.setHorizontalHeaderLabels(columns)
        
        factors = [f.name for f in self.project_data.factors if f.name]
        responses = [r.name for r in self.project_data.responses if r.name]

        for r_idx, row_dict in enumerate(matrix):
            for c_idx, col_name in enumerate(columns):
                val = row_dict[col_name]
                item = QTableWidgetItem(str(val))
                
                # 如果是因子列或者是ID，设为只读并标背色；
                # 如果是响应列，允许编辑以便填入结果
                if col_name in factors or col_name == "Run_ID":
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    item.setBackground(QtGui.QColor("#e9ecef"))
                elif col_name in responses:
                    item.setBackground(QtGui.QColor("#ffffff"))

                self.table.setItem(r_idx, c_idx, item)
                
        self.analyze_btn.setEnabled(True)

    def save_results(self):
        """将用户在表格中手动敲入的响应结果写回 ProjectData，打通与下一步（建模优化）的通道"""
        if self.table.rowCount() == 0:
            return
            
        columns = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        responses = [r.name for r in self.project_data.responses if r.name]
        
        for r_idx in range(self.table.rowCount()):
            row_dict = self.project_data.doe_matrix[r_idx]
            for c_idx, col_name in enumerate(columns):
                # 仅更新响应结果列
                if col_name in responses:
                    item = self.table.item(r_idx, c_idx)
                    val_str = item.text() if item else ""
                    try:
                        row_dict[col_name] = float(val_str) if val_str.strip() else ""
                    except ValueError:
                        QMessageBox.warning(self, "转换错误", f"第 {r_idx+1} 行 {col_name} 列填入的必须是数字！")
                        return
                        
        print("======== 实验结果数据已落盘 ========")
        for r in self.project_data.doe_matrix[:3]: # 仅打印前三个
            print(r)
        print("... (等) ... 数据已可以服务于代理建模/响应曲面拟合！")
        QMessageBox.information(self, "成功", "实验数据保存成功，已同步至后端底座，可进入下一阶段！")

    def simple_analysis(self):
        # 简单占位：比如检测是否有空数据或计算响应的最大值最小值等
        responses = [r.name for r in self.project_data.responses if r.name]
        msg = "数据填写概览：\n"
        for r_name in responses:
            valid_cnt = 0
            for row in self.project_data.doe_matrix:
                if row.get(r_name) != "":
                    valid_cnt += 1
            msg += f"- 指标 [{r_name}]: 已填入 {valid_cnt} / {len(self.project_data.doe_matrix)} 项\n"
        QMessageBox.information(self, "数据完整性自检", msg)

    def export_to_csv(self):
        if not self.project_data.doe_matrix:
            QMessageBox.warning(self, "警告", "没有可导出的数据！")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "导出CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.project_data.doe_matrix[0].keys())
                    writer.writeheader()
                    writer.writerows(self.project_data.doe_matrix)
                QMessageBox.information(self, "成功", "DOE表导出成功！你可以将该表发给仿真脚本加载执行。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def import_from_csv(self):
        """导入外部算好的 CSV (如流体仿真自动生成的带结果列的文件)"""
        file_path, _ = QFileDialog.getOpenFileName(self, "导入测试结果 CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    imported_data = list(reader)
                    
                # 简单校验字段是否匹配
                if imported_data and "Run_ID" not in imported_data[0]:
                    QMessageBox.warning(self, "格式错误", "CSV 缺少必须的 Run_ID 列！")
                    return
                    
                # 覆盖刷新内部模型
                self.project_data.doe_matrix = imported_data
                self.load_doe_matrix() 
                QMessageBox.information(self, "成功", "外部实验数据导入成功并已填入表格！请点击下方保存确认接轨后端！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")