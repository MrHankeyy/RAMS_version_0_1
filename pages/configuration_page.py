"""方案配置页面：实验设计类型选择、参数细化与代理模型设置入口。"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QGroupBox, QTextEdit, QSpinBox,
    QDoubleSpinBox, QMessageBox
)

import doe_engine as engine
from models import ProjectData
from pages.surrogate_dialogs import SurrogateDialog, MODEL_TITLES


class ConfigurationPage(QWidget):
    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.surrogate_config = {}
        self.init_ui()

    # ------------------------------------------------------------------ 布局
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        info_group = QGroupBox("当前业务模型配置")
        info_layout = QVBoxLayout()
        self.info_label = QLabel()
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        design_group = QGroupBox("方案与设计方法选择")
        design_layout = QVBoxLayout()
        design_layout.addWidget(QLabel("核心设计方法："))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "两水平筛选设计（全因子/部分析因/PB）",
            "响应曲面设计（CCD/Box-Behnken）",
            "田口稳健设计（内表×外表）",
            "代理模型建模 · Kriging",
            "代理模型建模 · SVR",
            "代理模型建模 · BP 神经网络",
        ])
        design_layout.addWidget(self.method_combo)

        self.dynamic_params_widget = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_params_widget)
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)

        self.screening_widget = self._build_screening_widget()
        self.rsm_widget = self._build_rsm_widget()
        self.taguchi_widget = self._build_taguchi_widget()
        self.surrogate_widget = self._build_surrogate_widget()

        for w in (self.screening_widget, self.rsm_widget,
                  self.taguchi_widget, self.surrogate_widget):
            self.dynamic_layout.addWidget(w)
        design_layout.addWidget(self.dynamic_params_widget)

        self.preview_label = QLabel("尚未估算试验次数")
        self.preview_label.setStyleSheet("color:#1d4ed8;font-weight:bold;")
        design_layout.addWidget(self.preview_label)

        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        self.desc_text.setMaximumHeight(110)
        design_layout.addWidget(self.desc_text)

        design_group.setLayout(design_layout)
        main_layout.addWidget(design_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.refresh_btn = QPushButton("刷新模型数据(从业务建模获取)")
        self.refresh_btn.clicked.connect(self.update_info_display)
        self.gen_btn = QPushButton("生成方案集")
        self.gen_btn.setMinimumWidth(150)
        self.gen_btn.setStyleSheet("font-weight:bold;background-color:#1d4ed8;color:white;")
        self.gen_btn.clicked.connect(self.on_generate_clicked)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.gen_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)
        self._connect_preview_signals()
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.update_info_display()
        self._on_method_changed(self.method_combo.currentText())

    # ------------------------------------------------------------- 控件构建
    def _build_screening_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("设计族："))
        self.screen_kind_combo = QComboBox()
        self.screen_kind_combo.addItems([
            "全因子 2^k（小因子数）",
            "部分析因 1/2 分数",
            "部分析因 1/4 分数",
            "Plackett-Burman（12/20 次）",
        ])
        row1.addWidget(self.screen_kind_combo)
        row1.addSpacing(24)
        row1.addWidget(QLabel("报告重点主效应数："))
        self.f_topn_spin = QSpinBox()
        self.f_topn_spin.setRange(1, 10)
        self.f_topn_spin.setValue(5)
        row1.addWidget(self.f_topn_spin)
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("中心点补充数："))
        self.f_center_spin = QSpinBox()
        self.f_center_spin.setRange(0, 10)
        self.f_center_spin.setValue(2)
        row2.addWidget(self.f_center_spin)
        row2.addSpacing(24)
        row2.addWidget(QLabel("整表重复次数："))
        self.f_repeat_spin = QSpinBox()
        self.f_repeat_spin.setRange(1, 5)
        self.f_repeat_spin.setValue(1)
        row2.addWidget(self.f_repeat_spin)
        row2.addSpacing(24)
        row2.addWidget(QLabel("试验顺序："))
        self.f_order_combo = QComboBox()
        self.f_order_combo.addItems(["随机", "保持相同"])
        row2.addWidget(self.f_order_combo)
        row2.addSpacing(24)
        row2.addWidget(QLabel("随机种子："))
        self.f_seed_spin = QSpinBox()
        self.f_seed_spin.setRange(0, 99999)
        self.f_seed_spin.setValue(123)
        row2.addWidget(self.f_seed_spin)
        row2.addStretch()
        layout.addLayout(row2)

        self.screen_center_hint = QLabel(
            "中心点用于检验曲率并估计纯误差；PB 设计主效应与双因子交互存在部分混杂。")
        self.screen_center_hint.setStyleSheet("color:#666;")
        layout.addWidget(self.screen_center_hint)
        return widget

    def _build_rsm_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("曲面设计类型："))
        self.rsm_type_combo = QComboBox()
        self.rsm_type_combo.addItems([
            "中心复合设计 (CCD)",
            "Box-Behnken 设计 (BBD)",
        ])
        row1.addWidget(self.rsm_type_combo)
        row1.addSpacing(24)
        row1.addWidget(QLabel("CCD 轴值（α）："))
        self.rsm_alpha_combo = QComboBox()
        self.rsm_alpha_combo.addItems([
            "可旋转（α=2^(k/4)，推荐）",
            "位于表面（面心，α=1）",
            "用户指定 α",
        ])
        row1.addWidget(self.rsm_alpha_combo)
        self.rsm_alpha_spin = QDoubleSpinBox()
        self.rsm_alpha_spin.setRange(0.5, 5.0)
        self.rsm_alpha_spin.setDecimals(3)
        self.rsm_alpha_spin.setValue(1.414)
        self.rsm_alpha_spin.setEnabled(False)
        row1.addWidget(self.rsm_alpha_spin)
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("中心点数量："))
        self.rsm_center_spin = QSpinBox()
        self.rsm_center_spin.setRange(0, 20)
        self.rsm_center_spin.setValue(3)
        row2.addWidget(self.rsm_center_spin)
        row2.addSpacing(24)
        row2.addWidget(QLabel("整表重复次数："))
        self.rsm_repeat_spin = QSpinBox()
        self.rsm_repeat_spin.setRange(1, 5)
        self.rsm_repeat_spin.setValue(1)
        row2.addWidget(self.rsm_repeat_spin)
        row2.addSpacing(24)
        row2.addWidget(QLabel("试验顺序："))
        self.rsm_order_combo = QComboBox()
        self.rsm_order_combo.addItems(["随机", "保持相同"])
        row2.addWidget(self.rsm_order_combo)
        row2.addSpacing(24)
        row2.addWidget(QLabel("随机种子："))
        self.rsm_seed_spin = QSpinBox()
        self.rsm_seed_spin.setRange(0, 99999)
        self.rsm_seed_spin.setValue(456)
        row2.addWidget(self.rsm_seed_spin)
        row2.addStretch()
        layout.addLayout(row2)

        self.rsm_hint = QLabel(
            "CCD＝立方点＋轴点＋中心点；可旋转 α 使预测方差只依赖到中心的距离。"
            "BBD 每个因子仅 3 个水平、不含立方体顶点。")
        self.rsm_hint.setStyleSheet("color:#666;")
        layout.addWidget(self.rsm_hint)

        self.rsm_type_combo.currentTextChanged.connect(self._on_rsm_type)
        self.rsm_alpha_combo.currentTextChanged.connect(self._on_alpha_mode)
        return widget

    def _build_taguchi_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("因子水平数："))
        self.t_levels_combo = QComboBox()
        self.t_levels_combo.addItems(["2 水平", "3 水平"])
        row1.addWidget(self.t_levels_combo)
        row1.addSpacing(20)
        row1.addWidget(QLabel("内表（控制因子）："))
        self.t_inner_combo = QComboBox()
        self.t_inner_combo.addItems(["自动推荐", "L4 (2^3)", "L8 (2^7)", "L9 (3^4)",
                                     "L12 (2^11)", "L16 (2^15)", "L27 (3^13)"])
        row1.addWidget(self.t_inner_combo)
        row1.addSpacing(20)
        row1.addWidget(QLabel("外表（噪声因子）："))
        self.t_outer_combo = QComboBox()
        self.t_outer_combo.addItems(["无（仅内表）", "自动推荐", "L4 (2^3)", "L8 (2^7)",
                                     "L9 (3^4)", "L12 (2^11)", "L16 (2^15)"])
        row1.addWidget(self.t_outer_combo)
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("信噪比目标："))
        self.t_snr_combo = QComboBox()
        self.t_snr_combo.addItems(["按各响应目标特征自动", "望大（越大越好）",
                                   "望小（越小越好）", "望目（名义最好）"])
        row2.addWidget(self.t_snr_combo)
        row2.addSpacing(24)
        row2.addWidget(QLabel("随机种子："))
        self.t_seed_spin = QSpinBox()
        self.t_seed_spin.setRange(0, 99999)
        self.t_seed_spin.setValue(789)
        row2.addWidget(self.t_seed_spin)
        row2.addStretch()
        layout.addLayout(row2)

        self.t_hint = QLabel(
            "内表安排控制因子（设计因子）、外表安排噪声因子（环境因子）；"
            "完整试验＝内表×外表叉积。环境因子列将自动填充外表水平。")
        self.t_hint.setStyleSheet("color:#666;")
        layout.addWidget(self.t_hint)

        self.t_levels_combo.currentTextChanged.connect(self._on_taguchi_levels)
        return widget

    def _build_surrogate_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.surrogate_info_label = QLabel()
        layout.addWidget(self.surrogate_info_label)
        row = QHBoxLayout()
        self.surrogate_open_btn = QPushButton("打开建模设置窗口…")
        self.surrogate_open_btn.setStyleSheet("background-color:#2563eb;color:white;")
        self.surrogate_open_btn.clicked.connect(self._open_surrogate_dialog)
        row.addWidget(self.surrogate_open_btn)
        self.surrogate_status = QLabel("尚未设置模型参数")
        self.surrogate_status.setStyleSheet("color:#b45309;")
        row.addWidget(self.surrogate_status)
        row.addStretch()
        layout.addLayout(row)

        self.surrogate_hint = QLabel(
            "设置窗口包含两部分：① 试验设计（变量维度/上下界、样本点数、抽样方法）；"
            "② 模型参数（相关/回归函数、核函数与 C/g 搜索、网络结构与寻优等）。")
        self.surrogate_hint.setStyleSheet("color:#666;")
        layout.addWidget(self.surrogate_hint)
        return widget

    # ------------------------------------------------------------- 联动刷新
    def _connect_preview_signals(self):
        for box in (self.method_combo, self.screen_kind_combo, self.rsm_type_combo,
                    self.rsm_alpha_combo, self.t_levels_combo, self.t_inner_combo,
                    self.t_outer_combo, self.t_snr_combo):
            box.currentTextChanged.connect(self._refresh_preview)
        for spin in (self.f_topn_spin, self.f_center_spin, self.f_repeat_spin,
                     self.rsm_center_spin, self.rsm_repeat_spin, self.rsm_alpha_spin):
            spin.valueChanged.connect(self._refresh_preview)

    def _on_method_changed(self, text):
        self.screening_widget.setVisible("筛选" in text)
        self.rsm_widget.setVisible("响应曲面" in text)
        self.taguchi_widget.setVisible("田口" in text)
        self.surrogate_widget.setVisible("代理模型" in text)
        self._update_surrogate_labels(text)
        self._set_description(text)
        self._refresh_preview()

    def _set_description(self, method):
        if "筛选" in method:
            desc = ("筛选设计遵循“效应稀疏 + 效应等级”原则，用尽量少的试验识别活跃主效应。"
                    "本模块支持全因子、1/2 与 1/4 部分析因以及 Plackett-Burman；"
                    "输出包含试验结构、主效应排序、设计诊断（正交性/分辨率/混杂说明）。")
        elif "响应曲面" in method:
            desc = ("二阶响应曲面模型（含平方项与两两交互）由 CCD 或 Box-Behnken 支撑。"
                    "CCD 可旋转 α、面心 α 或自定义 α；BBD 将每个因子控制在 3 个水平，避免顶点极端工况。"
                    "后续在【设计优化】拟合二次模型并给出优选设置。")
        elif "田口" in method:
            desc = ("田口稳健设计以正交表安排控制因子（内表）与噪声因子（外表），"
                    "通过内×外叉积试验估计均值与波动，并按信噪比（S/N）挑选稳健水平组合。"
                    "业务建模中的环境因子将自动作为噪声因子。")
        elif "代理模型" in method:
            desc = ("代理模型面向高成本仿真黑盒：先按试验设计生成样本点（响应留空，"
                    "在【数据管理】填入或导入结果），再由所选模型训练并评估。"
                    "三种模型均可通过独立窗口设置试验设计与模型参数。")
        self.desc_text.setPlainText(desc)

    def _on_rsm_type(self, text):
        self.rsm_alpha_combo.setEnabled("CCD" in text)
        self.rsm_alpha_spin.setEnabled("CCD" in text and self.rsm_alpha_combo.currentText().startswith("用户"))
        if "BBD" in text:
            self.rsm_center_spin.setValue(3)

    def _on_alpha_mode(self, text):
        self.rsm_alpha_spin.setEnabled(text.startswith("用户"))

    def _on_taguchi_levels(self, text):
        levels = "2 水平" if "2" in text else "3 水平"
        inner_items = {
            "2 水平": ["自动推荐", "L4 (2^3)", "L8 (2^7)", "L12 (2^11)", "L16 (2^15)"],
            "3 水平": ["自动推荐", "L9 (3^4)", "L27 (3^13)"],
        }
        outer_items = {
            "2 水平": ["无（仅内表）", "自动推荐", "L4 (2^3)", "L8 (2^7)", "L12 (2^11)", "L16 (2^15)"],
            "3 水平": ["无（仅内表）", "自动推荐", "L9 (3^4)", "L27 (3^13)"],
        }
        self._reset_combo(self.t_inner_combo, inner_items[levels])
        self._reset_combo(self.t_outer_combo, outer_items[levels])

    @staticmethod
    def _reset_combo(combo, items):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if current in items:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    # ------------------------------------------------------------- 因子/预览
    def _design_factors(self):
        """仅设计因子进入经典 DOE/代理抽样；环境因子仅用于田口外表。"""
        return [f for f in self.project_data.factors
                if f.name and f.source == "设计" and not (f.is_fixed and f.fixed_value is not None)]

    def _noise_factors(self):
        return [f for f in self.project_data.factors if f.name and f.source == "环境"]

    def _k(self):
        return len(self._design_factors())

    def _screen_kind_key(self):
        text = self.screen_kind_combo.currentText()
        if "1/2" in text:
            return "half"
        if "1/4" in text:
            return "quarter"
        if "Plackett" in text:
            return "pb"
        return "full"

    def _screen_kind_label(self):
        return engine.SCREENING_KINDS[self._screen_kind_key()]

    def _refresh_preview(self):
        method = self.method_combo.currentText()
        k = self._k()
        if "代理模型" not in method and k == 0:
            self.preview_label.setText("尚未在【业务建模】中配置设计因子，无法估算试验次数")
            return
        if "代理模型" in method and k == 0:
            self.preview_label.setText("代理模型抽样需要至少 1 个设计因子")
            return
        try:
            if "筛选" in method:
                rows, meta = engine.build_screening(
                    engine.simple_factors(self._design_factors()),
                    self._screen_kind_key(), seed=0,
                    centers=self.f_center_spin.value(),
                    replicates=self.f_repeat_spin.value(), randomize=False)
                n = meta["total_runs"]
                extra = f"（k={k}，{meta['kind']}，中心点 {self.f_center_spin.value()}）"
            elif "响应曲面" in method:
                rsm_type = "BBD" if "BBD" in self.rsm_type_combo.currentText() else "CCD"
                alpha = 0.0
                if rsm_type == "CCD":
                    alpha = {"可旋转": engine.rotatable_alpha(k), "位于表面": 1.0}.get(
                        self.rsm_alpha_combo.currentText().split("（")[0],
                        self.rsm_alpha_spin.value())
                rows, meta = engine.build_response_surface(
                    engine.simple_factors(self._design_factors()), rsm_type,
                    "face" if "位于表面" in self.rsm_alpha_combo.currentText()
                    else ("rotatable" if "可旋转" in self.rsm_alpha_combo.currentText() else "custom"),
                    self.rsm_alpha_spin.value(), centers=self.rsm_center_spin.value(),
                    replicates=self.rsm_repeat_spin.value(), randomize=False)
                n = meta["total_runs"]
                extra = f"（k={k}，{meta['design']}，中心点 {self.rsm_center_spin.value()}）"
            elif "田口" in method:
                ctl = engine.simple_factors(self._design_factors())
                noise = engine.simple_factors(self._noise_factors())
                levels = 2 if "2 水平" in self.t_levels_combo.currentText() else 3
                inner = self._resolve_oa(self.t_inner_combo.currentText(), len(ctl), levels)
                if noise:
                    outer = self._resolve_oa(self.t_outer_combo.currentText(), len(noise), levels)
                else:
                    outer = "无"
                rows, meta = engine.build_taguchi(ctl, noise, inner, outer, levels, seed=0)
                n = meta["total_runs"]
                extra = (f"（k={len(ctl)} 控制+{len(noise)} 噪声，内表 {meta['inner_label']}"
                         f"×外表 {meta['outer_label']}）")
            else:
                n = (self.surrogate_config.get("sample_count")
                     if self.surrogate_config else 50)
                extra = f"（k={k}，LHS/空间填充样本点）"
            self.preview_label.setText(f"预计试验次数：{n} 次 {extra}")
        except Exception as exc:
            self.preview_label.setText(f"试验次数估算不可用：{exc}")

    def _resolve_oa(self, label, k, levels):
        """把“自动推荐”解析为具体正交表标签。"""
        if label.startswith("无"):
            return "无"
        if label != "自动推荐":
            return label.replace(" (", "(")
        if k <= 0:
            return "无"
        if levels == 2:
            for cand in ("L4", "L8", "L12", "L16"):
                try:
                    engine.taguchi_arrays(cand, k)
                    return cand
                except ValueError:
                    continue
            raise ValueError("两水平因子数量超过 L16 容量（15）")
        for cand in ("L9", "L27"):
            try:
                engine.taguchi_arrays(cand, k)
                return cand
            except ValueError:
                continue
        raise ValueError("三水平因子数量超过 L27 容量（13）")

    def _update_surrogate_labels(self, method):
        if "代理模型" not in method:
            return
        key = "Kriging" if "Kriging" in method else ("SVR" if "SVR" in method else "ANN")
        k = self._k()
        self.surrogate_info_label.setText(
            f"代理模型：{MODEL_TITLES[key]} ｜ 输入变量 {k} 个（设计因子）"
            f" ｜ 输出变量 {len(self.project_data.responses)} 个（响应）")
        if self.surrogate_config and self.surrogate_config.get("model") == key:
            self.surrogate_status.setText(
                f"已保存：样本 {self.surrogate_config.get('sample_count')} 点，"
                f"{self.surrogate_config.get('doe_method')}")
        else:
            self.surrogate_status.setText("尚未设置该模型的参数")

    # ------------------------------------------------------------- 数据刷新
    def update_info_display(self):
        k = len(self._design_factors())
        noise = len(self._noise_factors())
        info = (
            f"设计因子：{k} 个 ｜ 环境（噪声）因子：{noise} 个 ｜ "
            f"响应/目标：{len(self.project_data.responses)} 个\n"
            f"当前设计方案：{self.project_data.design_method or '尚未生成'}"
        )
        self.info_label.setText(info)
        self._on_method_changed(self.method_combo.currentText())

    # ------------------------------------------------------------- 代理设置
    def _open_surrogate_dialog(self):
        method = self.method_combo.currentText()
        key = "Kriging" if "Kriging" in method else ("SVR" if "SVR" in method else "ANN")
        factors = self._design_factors()
        if not factors:
            QMessageBox.warning(self, "缺少因子",
                                "当前没有可用的设计因子。请先在【业务建模】添加设计因子并保存。")
            return
        dialog = SurrogateDialog(key, factors, self)
        if dialog.exec() == SurrogateDialog.DialogCode.Accepted:
            self.surrogate_config = dialog.collect_config()
            self._update_surrogate_labels(method)
            QMessageBox.information(
                self, "设置已保存",
                "代理模型与试验设计参数已保存，点击下方【生成方案集】即可生成样本点（响应列留空，"
                "随后在【数据管理】填入或导入响应结果）。")

    # ------------------------------------------------------------- 生成方案
    def on_generate_clicked(self):
        method = self.method_combo.currentText()
        try:
            self._generate_by_method(method)
        except ValueError as exc:
            QMessageBox.warning(self, "无法生成方案", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "生成失败", f"发生未预期错误：{exc}")
            return
        self.project_data.design_method = method
        self.info_label.setText(
            f"设计因子：{self._k()} 个 ｜ 环境因子：{len(self._noise_factors())} 个 ｜ "
            f"响应：{len(self.project_data.responses)} 个\n当前设计方案：{method}"
            f"（{len(self.project_data.doe_matrix)} 次试验）")
        QMessageBox.information(
            self, "方案已生成",
            f"已生成 {method} 试验方案，共 {len(self.project_data.doe_matrix)} 次试验。\n"
            "请进入【数据管理】查看 DOE 矩阵并填入/导入响应结果。")

    def _generate_by_method(self, method):
        ctl_factors = engine.simple_factors(self._design_factors())
        if not ctl_factors:
            raise ValueError("未发现设计因子，无法生成方案。请先在【业务建模】添加设计因子。")

        if "筛选" in method:
            rows, meta = engine.build_screening(
                ctl_factors, self._screen_kind_key(), seed=self.f_seed_spin.value(),
                centers=self.f_center_spin.value(), replicates=self.f_repeat_spin.value(),
                randomize=self.f_order_combo.currentText() == "随机")
            self._commit(rows, groups=None)
            self.project_data.design_params = {
                "doe_family": meta["kind"], "meta": meta,
                "top_n": self.f_topn_spin.value(),
            }
            self._write_screening_result(meta)
            return

        if "响应曲面" in method:
            rsm_type = "BBD" if "BBD" in self.rsm_type_combo.currentText() else "CCD"
            alpha_text = self.rsm_alpha_combo.currentText()
            if "位于表面" in alpha_text:
                alpha_mode, alpha_val = "face", 1.0
            elif "可旋转" in alpha_text:
                alpha_mode, alpha_val = "rotatable", 0.0
            else:
                alpha_mode, alpha_val = "custom", self.rsm_alpha_spin.value()
            rows, meta = engine.build_response_surface(
                ctl_factors, rsm_type, alpha_mode, alpha_val,
                centers=self.rsm_center_spin.value(),
                replicates=self.rsm_repeat_spin.value(),
                seed=self.rsm_seed_spin.value(),
                randomize=self.rsm_order_combo.currentText() == "随机")
            self._commit(rows, groups=None)
            self.project_data.design_params = {"meta": meta}
            self.project_data.rsm_result = {
                "design": meta,
                "fit": None,
                "message": "试验设计已生成；在【数据管理】填入响应后，可到【设计优化】拟合响应曲面。",
            }
            return

        if "田口" in method:
            noise_factors = engine.simple_factors(self._noise_factors())
            levels = 2 if "2 水平" in self.t_levels_combo.currentText() else 3
            inner = self._resolve_oa(self.t_inner_combo.currentText(), len(ctl_factors), levels)
            outer = self._resolve_oa(self.t_outer_combo.currentText(), len(noise_factors), levels)
            if noise_factors and outer == "无":
                raise ValueError("当前存在环境（噪声）因子，田口外表不能为“无”；"
                                "请选择外表正交表，或先在【业务建模】中移除环境因子。")
            rows, meta = engine.build_taguchi(
                ctl_factors, noise_factors, inner, outer, levels,
                seed=self.t_seed_spin.value())
            self._commit(rows, groups=meta["groups"])
            self.project_data.design_params = {
                "meta": meta, "snr_mode": self.t_snr_combo.currentText(),
                "levels": levels,
            }
            self._write_taguchi_result(meta)
            return

        # 代理模型
        if not self.surrogate_config:
            raise ValueError("尚未设置代理模型参数，请先点击【打开建模设置窗口…】。")
        expected = "Kriging" if "Kriging" in method else ("SVR" if "SVR" in method else "ANN")
        cfg = self.surrogate_config
        if cfg.get("model") != expected:
            raise ValueError(f"当前窗口中的设置属于 {cfg.get('model')}，与所选 {expected} 不一致，"
                            "请重新打开对应模型的设置窗口。")
        bounds = cfg.get("bounds") or {}
        sample_factors = []
        for f in ctl_factors:
            name, _lo, _hi, fixed = f
            lo, hi = bounds.get(name, [_lo, _hi])
            sample_factors.append((name, lo, hi, fixed))
        rows = engine.build_samples(sample_factors, cfg["sample_count"],
                                    cfg["doe_method"], seed=self.f_seed_spin.value())
        self._commit(rows, groups=None)
        self.project_data.design_params = {
            "surrogate_config": cfg,
            "sample_count": cfg["sample_count"],
            "doe_method": cfg["doe_method"],
        }
        self.project_data.surrogate_config = cfg
        return

    def _commit(self, rows, groups=None):
        """把试验行写入 ProjectData.doe_matrix（响应列留空）。"""
        responses = [r.name for r in self.project_data.responses if r.name]
        matrix = []
        for i, row in enumerate(rows):
            record = {"Run_ID": i + 1}
            record.update(row)
            if groups is not None:
                record["内表组号"] = groups[i]["inner"] + 1
                if groups[i]["outer"] is not None:
                    record["外表组号"] = groups[i]["outer"] + 1
            for r in responses:
                record[r] = ""
            matrix.append(record)
        self.project_data.doe_matrix = matrix
        print(f"[OK] DOE 矩阵生成完成：{len(matrix)} 行，列={list(matrix[0].keys())}")

    # ------------------------------------------------------------- 后端结果
    def _effect_prior(self):
        """无响应数据时，用因子变动幅度作主效应先验排序。"""
        scored = []
        for f in self._design_factors():
            lo, hi = engine.factor_limits(f)
            if f.uncertainty == "概率":
                mu = engine._num(f.param1, 0.0)
                sigma = abs(engine._num(f.param2, 1.0))
                effect = max(abs(mu), 1.0) * (1.0 + sigma)
            else:
                effect = max(abs(hi - lo), 0.1) * (1.0 + abs((lo + hi) / 2.0) * 0.2)
            scored.append({"name": f.name, "effect": effect})
        scored.sort(key=lambda item: item["effect"], reverse=True)
        return scored

    def _write_screening_result(self, meta):
        scored = self._effect_prior()
        total = sum(i["effect"] for i in scored) or 1.0
        ranking = [{
            "rank": i + 1, "name": item["name"], "effect": item["effect"],
            "share": round(item["effect"] / total * 100, 2),
        } for i, item in enumerate(scored)]
        top = ranking[: max(1, self.f_topn_spin.value())]
        names = "、".join(i["name"] for i in top)
        self.project_data.screening_result = {
            "factor_rankings": ranking,
            "main_effects": top,
            "stability_impact": [{"name": i["name"], "impact": i["share"], "rank": i["rank"]}
                                 for i in ranking],
            "diagnostic_summary": (
                f"筛选方案：{meta['kind']}，{meta['total_runs']} 次试验"
                f"（基础 {meta['base_runs']} 次 + 中心点 {meta['centers']} 个 ×"
                f"重复 {meta['replicates']}）。{meta['diagnostic']}。"
                "当前为变动幅度先验排序；填入响应数据后将由【设计优化】按 ± 对比重算真实主效应。"),
            "conclusion": f"先验排序识别出 {names} 等关键因子。",
            "plot_points": [{"name": i["name"], "value": i["effect"]} for i in ranking],
        }

    def _write_taguchi_result(self, meta):
        groups = meta.get("groups") or []
        self.project_data.taguchi_result = {
            "array_name": meta["inner_label"],
            "outer_label": meta["outer_label"],
            "inner_runs": meta["inner_runs"],
            "outer_runs": meta["outer_runs"],
            "has_outer": meta["has_outer"],
            "levels": meta["levels"],
            "factor_levels": meta["factor_levels"],
            "noise_levels": meta.get("noise_levels", {}),
            "orthogonal_array": [],
            "signal_to_noise": [],
            "best_levels": {},
            "groups": groups,
            "recommendation": (
                "内表×外表叉积试验已完成安排。在【数据管理】填入各次试验响应后，"
                "可到【设计优化】计算均值响应表、S/N 表并给出稳健最优水平组合。"),
            "summary": (f"已按内表 {meta['inner_label']}（{meta['inner_runs']} 组）×"
                        f"外表 {meta['outer_label']}（{meta['outer_runs']} 组）生成"
                        f" {meta['total_runs']} 次叉积试验。"),
        }
