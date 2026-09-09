"""设计优化页面：把方案配置生成的设计在填入响应数据后做后端分析并展示。

支持的结果视图：
- 方案设计摘要与设计诊断；
- 两水平筛选（± 对比主效应 / 方差贡献）；
- 响应曲面（二次模型拟合、拟合优度、优选设置）；
- 田口稳健设计（均值响应表、S/N 表、稳健最优水平组合）；
- 代理模型训练（Kriging / SVR / BPANN 的误差与耗时）。
"""
from __future__ import annotations

import math
import statistics

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QGroupBox, QTextEdit, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpinBox, QDoubleSpinBox, QFrame
)
from PyQt6.QtCore import pyqtSignal

import doe_engine as engine
import optimizer_engine as oe
from models import ProjectData

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class OptimizationPage(QWidget):
    optimization_finished = pyqtSignal()

    VIEWS = [
        "方案设计摘要",
        "筛选设计结果",
        "响应曲面结果",
        "田口稳健结果",
        "代理模型训练结果",
    ]

    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 顶部：状态 + 主执行按钮
        top = QHBoxLayout()
        self.status_label = QLabel("尚未运行稳定性设计与鲁棒优化")
        self.status_label.setStyleSheet("color:#b45309; font-weight:bold;")
        top.addWidget(self.status_label)
        top.addStretch()
        self.run_all_btn = QPushButton("执行稳定性设计与鲁棒优化")
        self.run_all_btn.setMinimumWidth(260)
        self.run_all_btn.setStyleSheet("font-weight:bold;background-color:#1d4ed8;color:white;")
        self.run_all_btn.clicked.connect(self.run_workbench)
        top.addWidget(self.run_all_btn)
        main_layout.addLayout(top)

        self.tabs = QTabWidget()
        self.tab_overview = QWidget(); self._build_overview_tab()
        self.tab_robust = QWidget(); self._build_robust_tab()
        self.tab_data = QWidget(); self._build_data_tab()
        self.tab_signif = QWidget(); self._build_signif_tab()
        self.tab_cv = QWidget(); self._build_cv_tab()
        self.tab_param = QWidget(); self._build_param_tab()
        self.tab_tol = QWidget(); self._build_tol_tab()
        self.tab_model = QWidget(); self._build_model_tab()
        self.tab_tools = QWidget(); self._build_tools_tab()
        self.tabs.addTab(self.tab_overview, "方案/结果汇总")
        self.tabs.addTab(self.tab_robust, "鲁棒优化设计")
        self.tabs.addTab(self.tab_data, "整体数据分析")
        self.tabs.addTab(self.tab_signif, "显著性/LOF")
        self.tabs.addTab(self.tab_cv, "交互验证/拟合残差")
        self.tabs.addTab(self.tab_param, "参数设计与取值")
        self.tabs.addTab(self.tab_tol, "容差贡献")
        self.tabs.addTab(self.tab_model, "模型提取")
        self.tabs.addTab(self.tab_tools, "优化工具集")
        main_layout.addWidget(self.tabs)

        self.setLayout(main_layout)
        self._oe_context = None

    def _build_overview_tab(self):
        layout = QVBoxLayout(self.tab_overview)
        head = QHBoxLayout()
        head.addWidget(QLabel("结果视图："))
        self.view_combo = QComboBox()
        self.view_combo.addItems(self.VIEWS)
        head.addWidget(self.view_combo)
        head.addStretch()
        self.run_btn = QPushButton("更新方案汇总")
        self.run_btn.clicked.connect(self.run_optimization)
        head.addWidget(self.run_btn)
        layout.addLayout(head)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family:Consolas, 'Microsoft YaHei'; font-size:13px;")
        layout.addWidget(self.result_text)
        self.view_combo.currentTextChanged.connect(self._refresh_view)
        self._refresh_view(self.view_combo.currentText())

    def _build_robust_tab(self):
        layout = QVBoxLayout(self.tab_robust)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("优化模式："))
        self.robust_mode_combo = QComboBox()
        self.robust_mode_combo.addItems(["多目标 NSGA-II", "加权单目标", "约束型"])
        controls.addWidget(self.robust_mode_combo)
        controls.addSpacing(16)
        controls.addWidget(QLabel("种群数："))
        self.pop_spin = QSpinBox(); self.pop_spin.setRange(20, 300); self.pop_spin.setValue(80)
        controls.addWidget(self.pop_spin)
        controls.addSpacing(8)
        controls.addWidget(QLabel("代数："))
        self.gen_spin = QSpinBox(); self.gen_spin.setRange(30, 1000); self.gen_spin.setValue(160)
        controls.addWidget(self.gen_spin)
        controls.addStretch()
        layout.addLayout(controls)

        controls2 = QHBoxLayout()
        controls2.addWidget(QLabel("均值权重 λ："))
        self.weight_spin = QDoubleSpinBox(); self.weight_spin.setRange(0.0, 1.0)
        self.weight_spin.setSingleStep(0.05); self.weight_spin.setValue(0.5)
        controls2.addWidget(self.weight_spin)
        controls2.addSpacing(16)
        controls2.addWidget(QLabel("约束边界退让 k："))
        self.k_constraint_spin = QDoubleSpinBox(); self.k_constraint_spin.setRange(1.0, 10.0)
        self.k_constraint_spin.setSingleStep(0.5); self.k_constraint_spin.setValue(6.0)
        controls2.addWidget(self.k_constraint_spin)
        controls2.addSpacing(8)
        controls2.addWidget(QLabel("输入边界退让 k："))
        self.k_design_spin = QDoubleSpinBox(); self.k_design_spin.setRange(1.0, 10.0)
        self.k_design_spin.setSingleStep(0.5); self.k_design_spin.setValue(6.0)
        controls2.addWidget(self.k_design_spin)
        controls2.addStretch()
        layout.addLayout(controls2)

        self.robust_mode_combo.currentTextChanged.connect(self._on_robust_mode)
        self._on_robust_mode(self.robust_mode_combo.currentText())
        self.pareto_fig = Figure(figsize=(5, 3.6), dpi=100)
        self.pareto_canvas = FigureCanvas(self.pareto_fig)
        layout.addWidget(self.pareto_canvas)
        self.robust_text = QLabel()
        self.robust_text.setWordWrap(True)
        self.robust_text.setStyleSheet("background:#f7f9fb;border:1px solid #d9e0e7;padding:8px;")
        layout.addWidget(self.robust_text)

    def _on_robust_mode(self, text):
        self.weight_spin.setEnabled("加权" in text)
        self.k_constraint_spin.setEnabled("约束" in text or True)

    def _new_text(self):
        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("font-family:Consolas,'Microsoft YaHei';font-size:13px;")
        return text

    def _build_data_tab(self):
        layout = QVBoxLayout(self.tab_data)
        self.data_overall_label = QLabel()
        self.data_overall_label.setWordWrap(True)
        layout.addWidget(self.data_overall_label)
        self.level_table = QTableWidget(0, 9)
        self.level_table.setHorizontalHeaderLabels(
            ["因子", "水平", "响应", "均值", "方差", "极差", "样本数", "F比", "p值"])
        self.level_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.level_table)

    def _build_signif_tab(self):
        layout = QVBoxLayout(self.tab_signif)
        self.signif_text = self._new_text()
        layout.addWidget(self.signif_text)

    def _build_cv_tab(self):
        layout = QVBoxLayout(self.tab_cv)
        self.cv_text = self._new_text()
        layout.addWidget(self.cv_text)
        self.resid_fig = Figure(figsize=(5, 3.4), dpi=100)
        self.resid_canvas = FigureCanvas(self.resid_fig)
        layout.addWidget(self.resid_canvas)

    def _build_param_tab(self):
        layout = QVBoxLayout(self.tab_param)
        self.param_text = QLabel()
        self.param_text.setWordWrap(True)
        layout.addWidget(self.param_text)
        self.factor_fig = Figure(figsize=(6, 4), dpi=100)
        self.factor_canvas = FigureCanvas(self.factor_fig)
        layout.addWidget(self.factor_canvas)

    def _build_tol_tab(self):
        layout = QVBoxLayout(self.tab_tol)
        self.tol_text = QLabel()
        self.tol_text.setWordWrap(True)
        self.tol_text.setStyleSheet("background:#f7f9fb;border:1px solid #d9e0e7;padding:8px;")
        layout.addWidget(self.tol_text)

    def _build_model_tab(self):
        layout = QVBoxLayout(self.tab_model)
        self.model_text = self._new_text()
        layout.addWidget(self.model_text)

    def _build_tools_tab(self):
        layout = QVBoxLayout(self.tab_tools)
        head = QHBoxLayout()
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["最陡上升", "EVOP 演化操作", "自适应 AOFAT"])
        head.addWidget(self.tool_combo)
        head.addStretch()
        self.run_tool_btn = QPushButton("运行工具")
        self.run_tool_btn.clicked.connect(self._run_tool)
        head.addWidget(self.run_tool_btn)
        layout.addLayout(head)
        self.tool_text = self._new_text()
        layout.addWidget(self.tool_text)

    # ------------------------------------------------------------- 公共工具
    def _responses(self):
        return [r for r in self.project_data.responses if r.name]

    def _matrix(self):
        return self.project_data.doe_matrix or []

    def _method(self):
        return self.project_data.design_method or ""

    def _count_filled(self, response_name):
        total = 0
        for row in self._matrix():
            val = row.get(response_name, "")
            if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip()):
                total += 1
        return total

    def _has_response_data(self, response_name):
        return self._count_filled(response_name) >= 4

    # ------------------------------------------------------------- 视图调度
    def _refresh_view(self, text):
        if "方案设计摘要" in text:
            self.result_text.setPlainText(self._summary_text())
        elif "筛选" in text:
            self.result_text.setPlainText(self._screening_text())
        elif "响应曲面" in text:
            self.result_text.setPlainText(self._rsm_text())
        elif "田口" in text:
            self.result_text.setPlainText(self._taguchi_text())
        else:
            self.result_text.setPlainText(self._surrogate_text())

    # ------------------------------------------------------------- 结果文本
    def _summary_text(self):
        method = self._method()
        if not method:
            return "尚未在【方案配置】生成任何方案。"
        params = self.project_data.design_params or {}
        meta = params.get("meta") or {}
        lines = [f"【当前方案】{method}", ""]
        if meta:
            lines.append("【设计结构】")
            for key, value in meta.items():
                if key == "groups":
                    continue
                lines.append(f"  {key} = {value}")
        lines.append("")
        matrix = self._matrix()
        lines.append(f"试验次数：{len(matrix)}")
        for resp in self._responses():
            lines.append(f"响应 [{resp.name}]：已填入 {self._count_filled(resp.name)} / {len(matrix)} 行")
        lines.append("")
        missing = [r.name for r in self._responses() if not self._has_response_data(r.name)]
        if missing:
            lines.append(f"提示：响应 {missing} 数据不足，请先在【数据管理】填入/导入结果后再执行分析。")
        else:
            lines.append("响应数据齐备，可执行设计优化与后端分析。")
        return "\n".join(lines)

    def _screening_text(self):
        result = self.project_data.screening_result or {}
        rankings = result.get("factor_rankings") or []
        lines = ["【两水平筛选设计结果】"]
        if not rankings:
            lines.append("尚未生成筛选方案。")
            return "\n".join(lines)
        lines.append("")
        lines.append(result.get("diagnostic_summary", ""))
        lines.append("")
        if result.get("response_effects"):
            lines.append("【按已填响应计算的 ± 主效应】")
            for resp_name, effects in result["response_effects"].items():
                lines.append(f"  响应：{resp_name}")
                for item in effects:
                    lines.append(f"    {item['name']}: 效应 {item['effect']:+.4g}"
                                 f"（贡献 {item['share']:.1f}%）")
            lines.append("")
        lines.append("【因子影响排序】")
        for item in rankings[:10]:
            lines.append(f"  {item.get('rank')}. {item.get('name')} — {item.get('share', 0)}%")
        lines.append("")
        lines.append(result.get("conclusion", ""))
        return "\n".join(lines)

    def _rsm_text(self):
        result = self.project_data.rsm_result or {}
        if not result:
            return "尚未生成响应曲面方案。请在【方案配置】选择响应曲面设计并点击生成。"
        lines = ["【响应曲面设计结果】"]
        design = result.get("design") or {}
        if design:
            for key, value in design.items():
                if key == "groups":
                    continue
                lines.append(f"  {key} = {value}")
        lines.append("")
        fit = result.get("fit")
        if not fit:
            lines.append(result.get("message",
                                   "请在【数据管理】填入响应数据后，点击【执行后端分析并更新结果】。"))
            return "\n".join(lines)
        for resp_name, resp_fit in fit.items():
            if resp_fit.get("error"):
                lines.append(f"◆ 响应：{resp_name}：{resp_fit['error']}")
                continue
            lines.append(f"◆ 响应：{resp_name}")
            lines.append(f"  R² = {resp_fit['r2']} ｜ 调整 R² = {resp_fit['adj_r2']} ｜ "
                         f"RMSE = {resp_fit['rmse']} ｜ 模型自由度 {resp_fit['df_model']} / "
                         f"残差自由度 {resp_fit['df_residual']}")
            lines.append("  系数（按绝对值排序）：")
            coefs = sorted(resp_fit["coefficients"], key=lambda c: abs(c[1]), reverse=True)
            for name, value in coefs[:8]:
                lines.append(f"    {name} = {value:+.5g}")
            opt = resp_fit.get("optimum")
            if opt:
                lines.append(f"  模型优选设置：{opt.get('x')} → 预测 {opt.get('predicted'):.5g}")
            lines.append("")
        return "\n".join(lines)

    def _taguchi_text(self):
        result = self.project_data.taguchi_result or {}
        if not result:
            return "尚未生成田口方案。"
        lines = ["【田口稳健设计结果】"]
        lines.append(f"内表 {result.get('array_name', '')} × 外表 {result.get('outer_label', '无')}"
                     f"｜ 因子水平数 {result.get('levels', 2)}")
        lines.append(result.get("summary", ""))
        lines.append("")
        snr_rows = result.get("signal_to_noise") or []
        if snr_rows:
            lines.append("【因子-水平 S/N 响应表】")
            for item in snr_rows[:14]:
                lines.append(f"  [{item.get('response')}] {item.get('factor')}："
                             f"{item.get('detail', '')}")
            lines.append("")
            lines.append(f"【稳健最优水平组合】{result.get('best_levels', '')}")
            lines.append("")
            lines.append(result.get("recommendation", ""))
        else:
            lines.append(result.get("recommendation", ""))
            lines.append("提示：填入/导入响应数据后可自动计算均值响应表与 S/N 表。")
        return "\n".join(lines)

    def _surrogate_text(self):
        result = self.project_data.surrogate_result or {}
        cfg = self.project_data.surrogate_config or {}
        lines = ["【代理模型训练结果】"]
        if not cfg:
            lines.append("尚未配置代理模型。请在【方案配置】选择代理模型方法并打开设置窗口。")
            return "\n".join(lines)
        lines.append(f"模型：{cfg.get('model')} ｜ 样本 {cfg.get('sample_count')} 点"
                     f" ｜ 抽样方法 {cfg.get('doe_method')}")
        lines.append("")
        per_resp = result.get("responses") or {}
        if not per_resp:
            lines.append("尚未执行训练。填入响应数据后点击【执行后端分析并更新结果】。")
            return "\n".join(lines)
        for resp_name, item in per_resp.items():
            if item.get("error"):
                lines.append(f"响应 {resp_name}：{item['error']}")
                continue
            params = item.get("params") or {}
            lines.append(f"◆ 响应：{resp_name}")
            lines.append(f"  测试集平均相对误差 {item.get('mape_test_%')}% ｜ "
                         f"R² = {item.get('r2_test')} ｜ "
                         f"训练样本 {item.get('train_samples')} / "
                         f"测试样本 {item.get('test_samples')} ｜ "
                         f"耗时 {item.get('train_time_s')}s")
            lines.append(f"  模型参数：{params}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------- 后端执行
    def run_optimization(self):
        method = self._method()
        if not method:
            QMessageBox.warning(self, "无方案", "请先在【方案配置】生成试验方案。")
            return
        try:
            if "筛选" in method:
                self._run_screening()
            elif "响应曲面" in method:
                self._run_rsm()
            elif "田口" in method:
                self._run_taguchi()
            elif "代理模型" in method:
                self._run_surrogate()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "分析失败", f"后端分析出错：{exc}")
            return
        self.status_label.setText("分析完成：结果已更新")
        self._refresh_view(self.view_combo.currentText())
        QMessageBox.information(
            self, "分析完成",
            "后端分析已完成，结果已写入当前数据模型。可在上方切换结果视图，"
            "随后前往【分析报告】查看可视化。")
        self.optimization_finished.emit()

    def _factors_for_doe(self):
        return [f for f in self.project_data.factors if f.name and f.source == "设计"]

    # ------------------------------------------------------------- 筛选分析
    def _run_screening(self):
        result = dict(self.project_data.screening_result or {})
        factors = self._factors_for_doe()
        resp_effects = {}
        agg = {}
        any_real = False
        for resp in self._responses():
            effects = engine.screening_from_responses(factors, self._matrix(), resp.name)
            if effects:
                any_real = True
                total = sum(abs(v) for v in effects.values()) or 1.0
                rows = [{"name": name, "effect": value,
                         "share": round(abs(value) / total * 100, 2)}
                        for name, value in sorted(effects.items(), key=lambda kv: -abs(kv[1]))]
                resp_effects[resp.name] = rows
                for name, value in effects.items():
                    agg[name] = agg.get(name, 0.0) + abs(value)
        if any_real:
            total = sum(agg.values()) or 1.0
            ranking = [{"rank": i + 1, "name": name, "effect": value,
                        "share": round(value / total * 100, 2)}
                       for i, (name, value) in enumerate(
                           sorted(agg.items(), key=lambda kv: -kv[1]))]
            result["factor_rankings"] = ranking
            result["main_effects"] = ranking[:8]
            result["response_effects"] = resp_effects
            result["conclusion"] = "基于已填响应计算：主效应排序见上，显著因子建议进入下一轮建模。"
            self.project_data.screening_result = result

    # ------------------------------------------------------------- 响应曲面
    def _run_rsm(self):
        if not engine.HAS_NUMPY:
            QMessageBox.warning(self, "缺少依赖", "当前环境没有 numpy，无法拟合响应曲面。")
            return
        factors = self._factors_for_doe()
        names = [f.name for f in factors]
        fit = {}
        for resp in self._responses():
            if not self._has_response_data(resp.name):
                fit[resp.name] = {"error": f"响应 {resp.name} 数据不足，无法拟合。"}
                continue
            X_rows, y = [], []
            for row in self._matrix():
                try:
                    yv = float(row.get(resp.name))
                except (TypeError, ValueError):
                    continue
                try:
                    X_rows.append([float(row[n]) for n in names])
                    y.append(yv)
                except (TypeError, ValueError):
                    continue
            X = engine._np.array(X_rows)
            yy = engine._np.array(y)
            lo = engine._np.array([engine.factor_limits(f)[0] for f in factors])
            hi = engine._np.array([engine.factor_limits(f)[1] for f in factors])
            span = (hi - lo) / 2.0
            span[span == 0] = 1.0
            Xc = (X - (lo + hi) / 2.0) / span
            fitted = engine.fit_response_surface(Xc, yy)
            feature = resp.feature
            obj = "max" if feature == "望大" else ("min" if feature == "望小" else "target")
            target = ((float(resp.lower) + float(resp.upper)) / 2.0
                      if feature == "望目" else None)
            opt = engine.quadratic_optimum(fitted, [-1] * len(factors), [1] * len(factors),
                                           obj, target=target)
            if opt:
                mids = (lo + hi) / 2.0
                opt["x"] = [round(float(mid + c * half), 6)
                            for mid, half, c in zip(mids, span, opt["x"])]
            fitted["optimum"] = opt
            fit[resp.name] = fitted
        self.project_data.rsm_result["fit"] = fit

    # ------------------------------------------------------------- 田口分析
    def _snr_type(self, resp, mode):
        if mode and "按各响应" not in mode:
            return "望大" if "望大" in mode else ("望小" if "望小" in mode else "望目")
        return resp.feature

    @staticmethod
    def _snr(values, snr_type):
        vals = [float(v) for v in values]
        if not vals:
            return None
        positive = [v for v in vals if v > 0] or vals
        mean = statistics.mean(positive)
        var = statistics.variance(positive) if len(positive) > 1 else 0.0
        if snr_type == "望大":
            inv = [1.0 / v for v in positive]
            return round(10 * math.log10(1.0 / (sum(v * v for v in inv) / len(inv)) + 1e-12), 4)
        if snr_type == "望小":
            return round(-10 * math.log10(sum(v * v for v in positive) / len(positive) + 1e-12), 4)
        return round(10 * math.log10(mean * mean / (var + 1e-12) + 1e-12), 4)

    def _run_taguchi(self):
        result = dict(self.project_data.taguchi_result or {})
        matrix = self._matrix()
        groups = result.get("groups")
        if not groups or len(groups) != len(matrix):
            return
        params = self.project_data.design_params or {}
        mode = params.get("snr_mode", "按各响应目标特征自动")
        level_maps = result.get("factor_levels") or {}
        has_outer = result.get("has_outer", False)
        inner_count = result.get("inner_runs", 1)
        per_resp_level_snr = {}

        for resp in self._responses():
            snr_type = self._snr_type(resp, mode)
            inner_values = {i: [] for i in range(inner_count)}
            filled = True
            for row, g in zip(matrix, groups):
                try:
                    y = float(row.get(resp.name))
                except (TypeError, ValueError):
                    filled = False
                    break
                inner_values[int(g["inner"])].append(y)
            if not filled:
                continue
            metrics = {}
            for i, vals in inner_values.items():
                if vals:
                    metrics[i] = (self._snr(vals, snr_type) if has_outer
                                  else round(sum(vals) / len(vals), 4))
            resp_table = {}
            for factor_name, levels in level_maps.items():
                buckets = {lv: [] for lv in range(len(levels))}
                for row, g in zip(matrix, groups):
                    try:
                        value = float(row.get(factor_name))
                    except (TypeError, ValueError):
                        continue
                    idx = min(range(len(levels)), key=lambda k: abs(levels[k] - value))
                    if g["inner"] in metrics:
                        buckets[idx].append(metrics[g["inner"]])
                resp_table[factor_name] = [
                    {"level_value": levels[lv], "metric": round(sum(v) / len(v), 4)}
                    for lv, v in buckets.items() if v
                ]
            per_resp_level_snr[resp.name] = resp_table

        if not per_resp_level_snr:
            return
        rows, best = [], {}
        for resp_name, factor_tables in per_resp_level_snr.items():
            for factor_name, table in factor_tables.items():
                best_entry = max(table, key=lambda kv: kv["metric"])
                best.setdefault(factor_name, best_entry["level_value"])
                detail = "；".join(f"水平{lv['level_value']}→{lv['metric']}" for lv in table)
                rows.append({"factor": factor_name, "response": resp_name,
                             "detail": detail, "snr": best_entry["metric"]})
        recommendation = "稳健最优水平组合：" + "、".join(
            f"{k}@{v}" for k, v in best.items())
        result["signal_to_noise"] = rows
        result["best_levels"] = best
        result["recommendation"] = recommendation + "。建议按该组合执行确认试验以验证稳健性。"
        self.project_data.taguchi_result = result

    # ------------------------------------------------------------- 代理模型
    def _run_surrogate(self):
        cfg = self.project_data.surrogate_config or {}
        if not cfg:
            raise ValueError("尚未配置代理模型，请先到【方案配置】打开设置窗口。")
        factors = [f for f in self._factors_for_doe() if f.name]
        names = [f.name for f in factors]
        bounds = cfg.get("bounds") or {}
        lo = {}
        hi = {}
        for name, f in zip(names, factors):
            default_lo, default_hi = engine.factor_limits(f)
            b = bounds.get(name) or [default_lo, default_hi]
            lo[name], hi[name] = b[0], b[1]
        per_resp = {}
        for resp in self._responses():
            X_rows, y = [], []
            for row in self._matrix():
                try:
                    yv = float(row.get(resp.name))
                except (TypeError, ValueError):
                    continue
                try:
                    x = [(float(row[n]) - lo[n]) / max(hi[n] - lo[n], 1e-12) for n in names]
                    X_rows.append(x)
                    y.append(yv)
                except (TypeError, ValueError):
                    continue
            if len(y) < 6:
                per_resp[resp.name] = {
                    "error": f"响应 {resp.name} 有效样本 {len(y)} < 6，无法训练。"}
                continue
            per_resp[resp.name] = engine.train_surrogate(
                X_rows, y, model=cfg.get("model"),
                model_params=cfg.get("model_params") or {})
        self.project_data.surrogate_result = {"responses": per_resp}

    # ============================================================ 工作台
    def run_workbench(self):
        context, errors = oe.build_models(self.project_data)
        self._oe_context = context
        if not context or not context["models"]:
            QMessageBox.warning(self, "无法分析",
                                "\n".join(errors or ["缺少可建模的响应数据。"]))
            self.status_label.setText("分析失败：缺少响应数据（请先在数据管理填入响应）。")
            return

        mode = {"多目标 NSGA-II": "multi", "加权单目标": "weighted",
                "约束型": "constraint"}[self.robust_mode_combo.currentText()]
        pop = self.pop_spin.value()
        gen = self.gen_spin.value()
        weight = self.weight_spin.value()
        kd = self.k_design_spin.value()
        kc = self.k_constraint_spin.value()
        self.status_label.setText("正在执行鲁棒优化与稳定性分析…（可能稍候）")
        robust = oe.robust_optimize(context, mode=mode, pop=pop, gen=gen,
                                    weight=weight, k_design=kd,
                                    k_constraint=kc, seed=7)
        self._robust_res = robust
        self.robust_text.setText(robust.get("text", robust.get("error", "")))
        self._plot_pareto(robust)

        self._refresh_view(self.view_combo.currentText())
        self.data_overall_label.setText(oe.describe_data(self.project_data))
        ls = oe.level_stats(self.project_data)
        self._fill_level_table(ls)
        self.signif_text.setPlainText(oe.anova_and_lof(context, self.project_data))
        cv_text = oe.cross_validation(context)
        resid_text, resid_data = oe.residual_analysis(context)
        self._resid_data = resid_data
        self.cv_text.setPlainText(cv_text + "\n\n" + resid_text)
        self._plot_residual(resid_data)
        param_res = oe.parameter_design(context, robust, kd)
        self._param_res = param_res
        self.param_text.setText(param_res.get("text", ""))
        self._plot_factor(param_res.get("factor_plots", []))
        tol_text, _tol = oe.tolerance_contribution(context, kd)
        self.tol_text.setText(tol_text)
        self.model_text.setPlainText(oe.model_extraction(context))

        self.status_label.setText("分析完成：全部工作台已更新")
        QMessageBox.information(
            self, "稳定性设计分析完成",
            "已执行鲁棒优化与全套稳定性分析。可切换标签页查看帕累托/Knee点、推荐取值、"
            "容差贡献与模型方程，或使用优化工具集做更深入的取值探索。")
        self.optimization_finished.emit()

    def _fill_level_table(self, ls):
        rows = ls.get("rows", [])
        self.level_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r["factor"], f"{r['level']:.4g}", r["response"],
                    f"{r['mean']:.4g}", f"{r['var']:.4g}", f"{r['range']:.4g}",
                    str(r["n"]), f"{r['F']:.4g}", f"{r['p']:.4g}"]
            for j, v in enumerate(vals):
                self.level_table.setItem(i, j, QTableWidgetItem(v))

    def _plot_pareto(self, robust):
        self.pareto_fig.clear()
        ax = self.pareto_fig.add_subplot(111)
        scatter = robust.get("front_scatter") or []
        if scatter:
            xs = [p[0] for p in scatter]
            ys = [p[1] for p in scatter]
            ax.scatter(xs, ys, s=18, alpha=0.6, color="#17a2b8", label="进化解")
            # 前沿曲线
            front = robust.get("front") or []
            if front:
                fx = [p["obj"][0] for p in front]
                fy = [p["obj"][1] for p in front if len(p["obj"]) > 1]
                if len(fy) == len(fx):
                    ax.plot(fx, fy, "-o", color="#1d4ed8", lw=1.6,
                            label="帕累托前沿")
            knee = robust.get("knee")
            if knee and len(knee["obj"]) > 1:
                ax.scatter([knee["obj"][0]], [knee["obj"][1]], s=120,
                           marker="*", color="#f97316", label="Knee Point")
            best = robust.get("best")
            if best and len(best["obj"]) > 1:
                ax.scatter([best["obj"][0]], [best["obj"][1]], s=60,
                           marker="D", color="#16a34a", label="推荐解")
            ax.set_xlabel("目标1：1-均值望性（越小越好）")
            ax.set_ylabel("目标2：归一化稳定性 σ（越小越好）")
            ax.set_title("鲁棒优化 帕累托前沿")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "暂无帕累托数据", ha="center", va="center")
        self.pareto_canvas.draw()

    def _plot_residual(self, resid_data):
        self.resid_fig.clear()
        ax = self.resid_fig.add_subplot(111)
        if resid_data:
            for name, d in resid_data.items():
                ax.scatter(d["predicted"], d["residual"], s=16, alpha=0.6,
                           label=name)
            ax.axhline(0, color="gray", ls="--")
            ax.set_xlabel("模型预测值")
            ax.set_ylabel("残差")
            ax.set_title("残差 vs 预测")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "暂无残差", ha="center", va="center")
        self.resid_canvas.draw()

    def _plot_factor(self, plots):
        self.factor_fig.clear()
        n = len(plots)
        if n == 0:
            ax = self.factor_fig.add_subplot(111)
            ax.text(0.5, 0.5, "暂无因子图", ha="center", va="center")
            self.factor_canvas.draw()
            return
        cols = min(2, n)
        rows = math.ceil(n / cols)
        for i, p in enumerate(plots):
            ax = self.factor_fig.add_subplot(rows, cols, i + 1)
            ax.plot(p["x"], p["mean"], "-o", color="#1d4ed8", label="均值望性")
            ax.plot(p["x"], p["std"], "-s", color="#dc2626", label="σ 归一")
            ax.set_title(p["factor"])
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
        self.factor_fig.tight_layout()
        self.factor_canvas.draw()

    def _run_tool(self):
        if not self._oe_context:
            QMessageBox.warning(self, "无模型", "请先点击【执行稳定性设计与鲁棒优化】。")
            return
        tool = self.tool_combo.currentText()
        kd = self.k_design_spin.value()
        start = (self._param_res or {}).get("recommended")
        if not start:
            d_inputs = oe.design_inputs(self._oe_context)
            start = {f.name: sum(de.factor_limits(f)) / 2.0 for f in d_inputs}
        if "最陡上升" in tool:
            text, _path = oe.steepest_ascent(self._oe_context, start, kd)
        elif "EVOP" in tool:
            text = oe.evop_cycle(self._oe_context, start, kd)
        else:
            text, _path = oe.aofat_search(self._oe_context, start, kd)
        self.tool_text.setPlainText(text)
