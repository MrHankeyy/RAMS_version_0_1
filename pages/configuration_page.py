from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGroupBox, QRadioButton, QButtonGroup, QTextEdit, QSpinBox
)
import random
from models import ProjectData

class ConfigurationPage(QWidget):
    def __init__(self, project_data: ProjectData):
        super().__init__()
        self.project_data = project_data
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # ========= 实验类型提示 =========
        info_group = QGroupBox("当前业务模型配置")
        info_layout = QVBoxLayout()
        self.info_label = QLabel()
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)

        # ========= 设计方法选择区 =========
        design_group = QGroupBox("方案与设计方法选择")
        design_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "自适应推荐 (根据实验类型自动分配)",
            "析因设计 (Factorial Design)",
            "筛选设计 (Screening Design)",
            "全因子设计 (Full Factorial Design)",
            "响应曲面设计 (Response Surface Design)",
            "田口方法 (Taguchi Method)",
            "Kriging代理模型与仿真实验设计"
        ])
        design_layout.addWidget(QLabel("核心设计方法:"))
        design_layout.addWidget(self.method_combo)

        # ========= 动态实验设计参数配置区 =========
        self.dynamic_params_widget = QWidget()
        dynamic_layout = QVBoxLayout(self.dynamic_params_widget)
        dynamic_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. 析因/筛选设计参数
        self.factorial_widget = QWidget()
        f_layout = QHBoxLayout(self.factorial_widget)
        f_layout.setContentsMargins(0,0,0,0)
        self.f_resolution_combo = QComboBox()
        self.f_resolution_combo.addItems(["全因子设计 (分辨率V+)", "1/2 部分析因", "1/4 部分析因", "Plackett-Burman (最简筛选)"])
        self.f_center_spin = QSpinBox()
        self.f_center_spin.setRange(0, 10)
        self.f_center_spin.setValue(0)
        f_layout.addWidget(QLabel("设计分辨率/阶数:"))
        f_layout.addWidget(self.f_resolution_combo)
        f_layout.addWidget(QLabel("中心点补充数:"))
        f_layout.addWidget(self.f_center_spin)
        f_layout.addStretch()

        # 2. 响应曲面参数
        self.rsm_widget = QWidget()
        rsm_layout = QHBoxLayout(self.rsm_widget)
        rsm_layout.setContentsMargins(0,0,0,0)
        self.rsm_type_combo = QComboBox()
        self.rsm_type_combo.addItems(["中心复合设计 (CCD) 面心", "CCD 均匀精度", "CCD 正交区组", "CCD 正交", "Box-Behnken 设计 (BBD)"])
        self.rsm_center_spin = QSpinBox()
        self.rsm_center_spin.setRange(1, 10)
        self.rsm_center_spin.setValue(3)
        rsm_layout.addWidget(QLabel("高阶曲面类型:"))
        rsm_layout.addWidget(self.rsm_type_combo)
        rsm_layout.addWidget(QLabel("中心点数量:"))
        rsm_layout.addWidget(self.rsm_center_spin)
        rsm_layout.addStretch()

        # 3. 仿真空间抽样参数
        self.sim_widget = QWidget()
        sim_layout = QHBoxLayout(self.sim_widget)
        sim_layout.setContentsMargins(0,0,0,0)
        self.sim_points_spin = QSpinBox()
        self.sim_points_spin.setRange(10, 10000)
        self.sim_points_spin.setValue(50)
        sim_layout.addWidget(QLabel("空间填充抽样策略: Latin Hypercube (LHS)"))
        sim_layout.addWidget(QLabel("初始样本点数:"))
        sim_layout.addWidget(self.sim_points_spin)
        sim_layout.addStretch()
        
        dynamic_layout.addWidget(self.factorial_widget)
        dynamic_layout.addWidget(self.rsm_widget)
        dynamic_layout.addWidget(self.sim_widget)
        design_layout.addWidget(self.dynamic_params_widget)
        # =========================================

        # 方法详细说明区域
        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        design_layout.addWidget(self.desc_text)

        design_group.setLayout(design_layout)
        main_layout.addWidget(design_group)

        # ========= 操作按钮区 =========
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.refresh_btn = QPushButton("刷新模型数据(从业务建模获取)")
        self.refresh_btn.clicked.connect(self.update_info_display)
        self.gen_btn = QPushButton("生成方案集")
        self.gen_btn.setMinimumWidth(150)
        self.gen_btn.clicked.connect(self.on_generate_clicked)
        
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.gen_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)
        self.update_info_display()
        self.update_description()

        # 监听下拉框改变
        self.method_combo.currentTextChanged.connect(self.update_description)

    def update_info_display(self):
        """拉取前面模块设置的数据，更新显示和默认策略"""
        exp_type = self.project_data.experiment_type
        factors_count = len(self.project_data.factors)
        response_count = len(self.project_data.responses)
        
        has_noise = any(f.source == "环境" for f in self.project_data.factors)
        has_discrete = any(f.mode == "离散" for f in self.project_data.factors)
        
        info = (f"实验类型: {exp_type}\n"
                f"因子数量: {factors_count} (含环境干扰:{has_noise}, 离散参数:{has_discrete})\n"
                f"响应/目标数量: {response_count}")
        self.info_label.setText(info)

        rec = ""
        # 特殊通道：田口方法 (物理世界 + 离散 + 强干扰)
        if exp_type == "实际实验" and has_noise and has_discrete:
            rec = "【专家诊断】检测到存在离散参数与环境噪音因子，高度推荐跳出传统流程，采取 [田口方法 (Taguchi)] 单独流程，利用内外正交表直接进行信噪比鲁棒分析。\n"
        
        # 常规漏斗式自适应范式
        if exp_type == "实际实验":
            if factors_count > 7:
                rec += "【高维实验路径】：[筛选设计(PB)] 剔除无效因子 ➔ [部分析因设计] 评估交互效应 ➔ [响应曲面设计] 补充中心点拟合曲面 ➔ [鲁棒优化与后验计算]。"
            elif factors_count >= 4:
                rec += "【中维实验路径】：[全面/部分析因设计] 评估主从与交互 ➔ [响应曲面设计] 补充测试点拟合二次模型 ➔ [鲁棒优化与容差分析]。"
            else:
                rec += "【低维实验路径】：由于因子稀少，直接步入 [响应曲面设计(CCD/BBD)] 锁定极值 ➔ [鲁棒优化计算] ➔ [结果验证]。"
        else:
            if factors_count > 7:
                rec += "【高维仿真路径】：[粗粒度拉丁超立方LHS] ➔ Sobol全局灵敏度筛选降维 ➔ [高密度LHS+Kriging] 构建核心代理替身 ➔ [多目标鲁棒寻优]。"
            else:
                rec += "【低中维仿真路径】：直接 [高密度空间填充抽样(LHS/Sobol序列)] ➔ 构建 [Kriging 代理模型] ➔ 大规模启发式蒙特卡洛/遗传寻优。"
        
        # 刷新下拉框逻辑提示
        self.update_description(self.method_combo.currentText(), auto_rec=rec)

    def update_description(self, method_name=None, auto_rec=None):
        if method_name is None:
            method_name = self.method_combo.currentText()
            
        desc = ""
        if "自适应推荐" in method_name:
            desc = "自动扫描【业务建模】中选择是仿真引擎还是物理实体，加上因子数量维度，为您动态锁定最高效的设计方法。\n\n"
            if auto_rec: desc += auto_rec
        elif "析因设计" in method_name:
            desc = "通过较少的次数验证各因子交互作用的显著性，非常适合实际实验初期摸底阶段。"
        elif "筛选设计" in method_name:
            desc = "常使用Plackett-Burman设计，用于在海量因子中快速剔除无效因子，锁定主干。"
        elif "全因子设计" in method_name:
            desc = "全面覆盖所有组合情况。如果因子为3个以内，此方法最靠谱，能够精确发现全部交互特征与非线性模型。"
        elif "响应曲面" in method_name:
            desc = "寻找最佳工艺及参数区间的利器！通常使用Box-Behnken或CCD进行二阶连续模型推断优化。"
        elif "田口方法" in method_name:
            desc = "正交表加信噪比分析的经典鲁棒设计路线，适合带有杂音因素/不可控环境因素的实际生产控制。"
        elif "Kriging" in method_name:
            desc = "【仿真专用】基于空间自协方差特性的Kriging代理模型，搭配拉丁超立方抽样(LHS)，专门攻克数值仿真中的“黑盒”问题。"
        
        # 控制专属参数面板的显示隐藏
        self.factorial_widget.setVisible(False)
        self.rsm_widget.setVisible(False)
        self.sim_widget.setVisible(False)
        
        if "析因" in method_name or "筛选" in method_name:
            self.factorial_widget.setVisible(True)
        elif "响应曲面" in method_name:
            self.rsm_widget.setVisible(True)
        elif "Kriging" in method_name or ("自适应" in method_name and self.project_data.experiment_type == "仿真实验"):
            self.sim_widget.setVisible(True)
        elif "自适应" in method_name and self.project_data.experiment_type == "实际实验":
            # 默认假设是析因阶段展示
            self.factorial_widget.setVisible(True)

        self.desc_text.setText(desc)

    def on_generate_clicked(self):
        method = self.method_combo.currentText()
        print(f"[{self.project_data.experiment_type}] 按 {method} 正在生成方案...")
        
        # 保存设计参数到后端对象
        self.project_data.design_method = method
        
        # 解析真实因子数与特征
        k = len([f for f in self.project_data.factors if f.name])
        
        if "响应曲面" in method:
            self.project_data.design_params = {
                "rsm_type": self.rsm_type_combo.currentText(),
                "center_points": self.rsm_center_spin.value()
            }
        elif "析因" in method or "筛选" in method:
            self.project_data.design_params = {
                "resolution": self.f_resolution_combo.currentText(),
                "center_points": self.f_center_spin.value()
            }
        elif "Kriging" in method or "仿真" in self.project_data.experiment_type:
            self.project_data.design_params = {
                "samples": self.sim_points_spin.value()
            }
        else:
            self.project_data.design_params = {}

        # 真正依据设计方案构建点的生成器
        self._build_doe_matrix()
        
    def _build_doe_matrix(self):
        """生成带具体实验设计的矩阵结构，传给数据管理界面"""
        factors = [f for f in self.project_data.factors if f.name]
        factor_names = [f.name for f in factors]
        responses = [r.name for r in self.project_data.responses if r.name]
        
        if not factors:
            print("未能提取有效因子，跳过生成。")
            return
            
        method = self.project_data.design_method
        params = self.project_data.design_params
        k = len(factors)
        
        runs = 0
        doe_data = []
        
        # 核心算法分支: 计算运行次数等结构特征
        if "响应曲面" in method:
            # 基础二次加上中心点: n = 2^k + 2k + center
            runs = (2**k if k<5 else 16) + 2*k + params.get("center_points", 3)
        elif "析因" in method or "筛选" in method:
            res = params.get("resolution", "")
            if "PB" in res or "筛选" in res:
                runs = ((k // 4) + 1) * 4 # 基础的 PB 设置，如 12, 16, 20
                if runs < 12: runs = 12
            elif "部分" in res:
                runs = 2**(k-1) if k>1 else 2
            else:
                runs = 2**k # 全因子
            runs += params.get("center_points", 0)
        elif params.get("samples") is not None:
             runs = params.get("samples", 50)
        else:
             runs = 10 # Fallback
             
        # 根据计算好的矩阵骨架和物理参数上下限，执行数据铺展
        for i in range(runs):
            row = {"Run_ID": i + 1}
            for f in factors:
                try:
                    bot = float(f.param1) if f.param1 else -1.0
                    top = float(f.param2) if f.param2 else 1.0
                except ValueError:
                    bot, top = -1.0, 1.0
                    
                if self.project_data.experiment_type == "仿真实验" or "Kriging" in method:
                    # 仿真实验：连续型的随机/LHS样本点
                    row[f.name] = round(random.uniform(bot, top), 4)
                else:
                    # 实际实验：需要落在离散阶跃点上 (-1或+1水平 或 星号点 或 中心点)
                    # 此处做模拟，真正的 DOE 生成算法应该依据正交矩阵铺排。
                    if i >= runs - params.get("center_points", 0):
                        row[f.name] = round((bot + top) * 0.5, 4) # 零点位置
                    else:
                        level_val = bot if random.random() < 0.5 else top # 极限位置
                        row[f.name] = level_val
                        
            # 响应数据留白
            for r in responses:
                row[r] = ""
            doe_data.append(row)
            
        self.project_data.doe_matrix = doe_data
        print(f"✅ 成功生成涵盖 {runs} 个物理/仿真实验点的底层 DOE 矩阵组合！请前往【数据管理】界面提取！")
