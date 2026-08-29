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
            "筛选设计 (Screening Design)",
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
        
        # 1. 筛选设计参数
        self.screening_widget = QWidget()
        f_layout = QHBoxLayout(self.screening_widget)
        f_layout.setContentsMargins(0,0,0,0)
        self.f_resolution_combo = QComboBox()
        self.f_resolution_combo.addItems(["Plackett-Burman (最简筛选)"])
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
        
        dynamic_layout.addWidget(self.screening_widget)
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
        """拉取业务建模数据，更新页面摘要。"""
        factors_count = len(self.project_data.factors)
        response_count = len(self.project_data.responses)
        
        has_noise = any(f.source == "环境" for f in self.project_data.factors)
        info = (f"因子数量: {factors_count} (含环境因子:{has_noise})\n"
                f"响应/目标数量: {response_count}")
        self.info_label.setText(info)

        self.update_description(self.method_combo.currentText())

    def update_description(self, method_name=None):
        if method_name is None:
            method_name = self.method_combo.currentText()
            
        desc = ""
        if "筛选设计" in method_name:
            desc = "常使用Plackett-Burman设计，用于在海量因子中快速剔除无效因子，锁定主干。"
        elif "响应曲面" in method_name:
            desc = "寻找最佳工艺及参数区间的利器！通常使用Box-Behnken或CCD进行二阶连续模型推断优化。"
        elif "田口方法" in method_name:
            desc = "正交表加信噪比分析的经典鲁棒设计路线，适合带有杂音因素/不可控环境因素的实际生产控制。"
        elif "Kriging" in method_name:
            desc = "【仿真专用】基于空间自协方差特性的Kriging代理模型，搭配拉丁超立方抽样(LHS)，专门攻克数值仿真中的“黑盒”问题。"
        
        # 控制专属参数面板的显示隐藏
        self.screening_widget.setVisible(False)
        self.rsm_widget.setVisible(False)
        self.sim_widget.setVisible(False)
        
        if "筛选" in method_name:
            self.screening_widget.setVisible(True)
        elif "响应曲面" in method_name:
            self.rsm_widget.setVisible(True)
        elif "Kriging" in method_name:
            self.sim_widget.setVisible(True)

        self.desc_text.setText(desc)

    def on_generate_clicked(self):
        method = self.method_combo.currentText()
        print(f"按 {method} 正在生成方案...")
        
        # 保存设计参数到后端对象
        self.project_data.design_method = method
        
        # 解析真实因子数与特征
        k = len([f for f in self.project_data.factors if f.name])
        
        if "响应曲面" in method:
            self.project_data.design_params = {
                "rsm_type": self.rsm_type_combo.currentText(),
                "center_points": self.rsm_center_spin.value()
            }
        elif "筛选" in method:
            self.project_data.design_params = {
                "resolution": self.f_resolution_combo.currentText(),
                "center_points": self.f_center_spin.value()
            }
        elif "Kriging" in method:
            self.project_data.design_params = {
                "samples": self.sim_points_spin.value()
            }
        else:
            self.project_data.design_params = {}

        # 真正依据设计方案构建点的生成器
        self._build_doe_matrix()
        
    def _build_doe_matrix(self):
        """生成带具体实验设计的矩阵结构，传给数据管理界面"""
        factors = [f for f in self.project_data.factors if f.name and f.source == "设计"]
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
        elif "筛选" in method:
            res = params.get("resolution", "")
            if "PB" in res or "筛选" in res:
                runs = ((k // 4) + 1) * 4 # 基础的 PB 设置，如 12, 16, 20
                if runs < 12: runs = 12
            else:
                runs = 10
            runs += params.get("center_points", 0)
        elif params.get("samples") is not None:
             runs = params.get("samples", 50)
        else:
             runs = 10 # Fallback
             
        # 根据计算好的矩阵骨架和物理参数上下限，执行数据铺展
        for i in range(runs):
            row = {"Run_ID": i + 1}
            for f in factors:
                if f.is_fixed and f.fixed_value:
                    try:
                        row[f.name] = float(f.fixed_value)
                    except ValueError:
                        row[f.name] = f.fixed_value
                    continue
                try:
                    bot = float(f.param1) if f.param1 else -1.0
                    top = float(f.param2) if f.param2 else 1.0
                except ValueError:
                    bot, top = -1.0, 1.0
                    
                if "Kriging" in method:
                    # 仿真实验：连续型的随机/LHS样本点
                    row[f.name] = round(random.uniform(bot, top), 4)
                else:
                    if i >= runs - params.get("center_points", 0):
                        row[f.name] = round((bot + top) * 0.5, 4)
                    else:
                        row[f.name] = bot if random.random() < 0.5 else top
                        
            # 响应数据留白
            for r in responses:
                row[r] = ""
            doe_data.append(row)
            
        self.project_data.doe_matrix = doe_data
        print(f"✅ 成功生成涵盖 {runs} 个物理/仿真实验点的底层 DOE 矩阵组合！请前往【数据管理】界面提取！")
