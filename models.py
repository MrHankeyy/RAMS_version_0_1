from dataclasses import dataclass, field


@dataclass
class ResponseItem:
    name: str = ""
    kind: str = "目标"
    feature: str = "望小"
    lower: str = ""       # 目标下限
    upper: str = ""       # 目标上限
    robust_limit: str = "" # 鲁棒约束边界极值 (针对带约束优化的范式3)
    unit: str = ""


@dataclass
class FactorItem:
    name: str = ""
    mode: str = "连续"
    source: str = "设计"
    uncertainty: str = "区间"      # 区间 或 概率
    distribution: str = "无"       # 正态分布, 均匀分布, 等
    param1: str = ""            # 区间下限 或 概率均值
    param2: str = ""            # 区间上限 或 概率方差
    unit: str = ""


@dataclass
class ProjectData:
    experiment_type: str = "实际实验" # 实际实验 或 仿真实验
    responses: list[ResponseItem] = field(default_factory=list)
    factors: list[FactorItem] = field(default_factory=list)
    
    # 方案配置与DOE数据
    design_method: str = ""
    design_params: dict = field(default_factory=dict)
    doe_matrix: list[dict] = field(default_factory=list) # 存储生成的DOE表格数据与测试结果