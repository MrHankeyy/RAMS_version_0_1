from dataclasses import dataclass, field


@dataclass
class ResponseItem:
    name: str = ""
    kind: str = "目标"          # 目标 / 约束 / 目标+约束
    feature: str = "望小"       # 望大 / 望小 / 望目
    lower: str = ""             # 望目时: 稳定性阈值下界
    upper: str = ""             # 望目时: 稳定性阈值上界
    robust_limit: str = ""      # 稳定性阈值: 望大=稳定要求最小值, 望小=稳定要求最大值
    unit: str = ""


@dataclass
class FactorItem:
    name: str = ""
    source: str = "设计"        # 设计 / 环境
    uncertainty: str = "区间"   # 区间 或 概率
    distribution: str = "无"    # 正态分布, 均匀分布, 等
    param1: str = ""            # 区间下限 或 概率均值
    param2: str = ""            # 区间上限 或 概率方差
    fixed_value: str = ""       # 固定后的均值或区间中点
    is_fixed: bool = False
    unit: str = ""


@dataclass
class ScreeningResult:
    factor_rankings: list[dict] = field(default_factory=list)
    main_effects: list[dict] = field(default_factory=list)
    stability_impact: list[dict] = field(default_factory=list)
    diagnostic_summary: str = ""
    conclusion: str = ""
    plot_points: list[dict] = field(default_factory=list)


@dataclass
class TaguchiResult:
    array_name: str = ""
    orthogonal_array: list[dict] = field(default_factory=list)
    signal_to_noise: list[dict] = field(default_factory=list)
    best_levels: dict = field(default_factory=dict)
    recommendation: str = ""
    summary: str = ""


@dataclass
class ProjectData:
    responses: list[ResponseItem] = field(default_factory=list)
    factors: list[FactorItem] = field(default_factory=list)

    # 方案配置与DOE数据
    design_method: str = ""
    design_params: dict = field(default_factory=dict)
    doe_matrix: list[dict] = field(default_factory=list) # 存储生成的DOE表格数据与测试结果

    # 真实分析结果（筛选与田口）
    screening_result: dict = field(default_factory=dict)
    taguchi_result: dict = field(default_factory=dict)

    # 响应曲面 / 代理模型分析结果
    rsm_result: dict = field(default_factory=dict)
    surrogate_config: dict = field(default_factory=dict)   # 方案配置中选择的代理模型参数
    surrogate_result: dict = field(default_factory=dict)   # 后端训练/评估结果
