from dataclasses import dataclass, field


@dataclass
class ResponseItem:
    name: str = ""
    kind: str = "目标"
    feature: str = "望小"
    lower: str = ""
    upper: str = ""
    unit: str = ""


@dataclass
class FactorItem:
    name: str = ""
    mode: str = "连续"
    source: str = "设计"
    uncertainty: str = "概率"
    lower: str = ""
    upper: str = ""
    unit: str = ""


@dataclass
class ProjectData:
    responses: list[ResponseItem] = field(default_factory=list)
    factors: list[FactorItem] = field(default_factory=list)