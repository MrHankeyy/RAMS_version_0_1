"""DOE 构造与后端分析引擎（参照 JMP《实验设计指南》思想的轻量实现）。

设计构造尽量使用真实结构：
- 两水平部分析因 / 全因子：编码 ±1 正交列；
- Plackett-Burman：以标准核心行循环移位生成 12/20 次试验；
- CCD / BBD：按标准点型组合（立方点/轴点/中心点，或 2k(k-1) 个 BBD 点）；
- 田口 L 表：由 GF(2)^d、GF(3)^d 的线性型生成 L4/L8/L9/L16/L27；
- 空间填充抽样：LHS / 均匀网格 / 随机。

响应面、田口信噪比与代理模型等“后端结果”计算放在同一模块，
供【设计优化】页面调用。
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
import time

try:
    import numpy as _np
    HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    HAS_NUMPY = False


# ---------------------------------------------------------------- 基础工具
def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def factor_limits(factor):
    """返回 (下限, 上限)，保证 low<=high。"""
    lo = _num(getattr(factor, "param1", None))
    hi = _num(getattr(factor, "param2", None))
    if getattr(factor, "uncertainty", "区间") == "概率":
        mean = lo
        variance = max(0.0, hi)
        if getattr(factor, "distribution", "") == "均匀分布":
            half_width = math.sqrt(3.0 * variance)
        else:
            half_width = 3.0 * math.sqrt(variance)
        lo, hi = mean - half_width, mean + half_width
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def simple_factors(factors, exclude_fixed=True):
    """转成 [(名称, 下限, 上限, 固定值|None), ...] 便于构造器统一处理。"""
    out = []
    for f in factors:
        if not (getattr(f, "name", "") or "").strip():
            continue
        fixed = None
        if exclude_fixed and getattr(f, "is_fixed", False):
            fixed = _num(getattr(f, "fixed_value", None), None)
        lo, hi = factor_limits(f)
        out.append((f.name, lo, hi, fixed))
    return out


def to_actual(code, low, high):
    """把编码值 -1/0/+1/α 映射到实际取值（0=中值）。"""
    return round((low + high) / 2.0 + code * (high - low) / 2.0, 8)


def _round_row(row):
    return {name: (round(v, 8) if isinstance(v, float) else v) for name, v in row.items()}


# ------------------------------------------------- GF 有限域正交表（L 表）
def _gf_oa_columns(prime: int, dim: int):
    """GF(prime)^dim 的全部非零线性型列，返回 (列数, 每列级别列表 0..p-1)。"""
    rows = list(itertools.product(range(prime), repeat=dim))
    coeffs = [c for c in itertools.product(range(prime), repeat=dim) if any(c)]
    # 首非零元素规范为 1，消除成比例的重复列
    norm = []
    for c in coeffs:
        i = next(i for i, v in enumerate(c) if v)
        if c[i] == 1:
            norm.append(c)
    columns = []
    for coeff in norm:
        col = [sum(a * b for a, b in zip(coeff, r)) % prime for r in rows]
        columns.append(col)
    return len(columns), columns


def _oa_levels(label: str):
    """返回标准 L 表的 (标签, 水平数, 行数, 列级别矩阵)。"""
    mapping = {
        "L4": (2, 2), "L8": (2, 3), "L16": (2, 4),
        "L9": (3, 2), "L27": (3, 3),
    }
    if label not in mapping:
        raise ValueError(f"不支持的正交表：{label}")
    prime, dim = mapping[label]
    col_count, columns = _gf_oa_columns(prime, dim)
    runs = prime ** dim
    matrix = [[columns[c][r] for c in range(col_count)] for r in range(runs)]
    return label, prime, runs, matrix


# ------------------------------------------------- Plackett-Burman
_PB_CORES = {
    12: "+ + - + + + - - - + -".replace(" ", ""),
    20: "+ + - - + + + + - + - + - - - - + + -".replace(" ", ""),
}


def _pb_matrix(n: int):
    """Plackett-Burman n×（n-1）正交矩阵（±1）。"""
    core = [1 if ch == "+" else -1 for ch in _PB_CORES[n]]
    rows = []
    for shift in range(n - 1):
        rows.append([core[(i - shift) % (n - 1)] for i in range(n - 1)])
    rows.append([-1] * (n - 1))  # 最后一行全低
    return rows


# ------------------------------------------------------------- 两水平筛选
SCREENING_KINDS = {
    "full": "全因子 2^k",
    "half": "部分析因 1/2 分数",
    "quarter": "部分析因 1/4 分数",
    "pb": "Plackett-Burman",
}


def screening_coded_runs(k: int, kind: str):
    """返回筛选设计的编码行 [[+1/-1/0(中心), ...], ...] 与元信息。"""
    limits = {"full": (1, 7), "half": (3, 10), "quarter": (5, 10), "pb": (1, 19)}
    if kind not in limits or not limits[kind][0] <= k <= limits[kind][1]:
        raise ValueError(f"{SCREENING_KINDS.get(kind, kind)}支持因子数：{limits.get(kind, '未知')}；当前 {k}。")
    if kind == "full":
        n = 2 ** k
        rows = [[1 if (i >> j) & 1 else -1 for j in range(k)] for i in range(n)]
        info_extra = f"全因子设计，2^{k} = {n} 次试验"
    elif kind == "half":
        n = 2 ** (k - 1)
        rows = []
        for i in range(n):
            row = [1 if (i >> j) & 1 else -1 for j in range(k - 1)]
            prod = 1
            for v in row:
                prod *= v
            rows.append(row + [prod])  # 第 k 个因子 = 前 k-1 个乘积
        info_extra = f"分辨率为 {k}（第 {k} 个因子作为生成元）"
    elif kind == "quarter":
        n = 2 ** (k - 2)
        masks = [m for m in range(1, n) if m.bit_count() >= 2]
        a, b = max(itertools.combinations(masks, 2), key=lambda pair: (
            min(pair[0].bit_count() + 1, pair[1].bit_count() + 1,
                (pair[0] ^ pair[1]).bit_count() + 2),
            pair[0].bit_count() + pair[1].bit_count() + (pair[0] ^ pair[1]).bit_count()))
        rows = []
        for i in range(n):
            row = [1 if (i >> j) & 1 else -1 for j in range(k - 2)]
            g1 = math.prod(v for j, v in enumerate(row) if a & (1 << j))
            g2 = math.prod(v for j, v in enumerate(row) if b & (1 << j))
            rows.append(row + [g1, g2])
        resolution = min(a.bit_count() + 1, b.bit_count() + 1, (a ^ b).bit_count() + 2)
        generator = lambda mask: "×".join(f"x{j+1}" for j in range(k-2) if mask & (1 << j))
        info_extra = f"分辨率 {resolution}；生成元 x{k-1}={generator(a)}，x{k}={generator(b)}；主效应正交，仍需检查交互混杂"
    elif kind == "pb":
        n = 12 if k <= 11 else 20
        matrix = _pb_matrix(n)
        rows = [row[:k] for row in matrix]
        info_extra = f"Plackett-Burman {n} 次试验（主效应正交）"
    else:
        raise ValueError(kind)
    return rows, n, info_extra


def build_screening(factors, kind: str, seed: int, centers: int = 0,
                    replicates: int = 1, randomize: bool = True):
    """构造两水平筛选设计表。factors: [(名称,低,高,固定值),...]"""
    if centers < 0 or replicates < 1:
        raise ValueError("中心点数不能为负，重复次数至少为 1")
    fixed_factors = [f for f in factors if f[3] is not None]
    factors = [f for f in factors if f[3] is None]
    k = len(factors)
    if kind == "full" and k > 7:
        raise ValueError("全因子设计仅支持不超过 7 个因子（试验次数 2^k 过大）")
    if kind == "half" and k > 10:
        raise ValueError("1/2 部分析因仅支持不超过 10 个因子")
    if kind == "quarter" and k < 5:
        raise ValueError("1/4 部分析因至少需要 5 个因子")
    if kind == "pb" and k > 19:
        raise ValueError("Plackett-Burman 最多支持 19 个因子")

    coded, base_runs, info = screening_coded_runs(k, kind)
    all_coded = list(coded)
    all_coded.extend([[0] * k] * centers)
    rows = []
    for _ in range(max(1, replicates)):
        for coded_row in all_coded:
            row = {}
            for (name, lo, hi, fixed), code in zip(factors, coded_row):
                row[name] = fixed if fixed is not None else to_actual(code, lo, hi)
            rows.append(_round_row(row))
            rows[-1].update({f[0]: f[3] for f in fixed_factors})
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(rows)
    meta = {
        "kind": SCREENING_KINDS[kind],
        "base_runs": base_runs,
        "centers": centers,
        "replicates": replicates,
        "total_runs": len(rows),
        "diagnostic": info,
        "active_factors": [f[0] for f in factors],
    }
    return rows, meta


# ------------------------------------------------------------------ CCD / BBD
def ccd_runs(k: int, alpha: float, centers: int):
    rows = []
    cube = int(2 ** k)
    for i in range(cube):  # 立方点 ±1
        row = [1 if (i >> j) & 1 else -1 for j in range(k)]
        rows.append(row)
    for j in range(k):  # 轴点
        for sign in (1, -1):
            row = [0.0] * k
            row[j] = sign * alpha
            rows.append(row)
    for _ in range(centers):
        rows.append([0.0] * k)
    return rows


def bbd_runs(k: int, centers: int):
    rows = []
    for i, j in itertools.combinations(range(k), 2):  # 每对因子 4 个点
        for xi in (-1, 1):
            for xj in (-1, 1):
                row = [0.0] * k
                row[i] = xi
                row[j] = xj
                rows.append(row)
    for _ in range(centers):
        rows.append([0.0] * k)
    return rows


def rotatable_alpha(k: int):
    return round((2 ** k) ** 0.25, 6)


def build_response_surface(factors, rsm_type: str, alpha_mode: str,
                           custom_alpha: float, centers: int,
                           replicates: int = 1, seed: int = 0,
                           randomize: bool = True):
    """构造 CCD 或 BBD。rsm_type: 'CCD'/'BBD'。"""
    if centers < 0 or replicates < 1:
        raise ValueError("中心点数不能为负，重复次数至少为 1")
    fixed_factors = [f for f in factors if f[3] is not None]
    factors = [f for f in factors if f[3] is None]
    k = len(factors)
    if rsm_type not in ("CCD", "BBD"):
        raise ValueError("响应曲面类型必须为 CCD 或 BBD")
    if rsm_type == "BBD":
        if k < 3:
            raise ValueError("Box-Behnken 设计至少需要 3 个因子")
        if k > 7:
            raise ValueError("Box-Behnken 设计最多支持 7 个因子（试验次数过多）")
        coded = bbd_runs(k, centers)
        alpha = 0.0
        design_label = f"Box-Behnken 设计（k={k}）"
    else:
        if k < 2:
            raise ValueError("中心复合设计至少需要 2 个因子")
        if k > 6:
            raise ValueError("中心复合设计最多支持 6 个因子")
        rot_alpha = rotatable_alpha(k)
        if alpha_mode == "rotatable":
            alpha = rot_alpha
        elif alpha_mode == "face":
            alpha = 1.0
        else:
            alpha = float(custom_alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("CCD 轴值必须为正数")
        coded = ccd_runs(k, alpha, centers)
        design_label = f"中心复合设计 CCD（k={k}, α={alpha}）"

    if _np.linalg.matrix_rank(_design_matrix(_np.asarray(coded, float))[0]) < (k + 1) * (k + 2) // 2:
        raise ValueError("当前点型无法识别完整二次模型；请增加中心点或调整 CCD 轴值。")
    rows = []
    for _ in range(max(1, replicates)):
        for coded_row in coded:
            row = {}
            for (name, lo, hi, fixed), code in zip(factors, coded_row):
                row[name] = fixed if fixed is not None else to_actual(code, lo, hi)
            rows.append(_round_row(row))
            rows[-1].update({f[0]: f[3] for f in fixed_factors})
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(rows)

    cube_count = (2 ** k) if rsm_type == "CCD" else (2 * k * (k - 1))
    meta = {
        "design": design_label,
        "alpha": alpha if rsm_type == "CCD" else None,
        "cube_points": cube_count,
        "axial_points": 2 * k if rsm_type == "CCD" else 0,
        "centers": centers,
        "replicates": replicates,
        "total_runs": len(rows),
        "active_factors": [f[0] for f in factors],
        "bounds_note": "CCD 的 α>1 轴点超出所填低/高水平；这些水平对应编码 ±1，不是硬边界。" if rsm_type == "CCD" and alpha > 1 else "所有设计点位于所填低/高水平内。",
    }
    return rows, meta


# ---------------------------------------------------------------- 田口设计
def taguchi_arrays(label: str, k_needed: int):
    """按标签返回级别矩阵（行×k 个因子）并校验容量。"""
    label = label.replace(" ", "")
    if label.startswith("L12"):
        n, levels, columns = 12, 2, _pb_matrix(12)
        if k_needed > 11:
            raise ValueError(f"{label} 最多容纳 11 个因子")
        matrix = [cols[:k_needed] for cols in columns]
        matrix = [[0 if v == -1 else 1 for v in row] for row in matrix]
        return label, levels, n, matrix
    short = next((s for s in ("L4", "L8", "L9", "L16", "L27") if label.startswith(s)), None)
    if not short:
        raise ValueError(f"不支持的正交表：{label}")
    _, levels, n, all_cols = _oa_levels(short)
    capacity = len(all_cols[0]) if all_cols else 0
    if k_needed > capacity:
        raise ValueError(f"{short} 最多容纳 {capacity} 个 {levels} 水平因子")
    matrix = [row[:k_needed] for row in all_cols]
    return short, levels, n, matrix


def _levels_of_factor(lo, hi, n_levels):
    if n_levels == 2:
        return [lo, hi]
    return [lo, (lo + hi) / 2.0, hi]


def build_taguchi(control_factors, noise_factors, inner_label, outer_label,
                  levels: int, seed: int = 0, replicates: int = 1, randomize: bool = True):
    """田口设计：内表(控制因子) × 外表(噪声因子) 叉积。返回行与分组信息。"""
    fixed = {f[0]: f[3] for f in control_factors + noise_factors if f[3] is not None}
    control_factors = [f for f in control_factors if f[3] is None]
    noise_factors = [f for f in noise_factors if f[3] is None]
    if not control_factors:
        raise ValueError("田口设计至少需要一个未固定的控制因子")
    if replicates < 1:
        raise ValueError("重复次数必须至少为 1")
    c_label, c_levels, n_inner, c_matrix = taguchi_arrays(inner_label, len(control_factors))
    if levels != c_levels:
        raise ValueError(f"内表 {c_label} 是 {c_levels} 水平表，与所选 {levels} 水平不一致")
    factor_level_maps = {}
    for (name, lo, hi, _fixed), idx in zip(control_factors, range(len(control_factors))):
        factor_level_maps[name] = _levels_of_factor(lo, hi, levels)

    if noise_factors:
        n_label, n_levels, n_outer, n_matrix = taguchi_arrays(outer_label, len(noise_factors))
        noise_level_maps = {}
        for (name, lo, hi, _fixed), idx in zip(noise_factors, range(len(noise_factors))):
            noise_level_maps[name] = _levels_of_factor(lo, hi, n_levels)
        has_outer = True
    else:
        n_label, n_outer, n_matrix, noise_level_maps, n_levels = "无", 1, [[0]], {}, levels
        has_outer = False

    rows = []
    groups = []
    for ci in range(n_inner):
        for oi in range(n_outer):
            row = {}
            for (name, *_rest), idx in zip(control_factors, range(len(control_factors))):
                row[name] = factor_level_maps[name][c_matrix[ci][idx]]
            if has_outer:
                for (name, *_rest), idx in zip(noise_factors, range(len(noise_factors))):
                    row[name] = noise_level_maps[name][n_matrix[oi][idx]]
            rows.append(_round_row(row))
            rows[-1].update(fixed)
            groups.append({"inner": ci, "outer": oi if has_outer else 0})
    base_rows, base_groups = rows, groups
    rows, groups = [], []
    for rep in range(replicates):
        rows.extend(dict(row) for row in base_rows)
        groups.extend({**g, "replicate": rep} for g in base_groups)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    if randomize:
        rng.shuffle(order)
    rows = [rows[i] for i in order]
    groups = [groups[i] for i in order]
    meta = {
        "inner_label": c_label,
        "outer_label": n_label if has_outer else "无",
        "levels": levels,
        "inner_runs": n_inner,
        "outer_runs": n_outer if has_outer else 1,
        "has_outer": has_outer,
        "replicates": replicates,
        "noise_level_count": n_levels if has_outer else 0,
        "total_runs": len(rows),
        "factor_levels": factor_level_maps,
        "noise_levels": noise_level_maps if has_outer else {},
        "groups": groups,
    }
    return rows, meta


# ---------------------------------------------------------------- 空间填充抽样
def _latin_hypercube(n_points, dimensions, rng):
    strata = [[(i + rng.random()) / n_points for i in range(n_points)]
              for _ in range(dimensions)]
    for values in strata:
        rng.shuffle(values)
    return [[strata[col][row] for col in range(dimensions)]
            for row in range(n_points)]


def _maximin_lhs(n_points, dimensions, rng, candidates=12):
    """用小规模候选搜索提高 LHS 的空间分散性，避免依赖 scipy。"""
    best = None
    best_score = -1.0
    for _ in range(candidates):
        points = _latin_hypercube(n_points, dimensions, rng)
        score = min(
            sum((a - b) ** 2 for a, b in zip(left, right))
            for i, left in enumerate(points)
            for right in points[i + 1:]
        ) if n_points > 1 else 1.0
        if score > best_score:
            best_score, best = score, points
    return best


def _van_der_corput(index, base):
    value, factor = 0.0, 1.0 / base
    while index:
        index, remainder = divmod(index, base)
        value += remainder * factor
        factor /= base
    return value


def _sobol_like(n_points, dimensions):
    """优先使用 scipy 的 Sobol；无 scipy 时回退到确定性的低差异序列。"""
    try:
        from scipy.stats import qmc  # type: ignore
        return qmc.Sobol(d=dimensions, scramble=False).random(n_points).tolist()
    except Exception:
        pass
    bases = [2, 3, 5, 7, 11, 13, 17, 19]
    return [[_van_der_corput(index + 1, bases[col % len(bases)])
             for col in range(dimensions)] for index in range(n_points)]


def build_samples(factors, n_points: int, method: str, seed: int):
    """代理模型试验样本（LHS、最优 LHS、Sobol、网格、稀疏网格或随机）。"""
    rng = random.Random(seed)
    k = len(factors)
    rows = []
    if method in ("均匀网格抽样", "均匀网格") and k <= 4 and n_points >= 2 ** k:
        side = max(2, int(round(n_points ** (1.0 / k))))
        points = [lo + (hi - lo) * (t / max(side - 1, 1))
                  for (_, lo, hi, _) in factors for t in range(side)]
        chunks = [points[i * side:(i + 1) * side] for i in range(k)]
        combos = list(itertools.product(*chunks))
        if len(combos) > n_points:
            combos = rng.sample(combos, n_points)
        for combo in combos:
            rows.append({factors[i][0]: combo[i] for i in range(k)})
    elif method in ("Latin Hypercube (LHS)", "最优 LHS"):
        points = (_maximin_lhs(n_points, k, rng)
                  if method == "最优 LHS" else _latin_hypercube(n_points, k, rng))
        for point in points:
            rows.append({name: lo + point[i] * (hi - lo)
                         for i, (name, lo, hi, _) in enumerate(factors)})
    elif method in ("低差异序列 SOBOL", "SOBOL（低差异序列）"):
        for point in _sobol_like(n_points, k):
            rows.append({name: lo + point[i] * (hi - lo)
                         for i, (name, lo, hi, _) in enumerate(factors)})
    elif method == "稀疏配点法":
        # 每次只激活少量维度，覆盖中心、轴向及低阶组合点。
        center = [0.5] * k
        unit_points = [center]
        for i in range(k):
            for value in (0.0, 1.0):
                point = list(center)
                point[i] = value
                unit_points.append(point)
        for i, j in itertools.combinations(range(k), 2):
            point = list(center)
            point[i], point[j] = 0.0, 1.0
            unit_points.append(point)
        while len(unit_points) < n_points:
            point = list(center)
            for i in rng.sample(range(k), min(2, k)):
                point[i] = rng.choice((0.0, 1.0))
            unit_points.append(point)
        for point in unit_points[:n_points]:
            rows.append({name: lo + point[i] * (hi - lo)
                         for i, (name, lo, hi, _) in enumerate(factors)})
    else:  # 随机抽样
        for _ in range(n_points):
            rows.append({name: lo + rng.random() * (hi - lo) for name, lo, hi, _ in factors})
    rng.shuffle(rows)
    return [_round_row(r) for r in rows]


# ===================================================================== 后端分析
def _design_matrix(X):
    """二次模型设计阵：1 + 线性 + 平方 + 两两交互。"""
    if not HAS_NUMPY:
        raise RuntimeError("需要 numpy 支持回归计算")
    n, k = X.shape
    cols = [_np.ones(n)]
    labels = ["常数项"]
    for j in range(k):
        cols.append(X[:, j])
        labels.append(f"x{j + 1}")
        cols.append(X[:, j] ** 2)
        labels.append(f"x{j + 1}²")
    for i in range(k):
        for j in range(i + 1, k):
            cols.append(X[:, i] * X[:, j])
            labels.append(f"x{i + 1}·x{j + 1}")
    return _np.column_stack(cols), labels


def fit_response_surface(X, y):
    """最小二乘拟合二次响应面，返回系数、拟合优度与预测值。"""
    A, labels = _design_matrix(X)
    n, p = A.shape
    if _np.linalg.matrix_rank(A) < p:
        raise ValueError("试验数据不能识别完整二次模型")
    coef, *_ = _np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    ss_tot = float(_np.sum((y - y.mean()) ** 2)) or 1e-12
    ss_res = float(_np.sum((y - y_hat) ** 2))
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p, 1)
    rmse = math.sqrt(ss_res / max(n, 1))
    return {
        "coefficients": list(zip(labels, [round(float(c), 6) for c in coef])),
        "r2": round(r2, 6),
        "adj_r2": round(adj_r2, 6),
        "rmse": round(rmse, 6),
        "df_model": p - 1,
        "df_residual": n - p,
        "predicted": [float(v) for v in y_hat],
    }


def quadratic_optimum(fit_result, X_min, X_max, objective, target=None, samples=8000):
    """在可行域内随机采样搜索二次模型的优选点。objective: max/min/target。"""
    if not HAS_NUMPY:
        return None
    rng = _np.random.default_rng(2026)
    k = len(X_min)
    X = rng.uniform(0, 1, size=(samples, k)) * (_np.array(X_max) - _np.array(X_min)) + _np.array(X_min)
    A, _ = _design_matrix(X)
    pred = A @ _np.array([c for _, c in fit_result["coefficients"]])
    if objective == "max":
        idx = int(_np.argmax(pred))
    elif objective == "min":
        idx = int(_np.argmin(pred))
    else:
        t = float(target)
        idx = int(_np.argmin(_np.abs(pred - t)))
    return {
        "x": [round(float(v), 6) for v in X[idx]],
        "predicted": round(float(pred[idx]), 6),
    }


def screening_from_responses(factors, doe_matrix, response_name):
    """根据已填响应按 ± 编码计算主效应（中心点自动剔除）。"""
    lows, highs = {}, {}
    for f in factors:
        lo, hi = factor_limits(f)
        lows[f.name], highs[f.name] = lo, hi
    vals_by_sign = {f.name: {"+": [], "-": []} for f in factors}
    for row in doe_matrix:
        try:
            y = float(row.get(response_name, ""))
        except (TypeError, ValueError):
            continue
        for f in factors:
            try:
                v = float(row.get(f.name))
            except (TypeError, ValueError):
                continue
            mid = (lows[f.name] + highs[f.name]) / 2.0
            if abs(v - mid) < 1e-9:
                continue
            vals_by_sign[f.name]["-" if v < mid else "+"].append(y)
    effects = {}
    for f in factors:
        plus = vals_by_sign[f.name]["+"]
        minus = vals_by_sign[f.name]["-"]
        if plus and minus:
            effects[f.name] = statistics.mean(plus) - statistics.mean(minus)
    return effects


# ------------------------------------------------------------------ 代理模型
def _kernel_rbf(x, z, gamma):
    diff = x[:, None, :] - z[None, :, :]
    return _np.exp(-gamma * _np.sum(diff ** 2, axis=2))


def _ann_forward(x, w1, b1, w2, b2, activation):
    z1 = x @ w1 + b1
    if activation == "relu":
        a1 = _np.maximum(z1, 0.0)
    elif activation == "tanh":
        a1 = _np.tanh(z1)
    else:
        a1 = 1.0 / (1.0 + _np.exp(-z1))
    out = a1 @ w2 + b2
    return a1, out


def train_kriging(Xtr, ytr, Xte, correlation="Gaussian", trend="常数"):
    if not HAS_NUMPY:
        raise RuntimeError("numpy 不可用")
    Xtr = _np.asarray(Xtr, float)
    ytr = _np.asarray(ytr, float)
    Xte = _np.asarray(Xte, float)
    n = Xtr.shape[0]
    span = Xtr.max(axis=0) - Xtr.min(axis=0)
    span[span == 0] = 1.0
    xz = Xtr / span
    tz = Xte / span

    corr_label = correlation
    if "Gaussian" in correlation or "高斯" in correlation:
        corr_label = "Gaussian"
    elif "指数" in correlation or "Exponential" in correlation:
        corr_label = "指数"
    elif "Matern" in correlation:
        corr_label = "Matern 3/2"

    def corr(a, b, theta):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        if corr_label == "指数":
            return _np.exp(-_np.sqrt(d2) * theta)
        if corr_label == "Matern 3/2":
            s = _np.sqrt(d2) * theta
            return (1 + s) * _np.exp(-s)
        return _np.exp(-d2 * theta)  # Gaussian / Power(p=2)

    def trend_matrix(values):
        columns = [_np.ones(len(values))]
        if "一次" in trend or "二次" in trend:
            columns.extend(values[:, j] for j in range(values.shape[1]))
        if "二次" in trend:
            columns.extend(values[:, j] ** 2 for j in range(values.shape[1]))
        return _np.column_stack(columns)

    Ftr = trend_matrix(xz)
    Fte = trend_matrix(tz)
    best = None
    for theta in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        R = corr(xz, xz, theta) + _np.eye(n) * 1e-8
        try:
            L = _np.linalg.cholesky(R)
        except Exception:
            continue
        r_inv_y = _np.linalg.solve(R, ytr)
        r_inv_f = _np.linalg.solve(R, Ftr)
        beta = _np.linalg.solve(Ftr.T @ r_inv_f, Ftr.T @ r_inv_y)
        res = ytr - Ftr @ beta
        # 集中似然同时估计过程方差，避免响应量纲左右相关长度选择。
        variance = max(float(res @ _np.linalg.solve(R, res)) / n, 1e-30)
        log_like = -_np.sum(_np.log(_np.diag(L))) - 0.5 * n * math.log(variance)
        if best is None or log_like > best[0]:
            best = (log_like, theta, R)
    _loglike, theta, R = best
    r_inv_y = _np.linalg.solve(R, ytr)
    r_inv_f = _np.linalg.solve(R, Ftr)
    beta = _np.linalg.solve(Ftr.T @ r_inv_f, Ftr.T @ r_inv_y)
    w = _np.linalg.solve(R, ytr - Ftr @ beta)
    rte = corr(xz, tz, theta)
    pred = Fte @ beta + rte.T @ w
    # 保留模型的光滑预测；裁剪到样本响应范围会产生虚假的零梯度/零方差平台。
    def predict_fn(values):
        values = _np.asarray(values, float).reshape(1, -1)
        scaled = values / span
        trend_value = trend_matrix(scaled)
        kernel_value = corr(xz, scaled, theta)
        return float((trend_value @ beta + kernel_value.T @ w)[0])
    return {"params": {"theta": theta, "correlation": corr_label, "trend": trend},
            "predicted": [float(v) for v in pred], "predict_fn": predict_fn}


def train_svr(Xtr, ytr, Xte, kernel="RBF", poly_degree=3,
              c_min=0.1, c_max=10.0, c_step=1.0,
              g_min=0.01, g_max=10.0, g_step=1.0):
    """SVR 训练。

    若环境装有 scikit-learn 则调用其 SVR；否则退回 RBF 核岭回归（KRR），
    并按 C/g 网格寻优，输出中会注明所用求解器。
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy 不可用")
    Xtr = _np.asarray(Xtr, float)
    Xte = _np.asarray(Xte, float)
    ytr = _np.asarray(ytr, float)

    def grid_range(lo, hi, step):
        vals = []
        v = float(lo)
        while v <= float(hi) + 1e-9:
            vals.append(round(v, 6))
            v += float(step)
        return vals or [1.0]

    try:
        from sklearn.svm import SVR  # type: ignore
        kernel_arg = ("rbf" if "RBF" in kernel or kernel == "RBF"
                      else ("poly" if "多项式" in kernel else "sigmoid"))
        best, best_err = None, float("inf")
        for c in grid_range(c_min, c_max, c_step):
            for gamma in grid_range(g_min, g_max, g_step):
                model = SVR(kernel=kernel_arg, degree=int(poly_degree), C=c,
                            gamma=gamma, epsilon=0.05)
                model.fit(Xtr, ytr)
                pred = model.predict(Xtr)
                err = float(_np.mean((pred - ytr) ** 2))
                if err < best_err:
                    best_err, best = err, (c, gamma)
        c, gamma = best
        model = SVR(kernel=kernel_arg, degree=int(poly_degree), C=c,
                gamma=gamma, epsilon=0.05)
        model.fit(Xtr, ytr)
        def predict_fn(values):
            value = _np.asarray(values, float).reshape(1, -1)
            return float(model.predict(value)[0])
        return {"params": {"kernel": kernel, "C": c, "gamma": gamma,
                   "poly_degree": int(poly_degree),
                           "solver": "scikit-learn SVR"},
                "predicted": [float(v) for v in model.predict(Xte)],
                "predict_fn": predict_fn}
    except Exception:
        # KRR 近似：核矩阵 + 岭回归，C 视为 1/λ 的缩放
        best_pred, best_err, best_params = None, float("inf"), (1.0, 1.0)
        for c in grid_range(c_min, c_max, c_step):
            for gamma in grid_range(g_min, g_max, g_step):
                K = _kernel_rbf(Xtr, Xtr, gamma)
                lam = 1.0 / max(c, 1e-6)
                alpha = _np.linalg.solve(K + lam * _np.eye(len(ytr)), ytr)
                pred_tr = K @ alpha
                err = float(_np.mean((pred_tr - ytr) ** 2))
                if err < best_err:
                    best_err = err
                    best_params = (c, gamma)
                    best_alpha = alpha.copy()
                    best_pred = _kernel_rbf(Xtr, Xte, gamma).T @ alpha
        def predict_fn(values):
            value = _np.asarray(values, float).reshape(1, -1)
            return float((_kernel_rbf(Xtr, value, best_params[1]).T @ best_alpha).item())
        return {"params": {"kernel": kernel, "C": best_params[0],
                           "gamma": best_params[1],
                   "poly_degree": int(poly_degree),
                           "solver": "核岭回归 KRR（未安装 scikit-learn 的 SVR 近似）"},
                "predicted": [float(v) for v in best_pred],
                "predict_fn": predict_fn}


def train_ann(Xtr, ytr, Xte, hidden_layers=1, hidden_nodes=8, activation="sigmoid",
              learning_rate=0.05, epochs=300, node_search=None):
    """BP 神经网络训练（numpy 实现；有 scikit-learn 则优先使用 MLPRegressor）。"""
    if not HAS_NUMPY:
        raise RuntimeError("numpy 不可用")
    Xtr = _np.asarray(Xtr, float)
    Xte = _np.asarray(Xte, float)
    ytr = _np.asarray(ytr, float)
    Xmean, Xstd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    ymean, ystd = ytr.mean(), ytr.std() + 1e-9
    Xs, ys = (Xtr - Xmean) / Xstd, (ytr - ymean) / ystd
    try:
        from sklearn.neural_network import MLPRegressor  # type: ignore
        if node_search:
            lo, hi, step = node_search
            layer_sizes = [int(max(2, lo + i * step)) for i in range(int((hi - lo) / max(step, 1)) + 1)]
            hidden = tuple(layer_sizes[:int(hidden_layers)]) if int(hidden_layers) > 1 else layer_sizes[0]
        else:
            hidden = tuple([int(hidden_nodes)] * int(hidden_layers)) if int(hidden_layers) > 1 else int(hidden_nodes)
        act = "logistic" if activation == "sigmoid" else activation
        model = MLPRegressor(hidden_layer_sizes=hidden, activation=act,
                             learning_rate_init=float(learning_rate),
                             max_iter=int(epochs), random_state=7)
        model.fit(Xs, ys.ravel())
        def predict_fn(values):
            value = (_np.asarray(values, float).reshape(1, -1) - Xmean) / Xstd
            return float(model.predict(value)[0] * ystd + ymean)
        pred = model.predict((Xte - Xmean) / Xstd) * ystd + ymean
        return {"params": {"layers": hidden, "solver": "scikit-learn MLP"},
            "predicted": [float(v) for v in pred], "predict_fn": predict_fn}
    except Exception:
        rng = _np.random.default_rng(7)
        if node_search:
            nodes = int(max(2, (node_search[0] + node_search[1]) / 2.0))
        else:
            nodes = int(hidden_nodes)
        layers = [nodes] * int(max(hidden_layers, 1))
        nh = layers[0]
        w1 = rng.normal(0, 0.5, (Xs.shape[1], nh))
        b1 = _np.zeros(nh)
        w2 = rng.normal(0, 0.5, (nh, 1))
        b2 = _np.zeros(1)
        for _ in range(int(epochs)):
            a1, out = _ann_forward(Xs, w1, b1, w2, b2, activation)
            err = ys[:, None] - out
            if activation == "relu":
                da1 = (a1 > 0).astype(float)
            elif activation == "tanh":
                da1 = 1 - a1 ** 2
            else:
                da1 = a1 * (1 - a1)
            dw2 = a1.T @ err
            db2 = err.sum(axis=0)
            dz1 = (err @ w2.T) * da1
            dw1 = Xs.T @ dz1
            db1 = dz1.sum(axis=0)
            w1 += learning_rate * dw1 / len(Xs)
            b1 += learning_rate * db1 / len(Xs)
            w2 += learning_rate * dw2 / len(Xs)
            b2 += learning_rate * db2 / len(Xs)
        _, out = _ann_forward((Xte - Xmean) / Xstd, w1, b1, w2, b2, activation)
        pred = out[:, 0] * ystd + ymean
        def predict_fn(values):
            value = (_np.asarray(values, float).reshape(1, -1) - Xmean) / Xstd
            _, result = _ann_forward(value, w1, b1, w2, b2, activation)
            return float(result[0, 0] * ystd + ymean)
        return {"params": {"layers": tuple(layers), "solver": "numpy BP"},
            "predicted": [float(v) for v in pred], "predict_fn": predict_fn}


def train_surrogate(x_rows, response_values, test_ratio=0.2, seed=2026,
                    model="Kriging", model_params=None):
    """训练单个响应的代理模型，返回评估指标（训练/测试误差、耗时等）。"""
    if not HAS_NUMPY:
        return {"error": "当前环境未安装 numpy，无法训练代理模型。"}
    model_params = model_params or {}
    # 设置对话框产生扁平参数；同时兼容早期脚本使用的按模型分组格式。
    model_key = {"Kriging": "kriging", "SVR": "svr"}.get(model, "ann")
    params = dict(model_params.get(model_key, model_params))
    params.pop("net_mode", None)  # 对话框选项，不是训练器参数。
    X = _np.array(x_rows, float)
    y = _np.array(response_values, float)
    if len(y) < 6:
        return {"error": f"有效样本仅 {len(y)} 个，至少需要 6 个才能训练与评估。"}
    rng = _np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(round(len(y) * 0.2)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    t0 = time.time()
    try:
        if model == "Kriging":
            result = train_kriging(Xtr, ytr, Xte, **params)
        elif model == "SVR":
            result = train_svr(Xtr, ytr, Xte, **params)
        else:
            result = train_ann(Xtr, ytr, Xte, **params)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"训练失败：{exc}"}
    elapsed = time.time() - t0
    pred = _np.array(result["predicted"])
    denom = float(_np.abs(yte).mean()) + 1e-9
    mape_test = float(_np.mean(_np.abs(yte - pred) / denom) * 100)
    ss_tot = float(_np.sum((yte - yte.mean()) ** 2)) + 1e-12
    r2_test = float(1 - _np.sum((yte - pred) ** 2) / ss_tot)
    return {
        "params": result["params"],
        "predict_fn": result.get("predict_fn"),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_time_s": round(elapsed, 4),
        "mape_test_%": round(mape_test, 4),
        "r2_test": round(r2_test, 4),
        "actual": [float(v) for v in yte],
        "predicted": [float(v) for v in pred],
    }
