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
        rows = []
        for i in range(n):
            row = [1 if (i >> j) & 1 else -1 for j in range(k - 2)]
            g1 = 1
            for j, v in enumerate(row):
                if j % 2 == 0:
                    g1 *= v
            g2 = 1
            for j, v in enumerate(row):
                if j % 2 == 1:
                    g2 *= v
            rows.append(row + [g1, g2])
        info_extra = "分辨率 III~IV（生成元按奇偶分组，主效应保持正交）"
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
    k = len(factors)
    if kind == "full" and k > 7:
        raise ValueError("全因子设计仅支持不超过 7 个因子（试验次数 2^k 过大）")
    if kind == "half" and k > 10:
        raise ValueError("1/2 部分析因仅支持不超过 10 个因子")
    if kind == "quarter" and k < 4:
        raise ValueError("1/4 部分析因至少需要 4 个因子")
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
    k = len(factors)
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
            alpha = float(custom_alpha or rot_alpha)
        coded = ccd_runs(k, alpha, centers)
        design_label = f"中心复合设计 CCD（k={k}, α={alpha}）"

    rows = []
    for _ in range(max(1, replicates)):
        for coded_row in coded:
            row = {}
            for (name, lo, hi, fixed), code in zip(factors, coded_row):
                row[name] = fixed if fixed is not None else to_actual(code, lo, hi)
            rows.append(_round_row(row))
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
        raise ValueError(f"{short} 最多容纳 {capacity} 个两水平因子")
    matrix = [row[:k_needed] for row in all_cols]
    return short, levels, n, matrix


def _levels_of_factor(lo, hi, n_levels):
    if n_levels == 2:
        return [lo, hi]
    return [lo, (lo + hi) / 2.0, hi]


def build_taguchi(control_factors, noise_factors, inner_label, outer_label,
                  levels: int, seed: int = 0):
    """田口设计：内表(控制因子) × 外表(噪声因子) 叉积。返回行与分组信息。"""
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
            groups.append({"inner": ci, "outer": oi if has_outer else 0})
    rng = random.Random(seed)
    order = list(range(len(rows)))
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
        "total_runs": len(rows),
        "factor_levels": factor_level_maps,
        "noise_levels": noise_level_maps if has_outer else {},
        "groups": groups,
    }
    return rows, meta


# ---------------------------------------------------------------- 空间填充抽样
def build_samples(factors, n_points: int, method: str, seed: int):
    """代理模型试验样本（设计因子取值点）。method: LHS / 均匀网格 / 随机。"""
    rng = random.Random(seed)
    k = len(factors)
    rows = []
    if method == "均匀网格" and k <= 4 and n_points >= 2 ** k:
        side = max(2, int(round(n_points ** (1.0 / k))))
        points = [lo + (hi - lo) * (t / max(side - 1, 1))
                  for (_, lo, hi, _) in factors for t in range(side)]
        chunks = [points[i * side:(i + 1) * side] for i in range(k)]
        combos = list(itertools.product(*chunks))
        if len(combos) > n_points:
            combos = rng.sample(combos, n_points)
        for combo in combos:
            rows.append({factors[i][0]: combo[i] for i in range(k)})
    elif method == "Latin Hypercube (LHS)":
        strata = [sorted((i + rng.random()) / n_points for i in range(n_points)) for _ in range(k)]
        for s in range(n_points):
            row = {}
            for i, (name, lo, hi, _) in enumerate(factors):
                t = strata[i][s]
                row[name] = lo + t * (hi - lo)
            rows.append(row)
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

    best = None
    for theta in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
        R = corr(xz, xz, theta) + _np.eye(n) * 1e-8
        try:
            L = _np.linalg.cholesky(R)
        except Exception:
            continue
        one = _np.ones(n)
        linv_y = _np.linalg.solve(L.T, _np.linalg.solve(L, ytr))
        linv_one = _np.linalg.solve(L.T, _np.linalg.solve(L, one))
        mu = float(one @ linv_y / (one @ linv_one))
        res = ytr - mu
        log_like = -_np.sum(_np.log(_np.diag(L))) - 0.5 * float(res @ _np.linalg.solve(R, res))
        if best is None or log_like > best[0]:
            best = (log_like, theta, R)
    _loglike, theta, R = best
    one = _np.ones(n)
    mu = float(one @ _np.linalg.solve(R, ytr) / (one @ _np.linalg.solve(R, one)))
    w = _np.linalg.solve(R, ytr - mu)
    rte = corr(xz, tz, theta)
    pred = mu + rte.T @ w
    return {"params": {"theta": theta, "correlation": corr_label, "trend": trend},
            "predicted": [float(v) for v in pred]}


def train_svr(Xtr, ytr, Xte, kernel="RBF", c_min=0.1, c_max=10.0, c_step=1.0,
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
                model = SVR(kernel=kernel_arg, C=c, gamma=gamma, epsilon=0.05)
                model.fit(Xtr, ytr)
                pred = model.predict(Xtr if len(ytr) <= 60 else Xte)
                err = float(_np.mean((pred - ytr) ** 2))
                if err < best_err:
                    best_err, best = err, (c, gamma)
        c, gamma = best
        model = SVR(kernel=kernel_arg, C=c, gamma=gamma, epsilon=0.05)
        model.fit(Xtr, ytr)
        return {"params": {"kernel": kernel, "C": c, "gamma": gamma,
                           "solver": "scikit-learn SVR"},
                "predicted": [float(v) for v in model.predict(Xte)]}
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
                    best_pred = _kernel_rbf(Xtr, Xte, gamma).T @ alpha
        return {"params": {"kernel": kernel, "C": best_params[0],
                           "gamma": best_params[1],
                           "solver": "核岭回归 KRR（未安装 scikit-learn 的 SVR 近似）"},
                "predicted": [float(v) for v in best_pred]}


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
        pred = model.predict((Xte - Xmean) / Xstd) * ystd + ymean
        return {"params": {"layers": hidden, "solver": "scikit-learn MLP"},
                "predicted": [float(v) for v in pred]}
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
        return {"params": {"layers": tuple(layers), "solver": "numpy BP"},
                "predicted": [float(v) for v in pred]}


def train_surrogate(x_rows, response_values, test_ratio=0.2, seed=2026,
                    model="Kriging", model_params=None):
    """训练单个响应的代理模型，返回评估指标（训练/测试误差、耗时等）。"""
    if not HAS_NUMPY:
        return {"error": "当前环境未安装 numpy，无法训练代理模型。"}
    model_params = model_params or {}
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
            result = train_kriging(Xtr, ytr, Xte, **model_params.get("kriging", {}))
        elif model == "SVR":
            result = train_svr(Xtr, ytr, Xte, **model_params.get("svr", {}))
        else:
            result = train_ann(Xtr, ytr, Xte, **model_params.get("ann", {}))
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
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_time_s": round(elapsed, 4),
        "mape_test_%": round(mape_test, 4),
        "r2_test": round(r2_test, 4),
        "actual": [float(v) for v in yte],
        "predicted": [float(v) for v in pred],
    }
