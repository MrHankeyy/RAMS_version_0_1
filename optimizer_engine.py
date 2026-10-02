"""鲁棒优化与稳定性设计分析引擎。

涵盖：
- 响应模型（二次/线性最小二乘）在任一输入点给出均值与方差（考虑噪声与环境因子
  的不确定性传播 + 残余噪声）；
- 三种鲁棒优化模式：NSGA-II 多目标（均值 vs 稳定性）、加权单目标、约束型
  （把有界限的响应转成独立约束，支持六西格玛退让系数）；
- 帕累托前沿、前沿曲线与 Knee Point；
- 统计工具：整体统计、逐水平均值/方差/极差、F 比、回归方差分析与失拟(LOF)、
  交叉验证、拟合度与残差、参数设计与取值器、容差贡献、模型提取、
  最陡上升 / EVOP / 自适应 AOFAT。
"""
from __future__ import annotations

import itertools
import math
import statistics

import doe_engine as de

try:
    import numpy as _np
    HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    HAS_NUMPY = False


# ============================================================== 回归工具
def _design_terms(k):
    """返回二次模型项与名称。"""
    terms = [("const", None, None)]
    names = ["常数"]
    for i in range(k):
        terms.append(("linear", i, None))
        names.append(f"x{i + 1}")
        terms.append(("quad", i, i))
        names.append(f"x{i + 1}²")
    for i in range(k):
        for j in range(i + 1, k):
            terms.append(("cross", i, j))
            names.append(f"x{i + 1}·x{j + 1}")
    return terms, names


def _design_row(z, terms):
    row = []
    for kind, i, j in terms:
        if kind == "const":
            row.append(1.0)
        elif kind == "linear":
            row.append(z[i])
        elif kind == "quad":
            row.append(z[i] ** 2)
        else:
            row.append(z[i] * z[j])
    return _np.array(row)


def _quad_hess_diag(terms, coef, z):
    """二次模型对角二阶导（对 z）。"""
    out = _np.zeros(len(z))
    for (kind, i, _j), c in zip(terms, coef):
        if kind == "quad":
            out[i] += 2.0 * c
    return out


def _quad_grad(terms, coef, z):
    """二次模型梯度（对 z）。"""
    g = _np.zeros(len(z))
    for (kind, i, j), c in zip(terms, coef):
        if kind == "linear":
            g[i] += c
        elif kind == "quad":
            g[i] += 2.0 * c * z[i]
        elif kind == "cross":
            g[i] += c * z[j]
            g[j] += c * z[i]
    return g


def _fit_model(X, y, full_quadratic=False):
    """拟合 X(输入，原始值) -> y，转为编码 z∈[-1,1] 的二次模型。返回模型 dict。"""
    if not HAS_NUMPY:
        raise RuntimeError("需要 numpy")
    X = _np.asarray(X, float)
    y = _np.asarray(y, float)
    n, k = X.shape
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = (hi - lo) / 2.0
    span[span == 0] = 1.0
    center = (lo + hi) / 2.0
    Z = (X - center) / span
    # 两水平数据（±1）中 x² 与常数共线，仅当因子有≥3个不同值才保留平方项
    distinct = [_np.unique(X[:, j]).size for j in range(k)]
    terms = [("const", None, None)]
    names = ["常数"]
    for i in range(k):
        terms.append(("linear", i, None))
        names.append(f"x{i + 1}")
        if full_quadratic or distinct[i] >= 3:
            terms.append(("quad", i, i))
            names.append(f"x{i + 1}²")
    for i in range(k):
        for j in range(i + 1, k):
            terms.append(("cross", i, j))
            names.append(f"x{i + 1}·x{j + 1}")
    A = _np.column_stack([_design_row(z, terms) for z in Z]).T
    nterms = len(terms)
    if n < nterms and not full_quadratic:
        # 点数不足时降为线性模型
        keep = [i for i, (kind, *_r) in enumerate(terms) if kind in ("const", "linear")]
        terms = [terms[i] for i in keep]
        names = [names[i] for i in keep]
        A = A[:, keep]
        nterms = len(keep)
    rank = int(_np.linalg.matrix_rank(A))
    if rank < nterms:
        raise ValueError(f"模型矩阵秩不足（{rank}/{nterms}）；无法独立估计全部系数，请补充有效设计点。")
    coef, *_ = _np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    ss_tot = float(_np.sum((y - y.mean()) ** 2)) or 1e-12
    ss_res = float(_np.sum((y - y_hat) ** 2))
    df_model = nterms - 1
    df_resid = n - nterms
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - nterms, 1)
    rmse = math.sqrt(ss_res / max(n, 1))
    residual = y - y_hat
    resid_std = math.sqrt(ss_res / max(df_resid, 1)) if df_resid > 0 else rmse
    return {
        "terms": terms, "names": names, "coef": coef,
        "full_quadratic": full_quadratic,
        "center": center, "half": span, "lo": lo, "hi": hi,
        "r2": round(r2, 6), "adj_r2": round(adj_r2, 6), "rmse": round(rmse, 6),
        "df_model": df_model, "df_residual": df_resid,
        "residual_std": resid_std, "residual": list(float(v) for v in residual),
        "predicted": list(float(v) for v in y_hat), "n": n, "k": k,
    }


def model_predict(model, x_raw):
    if not HAS_NUMPY:
        return None
    if "predict_fn" in model:
        return float(model["predict_fn"](_np.asarray(x_raw, float)))
    z = (_np.asarray(x_raw, float) - model["center"]) / model["half"]
    row = _design_row(z, model["terms"])
    return float(row @ model["coef"])


def model_grad_x(model, x_raw):
    """梯度 wrt 原始单位 x。"""
    if "grad_fn" in model:
        return _np.asarray(model["grad_fn"](_np.asarray(x_raw, float)), float)
    if "predict_fn" in model:
        x = _np.asarray(x_raw, float)
        step = float(model.get("finite_difference_step", 1e-4))
        grad = _np.zeros(len(x))
        for i in range(len(x)):
            delta = _np.zeros(len(x)); delta[i] = step
            grad[i] = (model["predict_fn"](x + delta)
                       - model["predict_fn"](x - delta)) / (2.0 * step)
        return grad
    z = (_np.asarray(x_raw, float) - model["center"]) / model["half"]
    gz = _quad_grad(model["terms"], model["coef"], z)
    return gz / model["half"]


def model_hess_diag_x(model, x_raw):
    if "hess_diag_fn" in model:
        return _np.asarray(model["hess_diag_fn"](_np.asarray(x_raw, float)), float)
    if "predict_fn" in model:
        x = _np.asarray(x_raw, float)
        step = float(model.get("finite_difference_step", 1e-4))
        center = model["predict_fn"](x)
        hess = _np.zeros(len(x))
        for i in range(len(x)):
            delta = _np.zeros(len(x)); delta[i] = step
            hess[i] = (model["predict_fn"](x + delta)
                       - 2.0 * center + model["predict_fn"](x - delta)) / step ** 2
        return hess
    z = (_np.asarray(x_raw, float) - model["center"]) / model["half"]
    hz = _quad_hess_diag(model["terms"], model["coef"], z)
    # 对 z 的二阶导转换：∂²f/∂x² = (1/h²)·∂²f/∂z²
    return hz / (model["half"] ** 2)


def model_equation(model):
    terms = model["terms"]
    labels = model["names"]
    coef = model["coef"]
    parts = []
    for t, label, c in zip(terms, labels, coef):
        parts.append((float(c), label))
    s = []
    for i, (c, label) in enumerate(parts):
        sign = "+" if c >= 0 else "-"
        mag = abs(c)
        body = f"{mag:.5g}·{label}" if label != "常数" else f"{mag:.5g}"
        s.append(f" {sign} {body}" if i else (f"{body}" if c >= 0 else f"-{body}"))
    return "y = " + "".join(s).strip()


# ============================================================== 数据准备
def build_models(project):
    """训练每个响应的二次模型（输入为所有非固定因子）。返回 (context, errors)。"""
    matrix = project.doe_matrix or []
    if not matrix:
        return None, ["尚无 DOE 试验数据，请先在【方案配置】生成方案。"]
    is_rsm = "响应曲面" in project.design_method
    inputs = [f for f in project.factors
              if f.name and not f.is_fixed and (not is_rsm or f.source == "设计")]
    names = [f.name for f in inputs]
    if not names:
        return None, ["没有可建模的输入因子。"]
    # 只使用矩阵中实际存在的输入列
    keys = set(matrix[0].keys())
    if is_rsm and any(n not in keys for n in names):
        return None, ["DOE 缺少设计因子列，不能删去因子后继续拟合。"]
    names = [n for n in names if n in keys]
    inputs = [f for f in inputs if f.name in names]
    if not names:
        return None, ["DOE 矩阵中未找到输入因子列。"]

    context = {"inputs": inputs, "names": names, "models": {},
               "errors": [], "_responses": project.responses,
               "_project": project}
    for resp in project.responses:
        if not resp.name:
            continue
        X, y = [], []
        for row in matrix:
            try:
                _ = [float(row[n]) for n in names]
            except (KeyError, TypeError, ValueError):
                continue
            try:
                yv = float(row.get(resp.name))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in _ + [yv]):
                continue
            X.append([float(row[n]) for n in names])
            y.append(yv)
        if len(y) < 3:
            context["errors"].append(f"响应 {resp.name} 有效样本 {len(y)} < 3，跳过建模。")
            continue
        if is_rsm and len(y) != len(matrix):
            context["errors"].append(f"响应 {resp.name} 存在缺失/非有限数据，请补齐后拟合完整二次响应面。")
            continue
        try:
            context["models"][resp.name] = _fit_model(X, y, full_quadratic=is_rsm)
        except ValueError as exc:
            context["errors"].append(f"响应 {resp.name}：{exc}")
    if not context["models"]:
        return context, context["errors"] or ["没有足够的响应数据可建模。"]
    return context, context["errors"]


def _sigma_of_factor(f, k_design):
    """因子不确定性标准差。"""
    lo, hi = de.factor_limits(f)
    if f.uncertainty == "概率":
        try:
            return max(0.0, float(f.param2)) ** 0.5
        except (TypeError, ValueError):
            return (hi - lo) / 6.0
    if f.source == "环境":
        if f.uncertainty == "概率":
            try:
                return float(f.param2) ** 0.5
            except (TypeError, ValueError):
                return (hi - lo) / math.sqrt(12)
        return (hi - lo) / math.sqrt(12)
    # 设计因子按六西格玛解释：±k·σ 覆盖区间
    return (hi - lo) / (2.0 * max(1.0, k_design))


def robust_moments(model, x_design, context, k_design, include_residual=True):
    """在设计点 x_design(设计因子名->值) 处计算任一响应的均值 μ 与标准差 σ。"""
    inputs = context["inputs"]
    names = context["names"]
    x = []
    for f in inputs:
        if f.source == "环境":
            # 噪声因子取均值点
            lo, hi = de.factor_limits(f)
            if f.uncertainty == "概率" and f.param1:
                x.append(float(f.param1))
            else:
                x.append((lo + hi) / 2.0)
        else:
            lo, hi = de.factor_limits(f)
            x.append(float(x_design.get(f.name, (lo + hi) / 2.0)))
    x = _np.array(x, float)
    mu = model_predict(model, x)
    grad = model_grad_x(model, x)
    hess = model_hess_diag_x(model, x)
    sigma = []
    for f, idx in zip(inputs, range(len(inputs))):
        sigma.append(_sigma_of_factor(f, k_design))
    sigma = _np.array(sigma, float)
    # 一阶 + 二阶矩修正
    curvature = 0.5 * float(_np.sum((hess * sigma ** 2)))
    mu_mean = mu + curvature
    var1 = float(_np.sum((grad * sigma) ** 2))
    var2 = float(_np.sum(0.5 * ((hess * sigma ** 2) ** 2)))
    var_res = model["residual_std"] ** 2 if include_residual else 0.0
    sigma_out = math.sqrt(var1 + var2 + var_res)
    return mu_mean, sigma_out


# ============================================================== 望性/目标
def _response_ref(resp):
    """返回响应参考界限 (上限/目标)。"""
    if resp.feature == "望目":
        try:
            return float(resp.lower), float(resp.upper)
        except (TypeError, ValueError):
            return None, None
    try:
        return float(resp.robust_limit)
    except (TypeError, ValueError):
        return None


def _desirability(resp, mu, data_min, data_max):
    """期望值转换为 0~1 的望性值 d（越大越好）。"""
    if resp.feature == "望大":
        ref = _response_ref(resp)
        lo = ref if ref is not None else min(data_min, mu)
        hi = max(data_max, ref if ref is not None else mu)
        if hi <= lo:
            hi = lo + 1.0
        return max(0.0, min(1.0, (mu - lo) / (hi - lo)))
    if resp.feature == "望小":
        ref = _response_ref(resp)
        hi = ref if ref is not None else max(data_max, mu)
        lo = min(data_min, ref if ref is not None else mu)
        if hi <= lo:
            hi = lo + 1.0
        return max(0.0, min(1.0, (hi - mu) / (hi - lo)))
    ref = _response_ref(resp)
    if not isinstance(ref, tuple) or len(ref) != 2:
        return 0.5
    lo, hi = ref
    if lo is None or hi is None:
        return 0.5
    target = (lo + hi) / 2.0
    tol = (hi - lo) / 2.0
    if tol <= 0:
        tol = 1.0
    return max(0.0, 1.0 - abs(mu - target) / tol)


def _constraint_limits(resp):
    """约束型响应的界限信息：(类型, 界限)。"""
    if resp.feature == "望小":
        ref = _response_ref(resp)
        return ("upper", ref) if ref is not None else ("upper", None)
    if resp.feature == "望大":
        ref = _response_ref(resp)
        return ("lower", ref) if ref is not None else ("lower", None)
    ref = _response_ref(resp)
    if not isinstance(ref, tuple) or len(ref) != 2:
        return ("both", (None, None))
    lo, hi = ref
    return ("both", (lo, hi)) if lo is not None and hi is not None else ("both", (None, None))


def _response_scale(model):
    """按响应数据幅度归一化，不让拟合精度改变优化目标的权重。"""
    values = [float(v) for v in model.get("predicted", []) if _isfinite(v)]
    if not values:
        return 1.0
    span = max(values) - min(values)
    return span if span > 1e-12 else max(max(abs(v) for v in values), 1.0)


def evaluate_design(context, x_design, k_design, weight_r, use_constraints=False,
                    k_constraint=6.0, resp_stds=None):
    """设计点评估。返回 dict{obj, mean_perf, std_norm, infeas, details}。"""
    models = context["models"]
    resp_objs, resp_cons, details = [], [], []
    data_min, data_max = {}, {}
    for name, m in models.items():
        data_min[name] = float(_np.min(m["predicted"])) if m["predicted"] else 0.0
        data_max[name] = float(_np.max(m["predicted"])) if m["predicted"] else 1.0
    std_sum, perf_sum, wsum = 0.0, 0.0, 0.0
    infeas = 0.0
    sigma_limits = context.get("sigma_limits", {})
    for resp in project_responses(context):
        m = models.get(resp.name)
        if m is None:
            continue
        mu, sigma = robust_moments(m, x_design, context, k_design)
        if resp.name in sigma_limits:
            bound = sigma_limits[resp.name]
            infeas += max(0.0, sigma - bound) / max(abs(bound), 1e-12)
            details.append({"name": resp.name, "sigma": sigma, "sigma_limit": bound, "margin": bound - sigma})
        d = _desirability(resp, mu, data_min[resp.name], data_max[resp.name])
        scale = _response_scale(m)
        std_n = sigma / max(scale, 1e-12)
        # 是否约束
        is_cons = self_is_constraint(resp)
        if is_cons:
            kind, lim = _constraint_limits(resp)
            if use_constraints and (lim is None or
                                    (kind == "both" and None in lim)):
                raise ValueError(f"约束响应 {resp.name} 未设置有效的约束界限。")
            if kind == "upper" and lim is not None:
                infeas_in = max(0.0, mu + k_constraint * sigma - lim)
            elif kind == "lower" and lim is not None:
                infeas_in = max(0.0, lim - (mu - k_constraint * sigma))
            elif kind == "both":
                lo, hi = lim
                infeas_in = 0.0
                if lo is not None:
                    infeas_in += max(0.0, lo - (mu - k_constraint * sigma))
                if hi is not None:
                    infeas_in += max(0.0, (mu + k_constraint * sigma) - hi)
            else:
                infeas_in = 0.0
            limit_scale = abs(lim if isinstance(lim, (int, float))
                              else ((lim[1] or 0.0) if lim else 0.0))
            if use_constraints:
                infeas += infeas_in / max(limit_scale, 1.0)
            resp_cons.append({"name": resp.name, "mu": mu, "sigma": sigma, "d": d})
        if not is_cons or "目标" in resp.kind:
            resp_objs.append({"name": resp.name, "mu": mu, "sigma": sigma, "d": d})
            perf_sum += d
            std_sum += std_n
            wsum += 1.0
    if not resp_objs:
        # 全为约束时，用第一个响应的 σ 作为目标并保持可行
        perf_sum = 1.0
        std_sum = sum(c["sigma"] / max(_response_scale(models[c["name"]]), 1e-12) for c in resp_cons) if resp_cons else 0.0
        wsum = len(resp_cons) or 1
    mean_perf = perf_sum / max(wsum, 1)
    std_norm = std_sum / max(wsum, 1)
    # 目标（最小化）
    obj = [1.0 - mean_perf, std_norm]
    if infeas > 0:
        obj = [v + 100.0 * infeas for v in obj]
    return {"obj": obj, "mean_perf": mean_perf, "std_norm": std_norm,
            "infeas": infeas, "objectives": resp_objs, "constraints": resp_cons,
            "details": details, "x": dict(x_design)}


def project_responses(context):
    return context["_responses"]


def self_is_constraint(resp):
    # 稳定性阈值属于目标的望性参考，只有用户明确勾选“约束响应”
    # 时才把该响应从目标集合移入约束集合。
    return "约束" in getattr(resp, "kind", "")


# ============================================================== NSGA-II
def _dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _fast_non_dominated_sort(fitness):
    n = len(fitness)
    dominates_count = [0] * n
    dominated = [[] for _ in range(n)]
    ranks = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(fitness[i], fitness[j]):
                dominated[i].append(j)
                dominates_count[j] += 1
            elif _dominates(fitness[j], fitness[i]):
                dominated[j].append(i)
                dominates_count[i] += 1
        if dominates_count[i] == 0:
            ranks[0].append(i)
    front = [ranks[0]]
    k = 0
    while True:
        nxt = []
        for i in front[k]:
            for j in dominated[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    nxt.append(j)
        if not nxt:
            break
        front.append(nxt)
        k = k + 1
    return front


def _crowding_distance(fitness, front):
    n_obj = len(fitness[0])
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    for m in range(n_obj):
        front_sorted = sorted(front, key=lambda i: fitness[i][m])
        lo = fitness[front_sorted[0]][m]
        hi = fitness[front_sorted[-1]][m]
        span = (hi - lo) or 1.0
        dist[front_sorted[0]] = dist.get(front_sorted[0], 0) + 1e9
        dist[front_sorted[-1]] = dist.get(front_sorted[-1], 0) + 1e9
        for a, b in zip(front_sorted, front_sorted[1:]):
            dist[a] = dist.get(a, 0) + (fitness[b][m] - fitness[a][m]) / span
    return dist


def _sbx_crossover(p1, p2, eta, lo, hi, rng):
    c1, c2 = list(p1), list(p2)
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-12:
            continue
        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    c1 = [min(hi[i], max(lo[i], v)) for i, v in enumerate(c1)]
    c2 = [min(hi[i], max(lo[i], v)) for i, v in enumerate(c2)]
    return c1, c2


def _polynomial_mutation(x, eta, pm, lo, hi, rng):
    out = list(x)
    for i in range(len(x)):
        if rng.random() > pm:
            continue
        u = rng.random()
        delta = (2.0 * u) ** (1.0 / (eta + 1)) - 1.0 if u < 0.5 else \
            1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1))
        out[i] = min(hi[i], max(lo[i], x[i] + delta * (hi[i] - lo[i])))
    return out


def nsga2(evaluate, lo, hi, pop=60, gen=100, seed=1, eta_c=15.0, eta_m=20.0, progress=None):
    """Elitist NSGA-II with feasibility-first survival and tournament selection."""
    if pop < 2 or gen < 1:
        raise ValueError("种群至少为 2，迭代代数至少为 1。")
    rng = _np.random.default_rng(seed)
    n = len(lo)
    evaluated = {}
    def assess(v):
        key = tuple(float(x) for x in v)
        if key not in evaluated:
            evaluated[key] = evaluate(v)
        return evaluated[key]
    def order(population):
        values = [assess(v) for v in population]
        feasible = [i for i, (_, info) in enumerate(values) if info.get("r", {}).get("infeas", 0) <= 1e-8]
        invalid = [i for i in range(len(values)) if i not in feasible]
        rank = {}
        distance = {}
        fronts = _fast_non_dominated_sort([values[i][0] for i in feasible])
        for r, front in enumerate(fronts):
            if not front:
                continue
            crowd = _crowding_distance([values[i][0] for i in feasible], front)
            for j in front:
                rank[feasible[j]] = r
                distance[feasible[j]] = crowd[j]
        keys = {i:(0, rank[i], -distance[i]) for i in feasible}
        keys.update({i:(1, values[i][1].get("r", {}).get("infeas", 0), sum(values[i][0])) for i in invalid})
        return sorted(range(len(population)), key=keys.get), keys
    population = [[rng.uniform(lo[i], hi[i]) for i in range(n)] for _ in range(pop)]
    for generation in range(gen):
        _, keys = order(population)
        def parent():
            a, b = rng.choice(len(population), 2, replace=False)
            return population[a if keys[a] <= keys[b] else b]
        children = []
        while len(children) < pop:
            c1, c2 = _sbx_crossover(parent(), parent(), eta_c, lo, hi, rng)
            children.extend([_polynomial_mutation(c, eta_m, 1.0/n, lo, hi, rng) for c in (c1,c2)])
        combined = population + children[:pop]
        indices, _ = order(combined)
        population = [combined[i] for i in indices[:pop]]
        if progress and (generation == 0 or (generation + 1) % max(1, gen//20) == 0 or generation + 1 == gen):
            progress(generation + 1, gen)
    return population, [(key, tuple(value[0])) for key, value in evaluated.items()]


# ============================================================== 鲁棒优化
def design_inputs(context):
    return [f for f in context["inputs"] if f.source == "设计"]


def _pareto(sorted_pairs):
    """从 [(vec, objs, info)] 中筛出非支配解并排序。"""
    filtered = []
    seen = set()
    for item in sorted_pairs:
        key = tuple(item[1])
        if key in seen or any(_dominates(other[1], item[1]) for other in filtered):
            continue
        filtered = [other for other in filtered if not _dominates(item[1], other[1])]
        filtered.append(item)
        seen.add(key)
    filtered.sort(key=lambda t: t[1][0])
    return filtered


def _knee_point(front):
    if not front:
        return None
    f1 = [obj[0] for _, obj, _ in front]
    f2 = [obj[1] for _, obj, _ in front]
    m1, M1 = min(f1), max(f1)
    m2, M2 = min(f2), max(f2)
    span1 = (M1 - m1) or 1e-9
    span2 = (M2 - m2) or 1e-9
    best, score = None, float("inf")
    for item, o1, o2 in zip(front, f1, f2):
        s = (o1 - m1) / span1 + (o2 - m2) / span2
        # 两个极端点的归一化和均为 1；平分时优先稳定性，不能由前沿排序决定。
        if s < score - 1e-12 or (abs(s - score) <= 1e-12 and
                                (best is None or o2 < best[1][1])):
            score, best = s, item
    return best


def robust_optimize(context, mode="multi", pop=60, gen=100, weight=0.5,
                    k_design=6.0, k_constraint=6.0, seed=1, sigma_limits=None, progress=None):
    """执行鲁棒优化，返回结果 dict。"""
    if mode not in ("multi", "weighted", "constraint"):
        raise ValueError("未知优化模式")
    context = dict(context)
    context.pop("sigma_limits", None)
    if mode == "constraint":
        objective_names = {r.name for r in project_responses(context) if not self_is_constraint(r) or "目标" in r.kind}
        if not objective_names:
            raise ValueError("独立约束模式需要至少一个目标响应。")
        limits = sigma_limits or {}
        if set(limits) != objective_names:
            raise ValueError("请为每个目标响应设置独立标准差上限：" + "、".join(sorted(objective_names)))
        limits = {name: float(value) for name, value in limits.items()}
        if any(not _np.isfinite(value) or value <= 0 for value in limits.values()):
            raise ValueError("标准差上限必须是有限正数，单位与响应一致。")
        context["sigma_limits"] = limits
    dinputs = design_inputs(context)
    if not dinputs:
        return {"error": "没有可优化的设计因子，请先在【业务建模】添加设计因子。"}
    dnames = [f.name for f in dinputs]
    lo = [de.factor_limits(f)[0] for f in dinputs]
    hi = [de.factor_limits(f)[1] for f in dinputs]
    # 响应角色来自业务建模，不能因选择 NSGA-II / 加权模式而把约束变成目标。
    use_cons = any(self_is_constraint(resp) for resp in project_responses(context))

    def evaluate(vec):
        xd = dict(zip(dnames, vec))
        r = evaluate_design(context, xd, k_design, weight, use_constraints=use_cons,
                            k_constraint=k_constraint)
        if mode == "weighted":
            obj = [weight * r["obj"][0] + (1.0 - weight) * r["obj"][1]]
        elif mode == "constraint":
            # Keep improving the mean even beyond the sampled response range;
            # clipped desirability would create false equal-optimum plateaus.
            means = {item["name"]: item["mu"] for item in r["objectives"]}
            losses = []
            for response in project_responses(context):
                if response.name not in means:
                    continue
                mu = means[response.name]
                scale = max(_response_scale(context["models"][response.name]), 1e-12)
                if response.feature == "望目":
                    ref = _response_ref(response)
                    if not isinstance(ref, tuple) or None in ref:
                        raise ValueError(f"{response.name} 缺少望目上下界。")
                    loss = abs(mu - (ref[0] + ref[1])/2)/scale
                else:
                    loss = (-mu if response.feature == "望大" else mu)/scale
                losses.append(loss)
            obj = [sum(losses)/len(losses)]
        else:
            obj = list(r["obj"])
        return obj, {"x": xd, "r": r}

    population, evaluated = nsga2(evaluate, lo, hi, pop=pop, gen=gen, seed=seed, progress=progress)
    pairs = []
    for vec_tuple, _obj in evaluated:
        vec = list(vec_tuple)
        obj, info = evaluate(vec)
        pairs.append((vec, obj, info))
    feasible = [t for t in pairs if t[2]["r"]["infeas"] <= 1e-8]
    # 最终推荐与前沿优先使用可行解，不能让低目标值抵消约束违反。
    candidates = feasible or pairs
    front_full = _pareto(feasible)

    # 选择推荐最优解
    if mode == "weighted":
        best = min(candidates, key=lambda t: t[1][0])
    elif mode == "constraint":
        best = min(candidates, key=lambda t: sum(t[1]))
    else:
        best = _knee_point(front_full) or min(candidates, key=lambda t: sum(t[1]))
    if not feasible:
        best = min(pairs, key=lambda t: (t[2]["r"]["infeas"], sum(t[1])))

    knee = _knee_point(front_full) if mode == "multi" else None
    recommended = best
    x_rec = recommended[2]["x"]
    r_rec = recommended[2]["r"]
    resp_detail = []
    objective_names = {item["name"] for item in r_rec["objectives"]}
    constraint_names = {item["name"] for item in r_rec["constraints"]}
    seen_responses = set()
    for item in r_rec["objectives"] + r_rec["constraints"]:
        if item["name"] in seen_responses:
            continue
        seen_responses.add(item["name"])
        role = ("目标+约束" if item["name"] in objective_names and item["name"] in constraint_names
                else "约束" if item["name"] in constraint_names else "目标")
        resp_detail.append({"name": item["name"], "mu": item["mu"],
                            "sigma": item["sigma"], "d": round(item["d"], 4), "role": role})

    result = {
        "sigma_limits": context.get("sigma_limits", {}),
        "population_size": pop, "generations": gen, "seed": seed,
        "mode": mode, "weight": weight, "k_design": k_design,
        "k_constraint": k_constraint, "design_names": dnames,
        "input_sigmas": {f.name: _sigma_of_factor(f, k_design) for f in context["inputs"]},
        "objective_names": sorted(objective_names), "constraint_names": sorted(constraint_names),
        "feasible_count": len(feasible),
        "front": [{"x": _info["x"], "obj": [round(float(o), 6) for o in _obj],
                   "infeas": _info["r"]["infeas"]}
                  for _vec, _obj, _info in front_full],
        "front_scatter": [[float(t[1][0]), float(t[1][1] if len(t[1]) > 1 else 0.0)]
                          for t in feasible if len(t[1]) > 1],
        "knee": ({"x": knee[2]["x"], "obj": [round(float(o), 6) for o in knee[1]],
                  "infeas": knee[2]["r"]["infeas"]} if knee else None),
        "best": {"x": x_rec, "obj": [round(float(o), 6) for o in recommended[1]],
                 "mean_perf": round(r_rec["mean_perf"], 4),
                 "std_norm": round(r_rec["std_norm"], 4),
                 "infeas": r_rec["infeas"],
                 "sigma_constraints": r_rec["details"],
                 "responses": resp_detail},
        "evaluated_count": len(pairs),
    }
    result["text"] = _robust_result_text(result)
    if mode == "constraint":
        result["text"] += "\n独立标准差约束（响应原单位）：" + "；".join(f"{n}: σ ≤ {v:g}" for n,v in context["sigma_limits"].items())
    return result


def _robust_result_text(res):
    mode_names = {"multi": "NSGA-II 多目标（均值 vs 稳定性）",
                  "weighted": "加权单目标（权重可设）",
                  "constraint": "约束型（稳定性转为独立约束）"}
    lines = [f"【{mode_names.get(res['mode'], res['mode'])}】",
             f"约束安全系数 k={res['k_constraint']}，输入不确定性系数 k={res['k_design']}",
             "输入标准差：" + "，".join(f"{n}={s:.6g}" for n, s in res.get("input_sigmas", {}).items()),
             "目标响应：" + "、".join(res.get("objective_names", [])),
             "约束响应：" + ("、".join(res.get("constraint_names", [])) or "无"),
             "稳定性按响应数据极差归一化；均值和方差采用局部二阶近似。"]
    if res["k_design"] < 3:
        lines.append("提示：区间因子的输入扰动较大，局部二阶近似可能失真，建议用扰动抽样复核。")
    if res["mode"] == "weighted":
        lines.append(f"权重 λ(目标均值) = {res['weight']}，稳定性权重 = {round(1 - res['weight'], 2)}")
    lines.append(f"评估解数量：{res['evaluated_count']}")
    lines.append(f"可行解数量：{res.get('feasible_count', 0)}；图中只显示可行解，不把约束罚项作为响应波动。")
    lines.append("")
    if res["mode"] == "multi" and res["front"]:
        lines.append(f"帕累托前沿解数量：{len(res['front'])}")
        knee = res.get("knee")
        if knee:
            lines.append(f"Knee Point：{knee['x']} ｜ 目标值 {knee['obj']}")
    best = res["best"]
    lines.append("")
    lines.append("★ 推荐优化设计（稳健设计点）" if res.get("feasible_count", 0)
                 else "尚未找到满足约束的设计；以下仅列出违例最小的候选点。")
    lines.append(f"  因子设置：{ {k: round(v, 5) for k, v in best['x'].items()} }")
    lines.append(f"  综合望性均值 {best['mean_perf']} ｜ 归一化稳定性 {best['std_norm']} ｜ "
                 f"约束违例 {best['infeas']}")
    lines.append("  各响应（扰动后近似均值 μ、标准差 σ、望性 d）：")
    for item in best["responses"]:
        lines.append(f"    {item['name']}（{item.get('role', '目标')}）: μ={item['mu']:.4g}, σ={item['sigma']:.4g}, d={item['d']}")
    return "\n".join(lines)


# ============================================================== 统计工具集
def _gammaln(x):
    return math.lgamma(x)


def _betacf(a, b, x):
    MAXIT = 200
    EPS = 3.0e-12
    FPMIN = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(_gammaln(a + b) - _gammaln(a) - _gammaln(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _f_pvalue(F, df1, df2):
    x = df2 / (df2 + df1 * F)
    return _betai(df2 / 2.0, df1 / 2.0, x)


def describe_data(project):
    matrix = project.doe_matrix or []
    if not matrix:
        return "尚无试验数据。"
    lines = ["【整体数据分析】"]
    n = len(matrix)
    for resp in project.responses:
        if not resp.name:
            continue
        vals = []
        for row in matrix:
            try:
                if _isfinite(row.get(resp.name)):
                    vals.append(float(row[resp.name]))
            except (TypeError, ValueError):
                continue
        if not vals:
            lines.append(f"  {resp.name}: 无有效数据")
            continue
        vals.sort()
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals)>1 else float("nan")
        lines.append(f"  响应 {resp.name}: n={len(vals)}/{n} 均值={mean:.4g} "
                     f"样本标准差={std:.4g} 最小={vals[0]:.4g} 最大={vals[-1]:.4g} "
                     f"极差={vals[-1] - vals[0]:.4g}")
    return "\n".join(lines)


def level_stats(project):
    matrix = project.doe_matrix or []
    factors = [f for f in project.factors if f.name and not (f.is_fixed and f.fixed_value not in (None, ""))]
    headers = ["因子", "水平", "响应", "均值", "方差", "极差", "样本数", "F比", "p值"]
    rows = []
    if not matrix:
        return {"headers": headers, "rows": rows}
    for f in factors:
        if f.name not in matrix[0]:
            continue
        level_values = sorted({float(row[f.name]) for row in matrix if _isfinite(row.get(f.name))})
        for resp in project.responses:
            if not resp.name:
                continue
            groups = []
            for lv in level_values:
                vals = []
                for row in matrix:
                    try:
                        if abs(float(row[f.name]) - lv) < 1e-9:
                            vals.append(float(row.get(resp.name)))
                    except (TypeError, ValueError):
                        continue
                groups.append((lv, [v for v in vals if _isfinite(v)]))
            groups = [(lv, v) for lv, v in groups if v]
            if not groups:
                continue
            grand = [x for _lv, v in groups for x in v]
            g = len(groups)
            n = len(grand)
            grand_mean = statistics.mean(grand)
            ss_between = sum(len(v) * (statistics.mean(v) - grand_mean) ** 2 for _lv, v in groups)
            ss_within = sum(sum((x - statistics.mean(v)) ** 2 for x in v) for _lv, v in groups)
            df_b = g - 1
            df_w = n - g
            fstat = (ss_between / df_b) / (ss_within / df_w) if ss_within > 0 and df_w > 0 and df_b > 0 else float("nan")
            p = _f_pvalue(fstat, df_b, df_w) if math.isfinite(fstat) else float("nan")
            for lv, v in groups:
                mean = statistics.mean(v)
                var = statistics.variance(v) if len(v) > 1 else float("nan")
                rng = max(v) - min(v)
                rows.append({
                    "factor": f.name, "level": lv, "response": resp.name,
                    "mean": mean, "var": var, "range": rng, "n": len(v),
                    "F": fstat, "p": p,
                })
    return {"headers": headers, "rows": rows}


def _isfinite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def anova_and_lof(context, project):
    if not context or not context["models"]:
        return "无模型可分析，请先填入响应数据。"
    matrix = project.doe_matrix or []
    names = context["names"]
    lines = ["【回归方差分析 / 显著性 / 失拟(LOF)】"]
    for resp_name, m in context["models"].items():
        if "predict_fn" in m:
            lines.append(f"◆ 响应 {resp_name}：代理模型不适用多项式回归 ANOVA/LOF；请查看留出集验证与残差。")
            continue
        n = m["n"]
        # 从残差计算
        y = [float(row[resp_name]) for row in matrix if row.get(resp_name) not in (None, "") and all(_isfinite(row[c]) for c in names)]
        if len(y) != n:
            continue
        y_mean = statistics.mean(y)
        ss_tot = sum((v - y_mean) ** 2 for v in y)
        ss_res = sum((v - p) ** 2 for v, p in zip(y, m["predicted"]))
        ss_reg = ss_tot - ss_res
        df_reg = m["df_model"]
        df_res = m["df_residual"]
        ms_reg = ss_reg / max(df_reg, 1)
        ms_res = ss_res / max(df_res, 1)
        fstat = ms_reg / ms_res if ms_res > 0 else 0.0
        p = _f_pvalue(fstat, df_reg, df_res)
        lines.append(f"◆ 响应 {resp_name}")
        lines.append(f"  回归：F={fstat:.4g}，df=({df_reg},{df_res})，p={p:.4g}，"
                     f"R²={m['r2']}，调整R²={m['adj_r2']}，RMSE={m['rmse']}")
        if df_res <= 0 or ms_res <= 1e-24:
            lines[-1] = f"  回归：R²={m['r2']}，RMSE={m['rmse']}；残差自由度不足或误差为零，F/p 不可估计。"
        # 失拟：从重复设计点估计纯误差
        pure_ss, pure_df = 0, 0
        point_groups = {}
        for row in matrix:
            try:
                pt = tuple(row.get(c) for c in names)
                yv = float(row.get(resp_name))
            except (TypeError, ValueError):
                continue
            if not all(_isfinite(v) for v in pt):
                continue
            point_groups.setdefault(pt, []).append(yv)
        for pt, vals in point_groups.items():
            if len(vals) > 1:
                gm = statistics.mean(vals)
                pure_ss += sum((v - gm) ** 2 for v in vals)
                pure_df += len(vals) - 1
        if pure_df > 0 and df_res > pure_df:
            lof_df = df_res - pure_df
            lof_ss = max(ss_res - pure_ss, 0.0)
            ms_pe = pure_ss / pure_df
            ms_lof = lof_ss / lof_df
            if ms_pe <= 1e-24:
                lines.append("  失拟(LOF)：重复点纯误差为零，F/p 不可估计；不能据此判定模型充分。")
                continue
            lof_f = ms_lof / ms_pe
            lof_p = _f_pvalue(lof_f, lof_df, pure_df)
            lines.append(f"  失拟(LOF)：SS_Lof={lof_ss:.4g}，SS_PE={pure_ss:.4g}，"
                         f"F_Lof={lof_f:.4g}，p={lof_p:.4g}，df=({lof_df},{pure_df})")
            lines.append("  判定：" + ("未检出显著失拟，不等于证明模型充分。" if lof_p > 0.05
                                      else "模型存在失拟，建议改进模型结构或补点。"))
        else:
            lines.append("  失拟(LOF)：无重复点/中心点，无法估计纯误差，跳过 LOF。")
        lines.append("")
    return "\n".join(lines)


def cross_validation(context):
    if not context or not context["models"] or not HAS_NUMPY:
        return "无模型可交叉验证。"
    lines = ["【交互验证（交叉验证）】"]
    for resp_name, m in context["models"].items():
        if "predict_fn" in m:
            validation = m.get("validation", {})
            lines.append(f"  响应 {resp_name}（{m.get('model_type', '代理模型')}）："
                         f"留出集验证，测试样本 {validation.get('test_samples', '未知')}，"
                         f"R²={validation.get('r2_test', '未知')}，RMSE={m['rmse']:.4g}；未执行 K 折验证。")
            continue
        X, y = _matrix_xy(context, resp_name)
        if len(y) < 4:
            lines.append(f"  {resp_name}: 样本不足")
            continue
        kfold = min(5, len(y))
        rng = _np.random.default_rng(123)
        idx = rng.permutation(len(y))
        splits = _np.array_split(idx, kfold)
        preds = _np.full(len(y), _np.nan)
        for test_idx in splits:
            train_idx = _np.setdiff1d(_np.arange(len(y)), test_idx)
            if len(train_idx) < 3:
                continue
            try:
                m_fold = _fit_model(X[train_idx], y[train_idx], full_quadratic=m.get("full_quadratic", False))
                for i in test_idx:
                    preds[i] = model_predict(m_fold, X[i])
            except Exception:
                continue
        if not _np.all(_np.isfinite(preds)):
            lines.append(f"  {resp_name}: 分折后样本不足或设计矩阵秩不足，无法验证同一模型；未降阶替代。")
            continue
        actual = y
        ss_tot = float(_np.sum((actual - actual.mean()) ** 2)) + 1e-12
        ss_res = float(_np.sum((actual - preds) ** 2))
        cv_r2 = 1 - ss_res / ss_tot
        cv_rmse = math.sqrt(ss_res / len(y))
        lines.append(f"  响应 {resp_name}: {kfold} 折 CV-RMSE={cv_rmse:.4g}, CV-R²={cv_r2:.4g}")
    return "\n".join(lines)


def _matrix_xy(context, resp_name):
    project = context["_project"]
    names = context["names"]
    X, y = [], []
    for row in (project.doe_matrix or []):
        try:
            xv = [float(row[c]) for c in names]
            yv = float(row[resp_name])
        except (KeyError, TypeError, ValueError):
            continue
        X.append(xv)
        y.append(yv)
    return _np.array(X, float), _np.array(y, float)


def residual_analysis(context):
    if not context or not context["models"]:
        return "无模型可做残差分析。", None
    lines = ["【拟合度与残差分析】"]
    data = {}
    for resp_name, m in context["models"].items():
        if "predict_fn" in m:
            validation = m.get("validation", {})
            predicted = validation.get("predicted", [])
            residuals = [a - p for a, p in zip(validation.get("actual", []), predicted)]
            lines.append(f"  {resp_name}：留出集残差，R²={validation.get('r2_test', '未知')}，RMSE={m['rmse']:.4g}")
            data[resp_name] = {"predicted": predicted, "residual": residuals}
            continue
        residuals = m["residual"]
        n = len(residuals)
        mean_r = statistics.mean(residuals) if n else 0.0
        std_r = statistics.pstdev(residuals) if n else 0.0
        std_res = [0.0 if std_r == 0 else (r - mean_r) / std_r for r in residuals]
        lines.append(f"  {resp_name}: R²={m['r2']}, 调整R²={m['adj_r2']}, "
                     f"RMSE={m['rmse']}, 残差均值={mean_r:.4g}, 残差标准差={std_r:.4g}")
        lines.append(f"    标准化残差区间 [{min(std_res):.3g}, {max(std_res):.3g}]，"
                     f"|z|>2 的占比 {sum(1 for z in std_res if abs(z) > 2) / max(n, 1) * 100:.1f}%")
        data[resp_name] = {"predicted": m["predicted"], "residual": residuals,
                           "std_residual": std_res, "r2": m["r2"]}
    return "\n".join(lines), data


def tolerance_contribution(context, k_design, point=None):
    if not context or not context["models"]:
        return "无模型可做容差贡献分析。", {}
    inputs = context["inputs"]
    names = context["names"]
    lines = ["【局部输入方差贡献（独立输入，对角二阶近似）】", f"参考点：{point or '设计区间中心'}", "包含一阶及对角曲率项，与当前优化器近似一致；不含混合二阶项，不代表全局重要性。"]
    detail = {}
    # 参考点：各设计因子取中心
    x_nom = {}
    x_raw = []
    for f in inputs:
        lo, hi = de.factor_limits(f)
        if f.source == "环境":
            val = float(f.param1) if (f.uncertainty == "概率" and f.param1) else (lo + hi) / 2.0
        else:
            val = (point or {}).get(f.name, (lo + hi) / 2.0)
            x_nom[f.name] = val
        x_raw.append(val)
    x = _np.array(x_raw, float)
    for resp_name, m in context["models"].items():
        grad = model_grad_x(m, x)
        sigmas = _np.array([_sigma_of_factor(f, k_design) for f in inputs])
        hess = model_hess_diag_x(m, x)
        contrib = (grad * sigmas) ** 2 + 0.5 * (hess * sigmas**2)**2
        residual_var = float(m.get("residual_std", 0))**2
        total = float(contrib.sum()) + residual_var
        rows = sorted(list(zip(names, contrib.tolist())) + [("模型残差", residual_var)], key=lambda kv: -kv[1])
        lines.append(f"  ◆ 响应 {resp_name}（总方差 {total:.5g}）")
        for name, c in rows:
            lines.append(f"      {name}: 贡献 {c:.5g}（{(c / total * 100 if total > 0 else 0):.2f}%）")
        detail[resp_name] = [{"name": name, "variance": c, "share": (c / total * 100 if total > 0 else 0)} for name, c in rows]
    return "\n".join(lines), detail


# ============================================================== 参数设计/取值器
def _metric_at(context, xd, k_design, k_constraint=6.0):
    r = evaluate_design(context, xd, k_design, weight_r=0.5, use_constraints=True,
                        k_constraint=k_constraint)
    return r["mean_perf"], r["std_norm"]


def parameter_design(context, robust_result, k_design, k_constraint=6.0):
    if not context or not context["models"]:
        return {"text": "无模型可做参数设计。"}
    d_inputs = design_inputs(context)
    dnames = [f.name for f in d_inputs]
    if not dnames:
        return {"text": "没有设计因子。"}
    lo = {f.name: de.factor_limits(f)[0] for f in d_inputs}
    hi = {f.name: de.factor_limits(f)[1] for f in d_inputs}
    # 推荐点（来自鲁棒优化）
    recommended = dict(robust_result.get("best", {}).get("x", {}))
    if not recommended:
        recommended = {n: (lo[n] + hi[n]) / 2.0 for n in dnames}
    # 稳定点：随机搜索最小化归一化 σ
    rng = _np.random.default_rng(7)
    initial = evaluate_design(context, recommended, k_design, .5, True, k_constraint)
    best_x = dict(recommended) if initial["infeas"] <= 1e-12 else None
    best_std = initial["std_norm"] if best_x is not None else float("inf")
    for _ in range(400):
        trial = {n: rng.uniform(lo[n], hi[n]) for n in dnames}
        evaluation = evaluate_design(context, trial, k_design, .5, True, k_constraint)
        std = evaluation["std_norm"]
        if evaluation["infeas"] <= 1e-12 and std < best_std:
            best_std, best_x = std, dict(trial)
    # 因子扫描图数据
    plots = []
    for f in d_inputs:
        xs, mean_curve, std_curve, feasibility = [], [], [], []
        n_pts = 9
        for t in range(n_pts):
            val = lo[f.name] + (hi[f.name] - lo[f.name]) * t / (n_pts - 1)
            xd = dict(recommended)
            xd[f.name] = val
            evaluated = evaluate_design(context, xd, k_design, .5, True, k_constraint)
            m, s = evaluated["mean_perf"], evaluated["std_norm"]
            feasibility.append(evaluated["infeas"] <= 1e-12)
            xs.append(round(val, 5))
            mean_curve.append(round(m, 4))
            std_curve.append(round(s, 4))
        plots.append({"factor": f.name, "x": xs, "mean": mean_curve, "std": std_curve, "feasible": feasibility})
    lines = ["【参数设计与取值器】"]
    lines.append(f"  优化返回参考点（约束违反={initial['infeas']:.5g}）：{ {k: round(v, 5) for k, v in recommended.items()} }")
    lines.append(f"  有限抽样内的可行稳定候选：{best_x or '未找到；不输出违反约束的稳定点'}")
    _m, _s = _metric_at(context, recommended, k_design, k_constraint)
    _m2, _s2 = _metric_at(context, best_x, k_design, k_constraint) if best_x is not None else (float("nan"), float("nan"))
    lines.append(f"  推荐点：望性均值 {_m:.4f}，σ 归一 {_s:.4f}")
    if best_x is not None:
        lines.append(f"  稳定候选：望性均值 {_m2:.4f}，σ 归一 {_s2:.4f}")
    lines.append("  提示：因子图见“参数设计”页，横轴为取值，纵轴为均值/稳定性。")
    return {"text": "\n".join(lines), "recommended": recommended,
            "stable": best_x, "factor_plots": plots}


def model_extraction(context):
    if not context or not context["models"]:
        return "无模型可提取。"
    lines = ["【模型提取（拟合方程）】"]
    for resp_name, m in context["models"].items():
        if "predict_fn" in m:
            lines.append(f"  {resp_name}：{m.get('model_type', '代理模型')} 已训练预测器；"
                         f"输入变量：{context['names']}。鲁棒优化直接调用该预测器，"
                         "采用数值梯度和 Hessian 传播输入不确定性，无二次多项式方程。")
            continue
        inputs = context["inputs"]
        names = context["names"]
        # 用实际的因子名替换 x_i 命名
        term_names = []
        for label in m["names"]:
            if label == "常数":
                term_names.append("常数")
            elif label.startswith("x") and "·" not in label and "²" not in label:
                idx = int(label[1:]) - 1
                term_names.append(names[idx] if idx < len(names) else label)
            elif "·" in label:
                a, b = label.split("·")
                ia, ib = int(a[1:]) - 1, int(b[1:]) - 1
                term_names.append(f"{names[ia]}·{names[ib]}")
            elif "²" in label:
                idx = int(label[1:-1]) - 1
                term_names.append(f"{names[idx]}²")
            else:
                term_names.append(label)
        s = []
        for i, (c, term) in enumerate(zip(m["coef"], term_names)):
            sign = " + " if c >= 0 else " - "
            mag = abs(float(c))
            body = f"{mag:.5g}" if term == "常数" else f"{mag:.5g}·{term}"
            s.append((body if i == 0 and c >= 0 else sign + body))
        lines.append("    以下符号均为编码变量：" + str(dict(zip(names, zip(m["center"].tolist(), m["half"].tolist())))) + "；编码值=(实际值−中心)/半跨度。")
        lines.append(f"  {resp_name}: y = " + "".join(s).strip())
        lines.append(f"    输入变量：{names}；R²={m['r2']}，调整R²={m['adj_r2']}")
    return "\n".join(lines)


# ============================================================== 优化工具集
def _design_bounds(context):
    d_inputs = design_inputs(context)
    lo = [de.factor_limits(f)[0] for f in d_inputs]
    hi = [de.factor_limits(f)[1] for f in d_inputs]
    return d_inputs, lo, hi


def local_search(context, start, k_design, k_constraint=6.0, weight=.5,
                 method="AOFAT", steps=12, step_frac=.1):
    """Bounded surrogate proposals, feasibility first; never labels an infeasible point feasible."""
    if not context or not context["models"]:
        raise ValueError("无模型，无法执行优化工具")
    inputs, lo, hi = _design_bounds(context)
    if not inputs:
        raise ValueError("无可调整设计因子")
    names = [f.name for f in inputs]
    lo, hi = _np.array(lo), _np.array(hi)
    span = hi-lo
    z = _np.clip((_np.array([start.get(n, (l+h)/2) for n,l,h in zip(names,lo,hi)])-lo)/span, 0, 1)
    def evaluate(z):
        point=dict(zip(names, (lo+z*span).tolist()))
        r=evaluate_design(context,point,k_design,weight,True,k_constraint)
        if not all(math.isfinite(r[key]) for key in ("mean_perf", "std_norm", "infeas")):
            raise ValueError("模型在探索点返回非有限值，停止探索")
        score=weight*(1-r["mean_perf"])+(1-weight)*r["std_norm"]
        return {"x":point,"mean":r["mean_perf"],"std":r["std_norm"],
                "violation":r["infeas"],"score":score,"feasible":r["infeas"]<=1e-12}
    def key(r):
        return (0 if r["feasible"] else 1, 0 if r["feasible"] else r["violation"], r["score"])
    current=evaluate(z);path=[current];trials=[]
    step=step_frac
    for iteration in range(1,steps+1):
        candidates=[]
        if method=="最陡上升":
            grad=[]
            for j in range(len(z)):
                zp=z.copy();zm=z.copy();zp[j]=min(1,z[j]+.001);zm[j]=max(0,z[j]-.001)
                rp,rm=evaluate(zp),evaluate(zm)
                field="score" if current["feasible"] else "violation"
                grad.append((rp[field]-rm[field])/(zp[j]-zm[j]))
            grad=_np.asarray(grad);norm=_np.linalg.norm(grad)
            if norm>1e-12:
                candidates=[_np.clip(z-step*scale*grad/norm,0,1) for scale in (1,.5,.25,.125)]
        else:
            for j in range(len(z)):
                for sign in (-1,1):
                    candidate=z.copy();candidate[j]=_np.clip(z[j]+sign*step,0,1);candidates.append(candidate)
        evaluated=[(candidate,evaluate(candidate)) for candidate in candidates]
        trials.extend({"iteration":iteration,**r} for _,r in evaluated)
        better=[pair for pair in evaluated if key(pair[1]) < key(current)]
        if better:
            z,current=min(better,key=lambda pair:key(pair[1]));path.append(current)
        else:
            step*=.5
        if method=="EVOP" or step<1e-5:break
    lines=[f"【{method} 模型辅助局部探索】", f"λ={weight:g}，输入 k={k_design:g}，约束 k={k_constraint:g}。先满足约束，再减小加权损失。",
           "所有点为模型计算，不是新增实测；EVOP 为局部单因子试探，不替代实际重复试验。"]
    for i,r in enumerate(path):
        lines.append(f"{i}: {r['x']}；望性={r['mean']:.5g}，σ归一={r['std']:.5g}，约束违反={r['violation']:.5g}，损失={r['score']:.5g}")
    lines.append("候选满足当前约束，需实测确认。" if current["feasible"] else "未找到可行候选，不能作为可行推荐。")
    return {"text":"\n".join(lines),"path":path,"trials":trials,"best":current}


def steepest_ascent(context,start,k_design,steps=6,step_frac=.08,k_constraint=6.,weight=.5):
    r=local_search(context,start,k_design,k_constraint,weight,"最陡上升",steps,step_frac)
    return r["text"],r["path"]


def evop_cycle(context,center,k_design,k_constraint=6.,weight=.5):
    return local_search(context,center,k_design,k_constraint,weight,"EVOP",1,.05)["text"]


def aofat_search(context,center,k_design,rounds=4,k_constraint=6.,weight=.5):
    r=local_search(context,center,k_design,k_constraint,weight,"AOFAT",rounds,.15)
    return r["text"],r["path"]
