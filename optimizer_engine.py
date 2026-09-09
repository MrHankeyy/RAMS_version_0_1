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


def _quad_hess_diag(terms, z):
    """二次模型对角二阶导（对 z）。"""
    out = _np.zeros(len(z))
    for kind, i, _j in terms:
        if kind == "quad":
            out[i] += 2.0
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


def _fit_model(X, y):
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
        if distinct[i] >= 3:
            terms.append(("quad", i, i))
            names.append(f"x{i + 1}²")
    for i in range(k):
        for j in range(i + 1, k):
            terms.append(("cross", i, j))
            names.append(f"x{i + 1}·x{j + 1}")
    A = _np.column_stack([_design_row(z, terms) for z in Z]).T
    nterms = len(terms)
    if n < nterms:
        # 点数不足时降为线性模型
        keep = [i for i, (kind, *_r) in enumerate(terms) if kind in ("const", "linear")]
        terms = [terms[i] for i in keep]
        names = [names[i] for i in keep]
        A = A[:, keep]
        nterms = len(keep)
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
        "center": center, "half": span, "lo": lo, "hi": hi,
        "r2": round(r2, 6), "adj_r2": round(adj_r2, 6), "rmse": round(rmse, 6),
        "df_model": df_model, "df_residual": df_resid,
        "residual_std": resid_std, "residual": list(float(v) for v in residual),
        "predicted": list(float(v) for v in y_hat), "n": n, "k": k,
    }


def model_predict(model, x_raw):
    if not HAS_NUMPY:
        return None
    z = (_np.asarray(x_raw, float) - model["center"]) / model["half"]
    row = _design_row(z, model["terms"])
    return float(row @ model["coef"])


def model_grad_x(model, x_raw):
    """梯度 wrt 原始单位 x。"""
    z = (_np.asarray(x_raw, float) - model["center"]) / model["half"]
    gz = _quad_grad(model["terms"], model["coef"], z)
    return gz / model["half"]


def model_hess_diag_x(model, x_raw):
    z = (_np.asarray(x_raw, float) - model["center"]) / model["half"]
    hz = _quad_hess_diag(model["terms"], z)
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
    inputs = [f for f in project.factors
              if f.name and not (f.is_fixed and f.fixed_value not in (None, ""))]
    names = [f.name for f in inputs]
    if not names:
        return None, ["没有可建模的输入因子。"]
    # 只使用矩阵中实际存在的输入列
    keys = set(matrix[0].keys())
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
            except (TypeError, ValueError):
                continue
            try:
                yv = float(row.get(resp.name))
            except (TypeError, ValueError):
                continue
            X.append([float(row[n]) for n in names])
            y.append(yv)
        if len(y) < 3:
            context["errors"].append(f"响应 {resp.name} 有效样本 {len(y)} < 3，跳过建模。")
            continue
        context["models"][resp.name] = _fit_model(X, y)
    if not context["models"]:
        return context, ["没有足够的响应数据可建模。"]
    return context, context["errors"]


def _sigma_of_factor(f, k_design):
    """因子不确定性标准差。"""
    lo, hi = de.factor_limits(f)
    if f.source == "环境":
        if f.uncertainty == "概率":
            try:
                return (float(f.param2) ** 0.5) * 0.5  # 将“方差”当作半宽处理
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
    lo, hi = _response_ref(resp)
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
    lo, hi = _response_ref(resp)
    return ("both", (lo, hi)) if lo is not None and hi is not None else ("both", (None, None))


def _response_scale(model):
    return model["rmse"] if model["rmse"] > 0 else 1.0


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
    for resp in project_responses(context):
        m = models.get(resp.name)
        if m is None:
            continue
        mu, sigma = robust_moments(m, x_design, context, k_design)
        d = _desirability(resp, mu, data_min[resp.name], data_max[resp.name])
        scale = _response_scale(m)
        std_n = sigma / max(scale, 1e-12)
        # 是否约束
        is_cons = use_constraints and self_is_constraint(resp)
        if is_cons:
            kind, lim = _constraint_limits(resp)
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
            infeas += infeas_in / (abs(lim if isinstance(lim, (int, float)) else (lim[1] if lim else 1.0)) + 1e-9)
            resp_cons.append({"name": resp.name, "mu": mu, "sigma": sigma, "d": d})
        else:
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
    kind = getattr(resp, "kind", "")
    if "约束" in kind:
        return True
    if resp.feature == "望小":
        ref = _response_ref(resp)
        return ref is not None
    if resp.feature == "望目":
        lo, hi = _response_ref(resp)
        return lo is not None and hi is not None
    return False


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


def nsga2(evaluate, lo, hi, pop=60, gen=100, seed=1, eta_c=15.0, eta_m=20.0):
    """最小化多目标。evaluate(vec)->(obj, info)。返回 (最终种群, 全部评估, 目标值)。"""
    rng = _np.random.default_rng(seed)
    n = len(lo)
    population = [[rng.uniform(lo[i], hi[i]) for i in range(n)] for _ in range(pop)]
    evaluated = []
    for _ in range(gen):
        objs = [evaluate(v)[0] for v in population]
        for v, o in zip(population, objs):
            evaluated.append((tuple(round(x, 6) for x in v), tuple(o)))
        fronts = _fast_non_dominated_sort(objs)
        next_pop = []
        for fr in fronts:
            dist = _crowding_distance(objs, fr)
            sorted_fr = sorted(fr, key=lambda i: (-dist[i]))
            need = pop - len(next_pop)
            next_pop.extend(sorted_fr[:need])
            if len(next_pop) >= pop:
                break
        next_indiv = [population[i] for i in next_pop]
        children = []
        while len(children) < pop:
            parents = rng.choice(len(next_indiv), 2, replace=False)
            p1, p2 = next_indiv[parents[0]], next_indiv[parents[1]]
            c1, c2 = _sbx_crossover(p1, p2, eta_c, lo, hi, rng)
            pm = 1.0 / n
            children.append(_polynomial_mutation(c1, eta_m, pm, lo, hi, rng))
            if len(children) < pop:
                children.append(_polynomial_mutation(c2, eta_m, pm, lo, hi, rng))
        population = children
    final_objs = [evaluate(v)[0] for v in population]
    for v, o in zip(population, final_objs):
        evaluated.append((tuple(round(x, 6) for x in v), tuple(o)))
    return population, evaluated


# ============================================================== 鲁棒优化
def design_inputs(context):
    return [f for f in context["inputs"] if f.source == "设计"]


def _pareto(sorted_pairs):
    """从 [(vec, objs, info)] 中筛出非支配解并排序。"""
    filtered = []
    for i, (vec, obj, _info) in enumerate(sorted_pairs):
        dominated = False
        for j, (vec2, obj2, _info2) in enumerate(sorted_pairs):
            if i == j:
                continue
            if _dominates(obj2, obj):
                dominated = True
                break
        if not dominated:
            filtered.append((vec, obj, _info))
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
        if s < score:
            score, best = s, item
    return best


def robust_optimize(context, mode="multi", pop=60, gen=100, weight=0.5,
                    k_design=6.0, k_constraint=6.0, seed=1):
    """执行鲁棒优化，返回结果 dict。"""
    dinputs = design_inputs(context)
    if not dinputs:
        return {"error": "没有可优化的设计因子，请先在【业务建模】添加设计因子。"}
    dnames = [f.name for f in dinputs]
    lo = [de.factor_limits(f)[0] for f in dinputs]
    hi = [de.factor_limits(f)[1] for f in dinputs]
    use_cons = mode in ("constraint", "weighted_constraint")

    def evaluate(vec):
        xd = dict(zip(dnames, vec))
        r = evaluate_design(context, xd, k_design, weight, use_constraints=use_cons,
                            k_constraint=k_constraint)
        if mode == "weighted":
            obj = [weight * r["obj"][0] + (1.0 - weight) * r["obj"][1]]
        else:
            obj = list(r["obj"])
        return obj, {"x": xd, "r": r}

    population, evaluated = nsga2(evaluate, lo, hi, pop=pop, gen=gen, seed=seed)
    pairs = []
    for vec_tuple, _obj in evaluated:
        vec = list(vec_tuple)
        obj, info = evaluate(vec)
        pairs.append((vec, obj, info))
    front_full = _pareto(pairs)

    # 选择推荐最优解
    if mode == "weighted":
        best = min(pairs, key=lambda t: t[1][0])
    elif mode == "constraint":
        feasible = [t for t in pairs if t[2]["r"]["infeas"] <= 1e-6]
        best = (min(feasible, key=lambda t: sum(t[1]))
                if feasible else min(pairs, key=lambda t: sum(t[1])))
    else:
        best = _knee_point(front_full) or min(pairs, key=lambda t: sum(t[1]))

    knee = None if mode == "weighted" else _knee_point(front_full)
    recommended = best
    x_rec = recommended[2]["x"]
    r_rec = recommended[2]["r"]
    resp_detail = []
    for item in r_rec["objectives"] + r_rec["constraints"]:
        resp_detail.append({"name": item["name"], "mu": item["mu"],
                            "sigma": item["sigma"], "d": round(item["d"], 4)})

    result = {
        "mode": mode, "weight": weight, "k_design": k_design,
        "k_constraint": k_constraint, "design_names": dnames,
        "front": [{"x": _info["x"], "obj": [round(float(o), 6) for o in _obj],
                   "infeas": _info["r"]["infeas"]}
                  for _vec, _obj, _info in front_full],
        "front_scatter": [[float(t[1][0]), float(t[1][1] if len(t[1]) > 1 else 0.0)]
                          for t in pairs if len(t[1]) > 1],
        "knee": ({"x": knee[2]["x"], "obj": [round(float(o), 6) for o in knee[1]],
                  "infeas": knee[2]["r"]["infeas"]} if knee else None),
        "best": {"x": x_rec, "obj": [round(float(o), 6) for o in recommended[1]],
                 "mean_perf": round(r_rec["mean_perf"], 4),
                 "std_norm": round(r_rec["std_norm"], 4),
                 "infeas": round(r_rec["infeas"], 6),
                 "responses": resp_detail},
        "evaluated_count": len(pairs),
    }
    result["text"] = _robust_result_text(result)
    return result


def _robust_result_text(res):
    mode_names = {"multi": "NSGA-II 多目标（均值 vs 稳定性）",
                  "weighted": "加权单目标（权重可设）",
                  "constraint": "约束型（稳定性转为独立约束）"}
    lines = [f"【{mode_names.get(res['mode'], res['mode'])}】",
             f"六西格玛设计系数：约束边界退让 k={res['k_constraint']}，"
             f"输入边界退让 k={res['k_design']}"]
    if res["mode"] == "weighted":
        lines.append(f"权重 λ(目标均值) = {res['weight']}，稳定性权重 = {round(1 - res['weight'], 2)}")
    lines.append(f"评估解数量：{res['evaluated_count']}")
    lines.append("")
    if res["mode"] != "weighted" and res["front"]:
        lines.append(f"帕累托前沿解数量：{len(res['front'])}")
        knee = res.get("knee")
        if knee:
            lines.append(f"Knee Point：{knee['x']} ｜ 目标值 {knee['obj']}")
    best = res["best"]
    lines.append("")
    lines.append("★ 推荐优化设计（稳健设计点）")
    lines.append(f"  因子设置：{ {k: round(v, 5) for k, v in best['x'].items()} }")
    lines.append(f"  综合望性均值 {best['mean_perf']} ｜ 归一化稳定性 {best['std_norm']} ｜ "
                 f"约束违例 {best['infeas']}")
    lines.append("  各响应（点 μ，σ，望性 d）：")
    for item in best["responses"]:
        lines.append(f"    {item['name']}: μ={item['mu']:.4g}, σ={item['sigma']:.4g}, d={item['d']}")
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
                vals.append(float(row.get(resp.name)))
            except (TypeError, ValueError):
                continue
        if not vals:
            lines.append(f"  {resp.name}: 无有效数据")
            continue
        vals.sort()
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals)
        lines.append(f"  响应 {resp.name}: n={len(vals)}/{n} 均值={mean:.4g} "
                     f"标准差={std:.4g} 最小={vals[0]:.4g} 最大={vals[-1]:.4g} "
                     f"极差={vals[-1] - vals[0]:.4g}")
    return "\n".join(lines)


def level_stats(project):
    matrix = project.doe_matrix or []
    factors = [f for f in project.factors if f.name and not (f.is_fixed and f.fixed_value not in (None, ""))]
    headers = ["因子", "水平", "响应", "均值", "方差", "极差", "样本数", "F比", "p值"]
    rows = []
    for f in factors:
        if f.name not in matrix[0]:
            continue
        level_values = sorted({float(row[f.name]) for row in matrix if row.get(f.name) not in (None, "")})
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
            df_w = max(n - g, 1)
            fstat = (ss_between / df_b) / (ss_within / df_w) if ss_within > 0 and df_w > 0 else 0.0
            p = _f_pvalue(fstat, df_b, df_w)
            for lv, v in groups:
                mean = statistics.mean(v)
                var = statistics.variance(v) if len(v) > 1 else 0.0
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
            lof_f = ms_lof / ms_pe if ms_pe > 0 else 0.0
            lof_p = _f_pvalue(lof_f, lof_df, pure_df)
            lines.append(f"  失拟(LOF)：SS_Lof={lof_ss:.4g}，SS_PE={pure_ss:.4g}，"
                         f"F_Lof={lof_f:.4g}，p={lof_p:.4g}，df=({lof_df},{pure_df})")
            lines.append("  判定：" + ("模型拟合充分（LOF 不显著）。" if lof_p > 0.05
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
        X, y = _matrix_xy(context, resp_name)
        if len(y) < 4:
            lines.append(f"  {resp_name}: 样本不足")
            continue
        kfold = min(5, len(y))
        rng = _np.random.default_rng(123)
        idx = rng.permutation(len(y))
        splits = _np.array_split(idx, kfold)
        preds = _np.zeros(len(y))
        for test_idx in splits:
            train_idx = _np.setdiff1d(_np.arange(len(y)), test_idx)
            if len(train_idx) < 3:
                continue
            try:
                m_fold = _fit_model(X[train_idx], y[train_idx])
                for i in test_idx:
                    preds[i] = model_predict(m_fold, X[i])
            except Exception:
                continue
        valid = preds != 0
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
        except (TypeError, ValueError):
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


def tolerance_contribution(context, k_design):
    if not context or not context["models"]:
        return "无模型可做容差贡献分析。"
    inputs = context["inputs"]
    names = context["names"]
    lines = ["【容差贡献度排序（一阶方差灵敏度）】"]
    detail = {}
    # 参考点：各设计因子取中心
    x_nom = {}
    x_raw = []
    for f in inputs:
        lo, hi = de.factor_limits(f)
        if f.source == "环境":
            val = float(f.param1) if (f.uncertainty == "概率" and f.param1) else (lo + hi) / 2.0
        else:
            val = (lo + hi) / 2.0
            x_nom[f.name] = val
        x_raw.append(val)
    x = _np.array(x_raw, float)
    for resp_name, m in context["models"].items():
        grad = model_grad_x(m, x)
        sigmas = _np.array([_sigma_of_factor(f, k_design) for f in inputs])
        contrib = (grad * sigmas) ** 2
        total = float(contrib.sum()) or 1e-12
        rows = sorted(zip(names, contrib.tolist()), key=lambda kv: -kv[1])
        lines.append(f"  ◆ 响应 {resp_name}（总方差 {total:.5g}）")
        for name, c in rows:
            lines.append(f"      {name}: 贡献 {c:.5g}（{c / total * 100:.2f}%）")
        detail[resp_name] = [{"name": name, "share": c / total * 100} for name, c in rows]
    return "\n".join(lines), detail


# ============================================================== 参数设计/取值器
def _metric_at(context, xd, k_design):
    r = evaluate_design(context, xd, k_design, weight_r=0.5, use_constraints=True,
                        k_constraint=6.0)
    return r["mean_perf"], r["std_norm"]


def parameter_design(context, robust_result, k_design):
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
    best_x, best_std = dict(recommended), float("inf")
    for _ in range(400):
        trial = {n: rng.uniform(lo[n], hi[n]) for n in dnames}
        _m, std = _metric_at(context, trial, k_design)
        if std < best_std:
            best_std, best_x = std, dict(trial)
    # 因子扫描图数据
    plots = []
    for f in d_inputs:
        xs, mean_curve, std_curve = [], [], []
        n_pts = 9
        for t in range(n_pts):
            val = lo[f.name] + (hi[f.name] - lo[f.name]) * t / (n_pts - 1)
            xd = dict(recommended)
            xd[f.name] = val
            m, s = _metric_at(context, xd, k_design)
            xs.append(round(val, 5))
            mean_curve.append(round(m, 4))
            std_curve.append(round(s, 4))
        plots.append({"factor": f.name, "x": xs, "mean": mean_curve, "std": std_curve})
    lines = ["【参数设计与取值器】"]
    lines.append(f"  推荐设计取值（鲁棒优化）：{ {k: round(v, 5) for k, v in recommended.items()} }")
    lines.append(f"  稳定性最优取值（最小 σ）：{ {k: round(v, 5) for k, v in best_x.items()} }")
    _m, _s = _metric_at(context, recommended, k_design)
    _m2, _s2 = _metric_at(context, best_x, k_design)
    lines.append(f"  推荐点：望性均值 {_m:.4f}，σ 归一 {_s:.4f}")
    lines.append(f"  稳定点：望性均值 {_m2:.4f}，σ 归一 {_s2:.4f}")
    lines.append("  提示：因子图见“参数设计”页，横轴为取值，纵轴为均值/稳定性。")
    return {"text": "\n".join(lines), "recommended": recommended,
            "stable": best_x, "factor_plots": plots}


def model_extraction(context):
    if not context or not context["models"]:
        return "无模型可提取。"
    lines = ["【模型提取（拟合方程）】"]
    for resp_name, m in context["models"].items():
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
        lines.append(f"  {resp_name}: y = " + "".join(s).strip())
        lines.append(f"    输入变量：{names}；R²={m['r2']}，调整R²={m['adj_r2']}")
    return "\n".join(lines)


# ============================================================== 优化工具集
def _design_bounds(context):
    d_inputs = design_inputs(context)
    lo = [de.factor_limits(f)[0] for f in d_inputs]
    hi = [de.factor_limits(f)[1] for f in d_inputs]
    return d_inputs, lo, hi


def steepest_ascent(context, start, k_design, steps=6, step_frac=0.08):
    if not context or not context["models"]:
        return "无模型，无法执行最陡上升。"
    d_inputs, lo, hi = _design_bounds(context)
    names = [f.name for f in d_inputs]
    x = dict(start)
    # 默认起点取中心
    for f, l, h in zip(d_inputs, lo, hi):
        x.setdefault(f.name, (l + h) / 2.0)
    lines = ["【最陡上升路径】"]
    lines.append(f"  起点：{ {k: round(v, 5) for k, v in x.items()} }")
    path = []
    for step in range(steps):
        base_m, base_s = _metric_at(context, x, k_design)
        grad = []
        for n, f in zip(names, d_inputs):
            l, h = de.factor_limits(f)
            delta = max((h - l) * 0.02, 1e-6)
            xp = dict(x); xp[n] = min(h, x[n] + delta)
            xm = dict(x); xm[n] = max(l, x[n] - delta)
            mp, _sp = _metric_at(context, xp, k_design)
            mm, _sm = _metric_at(context, xm, k_design)
            grad.append((mp - mm) / (2 * delta))
        norm = math.sqrt(sum(g * g for g in grad)) or 1e-12
        scale = max((h - l) * step_frac for l, h in zip(lo, hi))
        moved = False
        for n, g, f, l, h in zip(names, grad, d_inputs, lo, hi):
            newv = x[n] + (g / norm) * scale
            newv = max(l, min(h, newv))
            if abs(newv - x[n]) > 1e-6:
                x[n] = newv
                moved = True
        nm, ns = _metric_at(context, x, k_design)
        path.append({"x": dict(x), "mean": nm, "std": ns})
        lines.append(f"  第 {step + 1} 步：{ {k: round(v, 5) for k, v in x.items()} }"
                     f" 望性={nm:.4f}, σ_norm={ns:.4f}")
        if nm < base_m - 0.005 and step > 0:
            lines.append("  望性未改善，提前终止。")
            break
        if not moved:
            break
    return "\n".join(lines), path


def evop_cycle(context, center, k_design):
    if not context or not context["models"]:
        return "无模型，无法执行 EVOP。"
    d_inputs, lo, hi = _design_bounds(context)
    names = [f.name for f in d_inputs]
    x = dict(center)
    for f, l, h in zip(d_inputs, lo, hi):
        x.setdefault(f.name, (l + h) / 2.0)
    lines = ["【EVOP 演化操作（单因素 ±1 步循环）】"]
    base_m, base_s = _metric_at(context, x, k_design)
    lines.append(f"  当前中心：{ {k: round(v, 5) for k, v in x.items()} } => "
                 f"望性={base_m:.4f}, σ_norm={base_s:.4f}")
    effects = []
    for n, f in zip(names, d_inputs):
        l, h = de.factor_limits(f)
        step = (h - l) * 0.05
        plus = dict(x); plus[n] = min(h, x[n] + step)
        minus = dict(x); minus[n] = max(l, x[n] - step)
        mp, sp = _metric_at(context, plus, k_design)
        mm, sm = _metric_at(context, minus, k_design)
        lines.append(f"  因子 {n}: +step 望性={mp:.4f} σ={sp:.4f}；"
                     f"-step 望性={mm:.4f} σ={sm:.4f}")
        effects.append((n, mp, sp, mm, sm))
    # 建议：选对望性提升最大的因子方向
    best = None
    for n, mp, sp, mm, sm in effects:
        dn = mp - mm
        if best is None or abs(dn) > abs(best[1]):
            best = (n, dn, mp > mm)
    if best and abs(best[1]) > 1e-5:
        n, dn, up = best
        f = next(f for f in d_inputs if f.name == n)
        l, h = de.factor_limits(f)
        step = (h - l) * 0.05
        newx = dict(x)
        newx[n] = min(h, x[n] + step) if up else max(l, x[n] - step)
        lines.append(f"  → 建议：沿 {n} 向 {'增' if up else '减'} 方向移动一步到 "
                     f"{newx[n]:.5f}")
    return "\n".join(lines)


def aofat_search(context, center, k_design, rounds=4):
    if not context or not context["models"]:
        return "无模型，无法执行 AOFAT。"
    d_inputs, lo, hi = _design_bounds(context)
    names = [f.name for f in d_inputs]
    x = dict(center)
    for f, l, h in zip(d_inputs, lo, hi):
        x.setdefault(f.name, (l + h) / 2.0)
    lines = ["【自适应 AOFAT（单因子自适应优化）】"]
    path = []
    step_scale = 0.15
    for rnd in range(rounds):
        base_m, base_s = _metric_at(context, x, k_design)
        score0 = base_m - 0.35 * base_s
        best_move, best_score = None, score0
        for n, f in zip(names, d_inputs):
            l, h = de.factor_limits(f)
            step = (h - l) * step_scale
            for sgn in (1, -1):
                xp = dict(x)
                xp[n] = min(h, max(l, x[n] + sgn * step))
                m, s = _metric_at(context, xp, k_design)
                sc = m - 0.35 * s
                if sc > best_score + 1e-6:
                    best_score, best_move = sc, (n, sgn)
        if best_move is None:
            break
        n, sgn = best_move
        f = next(f for f in d_inputs if f.name == n)
        l, h = de.factor_limits(f)
        step = (h - l) * step_scale
        x[n] = min(h, max(l, x[n] + sgn * step))
        m, s = _metric_at(context, x, k_design)
        path.append({"x": dict(x), "mean": m, "std": s})
        lines.append(f"  第 {rnd + 1} 轮：{n} {'+' if sgn > 0 else '-'}"
                     f" 到 {x[n]:.5f} => 望性={m:.4f} σ_norm={s:.4f}")
        step_scale *= 0.7
    return "\n".join(lines), path
