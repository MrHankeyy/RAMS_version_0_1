"""Classical DOE analyses. Screening effects and Taguchi loss are distinct outputs."""
import itertools
import math
import statistics
from collections import Counter, defaultdict

import numpy as np
import doe_engine as de


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("试验数据包含非有限数值")
    return value


def active_controls(project):
    return [f for f in project.factors if f.name and f.source == "设计" and not f.is_fixed]


def screening(project):
    factors = active_controls(project)
    matrix = project.doe_matrix
    if not factors or not matrix:
        raise ValueError("缺少筛选设计因子或试验数据")
    bounds = [de.factor_limits(f) for f in factors]
    X = np.array([[finite(row[f.name]) for f in factors] for row in matrix])
    Z = np.column_stack([(X[:, j] - (lo + hi) / 2) / ((hi-lo)/2)
                         for j, (lo, hi) in enumerate(bounds)])
    center = np.all(np.isclose(Z, 0), axis=1)
    corner = np.all(np.isclose(np.abs(Z), 1), axis=1)
    if not np.all(center | corner):
        raise ValueError("筛选数据应为两水平点或全因子中心点，输入坐标与方案不一致")
    if not corner.any():
        raise ValueError("没有可估计主效应的两水平试验点")
    A = np.column_stack([np.ones(corner.sum()), Z[corner]])
    if np.linalg.matrix_rank(A) != len(factors) + 1:
        raise ValueError("主效应列存在混杂/重复，无法独立估计；请重新生成可识别的设计")
    if not np.allclose(Z[corner].sum(axis=0), 0) or not np.allclose(Z[corner].T @ Z[corner], corner.sum()*np.eye(len(factors))):
        raise ValueError("筛选矩阵不再平衡正交，请检查是否遗漏、重复或更改试验行")
    # Detect main-effect / two-factor-interaction aliasing, including PB partial aliasing.
    aliases = []
    z = Z[corner]
    for i, f in enumerate(factors):
        for j, k in itertools.combinations(range(len(factors)), 2):
            rho = float(np.mean(z[:, i] * z[:, j] * z[:, k]))
            if abs(rho) > 1e-8:
                aliases.append(f"{f.name} 与 {factors[j].name}×{factors[k].name}：相关 {rho:+.3g}")
    result = {"response_analysis": {}, "response_effects": {}, "factor_rankings": [],
              "main_effects": [], "plot_points": [], "stability_impact": [],
              "errors": {}, "aliases": aliases,
              "diagnostic_summary": (project.design_params.get("meta", {}).get("diagnostic", "") +
                  "。主效应=高水平均值−低水平均值；效应份额为平方效应归一化，不等于显著性或方差归因。"),
              "conclusion": "填入完整响应后计算；筛选不输出连续优化点。"}
    shares = defaultdict(list)
    for resp in project.responses:
        if not resp.name:
            continue
        try:
            y = np.array([finite(row.get(resp.name)) for row in matrix])
        except (ValueError, TypeError):
            result["errors"][resp.name] = "响应缺失或非有限；不使用不平衡的部分数据计算排名"
            continue
        grouped = defaultdict(list)
        for row, value in zip(X, y):
            grouped[tuple(row)].append(float(value))
        pure_df = sum(len(v)-1 for v in grouped.values())
        pure_ss = sum(sum((v-statistics.mean(values))**2 for v in values) for values in grouped.values())
        mse = pure_ss / pure_df if pure_df else None
        effects = []
        for j, f in enumerate(factors):
            low, high = y[corner & (Z[:, j] < 0)], y[corner & (Z[:, j] > 0)]
            effect = float(high.mean()-low.mean())
            se = math.sqrt(mse*(1/len(low)+1/len(high))) if mse is not None else None
            p = None
            if se is not None and se > 0:
                try:
                    from scipy.stats import t
                    p = float(2*t.sf(abs(effect/se), pure_df))
                except ImportError:
                    pass
            effects.append({"name": f.name, "effect": effect, "low_mean": float(low.mean()),
                            "high_mean": float(high.mean()), "low_n": len(low), "high_n": len(high),
                            "se": se, "p": p})
        total = sum(e["effect"]**2 for e in effects)
        effects.sort(key=lambda e: -abs(e["effect"]))
        for i, e in enumerate(effects):
            e.update(rank=i+1, share=100*e["effect"]**2/total if total else 0)
            shares[e["name"]].append(e["share"])
        curvature = float(y[center].mean()-y[corner].mean()) if center.any() else None
        item = {"effects": effects, "pure_error_df": pure_df, "pure_error_variance": mse,
                "curvature": curvature, "center_count": int(center.sum()),
                "note": "无重复试验，无法独立估计纯误差；不报告显著性。" if not pure_df else
                        "p 值基于重复点纯误差、独立同方差假设，未经多重比较校正；混杂仍须结合设计判断。"}
        result["response_analysis"][resp.name] = item
        result["response_effects"][resp.name] = effects
    ranking = [{"name": name, "share": statistics.mean(values), "effect": statistics.mean(values)}
               for name, values in shares.items()]
    ranking.sort(key=lambda e: -e["share"])
    for i, item in enumerate(ranking):
        item["rank"] = i+1
    result["factor_rankings"] = ranking
    result["main_effects"] = ranking[:project.design_params.get("top_n", 5)]
    result["plot_points"] = [{"name": e["name"], "value": e["share"]} for e in ranking]
    if ranking:
        result["conclusion"] = "已计算各响应的有符号主效应。综合排名为各响应平方效应份额的等权平均，不直接比较不同量纲；建议重点因子进入后续建模与确认试验。"
    result["analysis_status"] = "部分响应未完成" if result["errors"] else "筛选主效应分析完成"
    return result


def snr(values, feature, target=None):
    values = [finite(v) for v in values]
    if not values:
        raise ValueError("信噪比没有有效重复观测")
    if feature == "望大":
        if any(v <= 0 for v in values):
            raise ValueError("望大信噪比要求响应为正；零或负值不能使用倒数平方公式")
        loss = statistics.mean(1/v**2 for v in values)
    elif feature == "望小":
        loss = statistics.mean(v*v for v in values)
    elif target is not None:
        loss = statistics.mean((v-target)**2 for v in values)
    else:
        if len(values) < 2:
            raise ValueError("名义最好信噪比至少需要两次观测")
        mean = statistics.mean(values)
        if mean == 0:
            raise ValueError("经典名义最好信噪比的均值不能为零；请给出目标上下界")
        loss = statistics.variance(values) / mean**2
    return -10*math.log10(loss) if loss > 0 else float("inf")


def taguchi(project):
    meta = project.design_params.get("meta") or project.taguchi_result
    matrix = project.doe_matrix
    factors = active_controls(project)
    if not matrix or not factors or not meta.get("factor_levels"):
        raise ValueError("缺少田口试验矩阵或水平定义")
    n_inner, n_outer = meta["inner_runs"], meta["outer_runs"]
    repeats = meta.get("replicates", 1)
    groups = meta.get("groups", [])
    # Prefer persistent row identifiers so sorting / CSV round trips do not change groups.
    if all("内表组号" in row and "外表组号" in row for row in matrix):
        if any(not finite(row[key]).is_integer() for row in matrix for key in ("内表组号", "外表组号")):
            raise ValueError("内外表组号必须为整数")
        ids = [(int(finite(row["内表组号"]))-1, int(finite(row["外表组号"]))-1) for row in matrix]
    elif len(groups) == len(matrix) and all("Run_ID" in row for row in matrix):
        run_ids = [finite(row["Run_ID"]) for row in matrix]
        if any(not v.is_integer() for v in run_ids) or set(run_ids) != set(range(1, len(matrix)+1)):
            raise ValueError("试验编号不完整或重复，无法恢复田口分组")
        ids = [(groups[int(v)-1]["inner"], groups[int(v)-1]["outer"]) for v in run_ids]
    elif len(groups) == len(matrix):
        ids = [(g["inner"], g["outer"]) for g in groups]
    else:
        raise ValueError("田口分组标识缺失，不能把行位置当作新的内外表分组")
    expected = Counter({(i, j): repeats for i in range(n_inner) for j in range(n_outer)})
    if Counter(ids) != expected:
        raise ValueError("内表×外表×重复的组数不完整，存在遗漏或重复行")
    settings = {}
    noise_settings = {}
    _, _, _, planned_inner = de.taguchi_arrays(meta["inner_label"], len(factors))
    noise_names = list(meta.get("noise_levels", {}))
    planned_outer = de.taguchi_arrays(meta["outer_label"], len(noise_names))[3] if noise_names else None
    for row, (i, j) in zip(matrix, ids):
        noise = tuple(finite(row[name]) for name in meta.get("noise_levels", {}))
        if j in noise_settings and noise_settings[j] != noise:
            raise ValueError(f"外表组 {j+1} 的噪声因子不一致")
        noise_settings[j] = noise
        for name, value in zip(meta.get("noise_levels", {}), noise):
            if not any(math.isclose(value, lv, abs_tol=1e-7) for lv in meta["noise_levels"][name]):
                raise ValueError(f"噪声因子 {name} 的值不在设定水平中")
        values = tuple(finite(row[f.name]) for f in factors)
        if i in settings and settings[i] != values:
            raise ValueError(f"内表组 {i+1} 的控制因子不一致，请检查响应回填与组号")
        for col, f in enumerate(factors):
            expected_value = meta["factor_levels"][f.name][planned_inner[i][col]]
            if not math.isclose(values[col], expected_value, abs_tol=1e-7):
                raise ValueError(f"内表组 {i+1} 的 {f.name} 与原设计不一致，请检查试验编号和输入列")
        if planned_outer is not None:
            for col, name in enumerate(noise_names):
                expected_value = meta["noise_levels"][name][planned_outer[j][col]]
                if not math.isclose(noise[col], expected_value, abs_tol=1e-7):
                    raise ValueError(f"外表组 {j+1} 的 {name} 与原设计不一致")
        settings[i] = values
        for f, value in zip(factors, values):
            if not any(math.isclose(value, lv, abs_tol=1e-7) for lv in meta["factor_levels"][f.name]):
                raise ValueError(f"因子 {f.name} 的值不在田口设定水平中")
    result = {"array_name": meta["inner_label"], "outer_label": meta["outer_label"],
              **meta, "response_analysis": {}, "signal_to_noise": [], "best_levels": {},
              "best_levels_by_response": {}, "errors": {},
              "recommendation": "按响应分别给出候选水平；组合采用主效应可加假设，必须进行确认试验。",
              "summary": f"{n_inner} 内表组 × {n_outer} 外表组 × {repeats} 次重复。"}
    has_replication = n_outer * repeats >= 2
    mode = project.design_params.get("snr_mode", "按各响应目标特征自动")
    for resp in project.responses:
        if not resp.name:
            continue
        try:
            values = defaultdict(list)
            for row, (i, j) in zip(matrix, ids):
                values[i].append(finite(row.get(resp.name)))
            feature = resp.feature if "按各响应" in mode else ("望大" if "望大" in mode else "望小" if "望小" in mode else "望目")
            target = None
            if feature == "望目":
                lo, hi = finite(resp.lower), finite(resp.upper)
                if hi < lo:
                    raise ValueError("目标上下界颠倒")
                target = (lo + hi)/2
            runs = []
            for i in range(n_inner):
                v = values[i]
                runs.append({"inner": i+1, "n": len(v), "mean": statistics.mean(v),
                             "std": statistics.stdev(v) if len(v)>1 else None,
                             "metric": snr(v, feature, target) if has_replication else statistics.mean(v),
                             "settings": dict(zip([f.name for f in factors], settings[i]))})
            level_rows, best = [], {}
            for f in factors:
                table = []
                for level in meta["factor_levels"][f.name]:
                    matching = [r for r in runs if math.isclose(r["settings"][f.name], level, abs_tol=1e-7)]
                    if not matching:
                        raise ValueError(f"因子 {f.name} 缺少水平 {level}")
                    table.append({"factor": f.name, "level_value": level,
                                  "mean": statistics.mean(r["mean"] for r in matching),
                                  "std": statistics.mean(r["std"] for r in matching) if has_replication else None,
                                  "metric": statistics.mean(r["metric"] for r in matching), "n": len(matching)})
                if has_replication or feature == "望大":
                    chosen = max(table, key=lambda r:r["metric"])
                elif feature == "望小":
                    chosen = min(table, key=lambda r:r["mean"])
                else:
                    chosen = min(table, key=lambda r:abs(r["mean"]-target))
                if "目标" in resp.kind:
                    best[f.name] = chosen["level_value"]
                for r in table:
                    r["recommended"] = bool(best) and r is chosen
                level_rows.extend(table)
                result["signal_to_noise"].append({"factor": f.name, "response": resp.name,
                    "detail": "；".join(f"水平 {r['level_value']}: 均值 {r['mean']:.5g}, 指标 {r['metric']:.5g}" for r in table),
                    "snr": chosen["metric"]})
            metric_label = ("目标偏差损失 S/N (dB)" if feature=="望目" else "S/N (dB)") if has_replication else "均值（单次观测，不能评估稳健性）"
            result["response_analysis"][resp.name] = {"runs": runs, "levels": level_rows,
                "best_levels": best, "metric_label": metric_label, "feature": feature,
                "has_replication": has_replication, "role": resp.kind,
                "note": "约束响应仅报告观测结果，不按 S/N 最优替代约束可行性检验。" if "约束" in resp.kind else
                        "望小 S/N 使用均方响应，衡量偏离零的大小。" if feature=="望小" else "最大化 S/N，候选水平需要确认试验。"}
            if best:
                result["best_levels_by_response"][resp.name] = best
        except (ValueError, TypeError, KeyError) as exc:
            result["errors"][resp.name] = str(exc)
    candidates = list(result["best_levels_by_response"].values())
    if candidates and all(c == candidates[0] for c in candidates) and not result["errors"]:
        result["best_levels"] = candidates[0]
    elif len(candidates)>1:
        result["recommendation"] += " 不同响应的候选水平冲突，未强行合并为一个最优组合。"
    result["analysis_status"] = ("S/N 计算完成" if has_replication else "仅均值分析完成：无外表或重复观测，不报告稳健性")
    if result["errors"]:
        result["analysis_status"] = "部分响应分析失败，请查看具体原因"
    return result
