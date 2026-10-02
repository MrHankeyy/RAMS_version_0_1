"""Structured statistics for the workbench; no fitting or silent model substitution."""
import math
import numpy as np
from scipy import stats
import optimizer_engine as oe


def descriptive(project):
    rows = []
    for response in project.responses:
        if not response.name:
            continue
        raw = [r.get(response.name) for r in project.doe_matrix]
        values = np.array([float(v) for v in raw if oe._isfinite(v)])
        n = len(values)
        row = {"响应": response.name, "有效数": n, "缺失或无效": len(raw)-n}
        if n:
            row.update(zip(["最小值", "Q1", "中位数", "Q3", "最大值"], np.quantile(values,[0,.25,.5,.75,1]).tolist()))
            row.update({"均值": float(values.mean()), "样本标准差": float(values.std(ddof=1)) if n>1 else None,
                        "IQR": float(np.quantile(values,.75)-np.quantile(values,.25)),
                        "偏度": float(stats.skew(values,bias=False)) if n>2 and np.ptp(values)>0 else None,
                        "超额峰度": float(stats.kurtosis(values,bias=False)) if n>3 and np.ptp(values)>0 else None})
        rows.append(row)
    return rows


def diagnostics(context, name):
    model = context['models'][name]
    surrogate = 'predict_fn' in model
    coefficients = []
    notes = []
    if surrogate:
        val = model.get('validation', {})
        actual = np.asarray(val.get('actual', []),float)
        predicted = np.asarray(val.get('predicted', []),float)
        ids = list(range(1,len(actual)+1))
        source = '留出验证集（序号不是原 DOE 行号）'
        notes.append('代理模型仅展示已有留出验证结果，不对黑盒预测器报告 OLS 系数检验、杠杆值或 Cook 距离。')
    else:
        names = context['names']
        data = [(i+1,r) for i,r in enumerate(context['_project'].doe_matrix)
                if all(oe._isfinite(r.get(k)) for k in names+[name])]
        ids = [i for i,_ in data]
        x = np.asarray([[float(r[k]) for k in names] for _,r in data])
        actual = np.asarray([float(r[name]) for _,r in data])
        predicted = np.asarray([oe.model_predict(model,row) for row in x])
        source = '训练集拟合（不是外部验证）'
    if len(actual) != len(predicted) or not len(actual) or not np.all(np.isfinite(actual+predicted)):
        return {'summary':[], 'coefficients':[], 'observations':[], 'notes':['没有匹配的有限观测与预测。'], 'source':source}
    residual = actual-predicted
    n = len(actual)
    sse = float(residual@residual)
    sst = float(np.sum((actual-actual.mean())**2))
    measures = {'数据来源':source,'样本数':n,'R²':1-sse/sst if sst>0 else None,
                'RMSE':math.sqrt(sse/n),'MAE':float(np.abs(residual).mean()),
                '最大绝对误差':float(np.abs(residual).max()),'残差均值':float(residual.mean()),
                '残差样本标准差':float(residual.std(ddof=1)) if n>1 else None}
    leverage = student = cooks = None
    if not surrogate:
        z = (x-model['center'])/model['half']
        a = np.array([oe._design_row(row,model['terms']) for row in z])
        p = a.shape[1];df=n-p
        inv = np.linalg.pinv(a.T@a)
        leverage=np.sum((a@inv)*a,axis=1)
        measures.update({'参数数':p,'残差自由度':df,'设计矩阵条件数':float(np.linalg.cond(a))})
        measures['回归平方和']=sst-sse
        measures['残差平方和']=sse
        measures['总平方和']=sst
        usable = df>0 and sse>np.finfo(float).eps**2*max(float(actual@actual),1)*100
        overall_f=((sst-sse)/(p-1))/(sse/df) if usable and p>1 else None
        measures.update({'回归 F':overall_f,'回归 p':float(stats.f.sf(overall_f,p-1,df)) if overall_f is not None else None})
        groups={}
        for row,value in zip(x,actual):groups.setdefault(tuple(row),[]).append(float(value))
        pure_df=sum(len(v)-1 for v in groups.values())
        pure_ss=sum(float(np.sum((np.array(v)-np.mean(v))**2)) for v in groups.values())
        lof_df=df-pure_df
        lof_f=(max(sse-pure_ss,0)/lof_df)/(pure_ss/pure_df) if lof_df>0 and pure_df>0 and pure_ss>1e-24 else None
        measures.update({'纯误差自由度':pure_df,'纯误差平方和':pure_ss,'失拟自由度':lof_df,
                         '失拟 F':lof_f,'失拟 p':float(stats.f.sf(lof_f,lof_df,pure_df)) if lof_f is not None else None})
        if usable:
            mse=sse/df
            se=np.sqrt(np.maximum(np.diag(inv)*mse,0))
            tcrit=float(stats.t.ppf(.975,df))
            with np.errstate(divide='ignore',invalid='ignore'):
                student=residual/np.sqrt(mse*np.maximum(1-leverage,0))
                cooks=residual**2/(p*mse)*leverage/(1-leverage)**2
        else:
            se=None;tcrit=None
            notes.append('残差自由度不足或误差数值上为零：不输出虚假的系数 p 值、区间或影响诊断。')
        for j,(label,coef) in enumerate(zip(model['names'],model['coef'])):
            serr=float(se[j]) if se is not None else None
            t=float(coef)/serr if serr and serr>0 else None
            coefficients.append({'编码项':label,'估计值':float(coef),'标准误':serr,'t':t,
                '未校正 p':float(2*stats.t.sf(abs(t),df)) if t is not None else None,
                '95%下限':float(coef)-tcrit*serr if serr is not None else None,
                '95%上限':float(coef)+tcrit*serr if serr is not None else None})
        if np.all(1-leverage>1e-8):
            press=float(np.sum((residual/(1-leverage))**2))
            measures.update({'LOO PRESS':press,'LOO RMSE':math.sqrt(press/n),'预测 R² (LOO)':1-press/sst if sst>0 else None})
        else:
            notes.append('部分杠杆值接近 1，留一后模型不可识别，PRESS/预测 R² 不可估计。')
        notes.append('系数区间为 OLS 条件推断，假设独立同方差正态误差；p 值未校正多重比较。编码 z=(实际值−中心)/半跨度。')
        notes.append('编码映射：'+str(dict(zip(context['names'],zip(model['center'].tolist(),model['half'].tolist())))))
    observations=[]
    for i in range(n):
        row={'序号':ids[i],'实测':float(actual[i]),'预测':float(predicted[i]),'残差':float(residual[i])}
        if leverage is not None:
            row.update({'杠杆值':float(leverage[i]),'内部学生化残差':float(student[i]) if student is not None and np.isfinite(student[i]) else None,
                        'Cook距离':float(cooks[i]) if cooks is not None and np.isfinite(cooks[i]) else None,
                        '高杠杆提示':bool(leverage[i]>2*p/n)})
        observations.append(row)
    if 3<=n<=5000 and np.ptp(residual)>1e-12:
        measures['残差 Shapiro p']=float(stats.shapiro(residual).pvalue)
        notes.append('Shapiro 仅作残差分布诊断，不显著不等于证明正态；DOE 行序不一定是实际试验时间。')
    return {'summary':[{'统计量':k,'值':v} for k,v in measures.items()], 'coefficients':coefficients,
            'observations':observations,'notes':notes,'source':source}


def point_analysis(context, point, kd, kc):
    for f in oe.design_inputs(context):
        lo,hi=oe.de.factor_limits(f)
        if f.name not in point or not oe._isfinite(point[f.name]) or not lo<=point[f.name]<=hi:
            raise ValueError(f'{f.name} 必须在 [{lo}, {hi}] 内')
    evaluation=oe.evaluate_design(context,point,kd,.5,use_constraints=True,k_constraint=kc)
    rows=[]
    for response in context['_responses']:
        model=context['models'].get(response.name)
        if model is None: continue
        mu,sigma=oe.robust_moments(model,point,context,kd)
        raw=[]
        for f in context['inputs']:
            lo,hi=oe.de.factor_limits(f)
            raw.append(float(f.param1) if f.source=='环境' and f.uncertainty=='概率' else
                       (lo+hi)/2 if f.source=='环境' else point[f.name])
        nominal=oe.model_predict(model,raw)
        lower,upper=mu-kc*sigma,mu+kc*sigma
        margin=None
        if oe.self_is_constraint(response):
            kind,limit=oe._constraint_limits(response)
            margin=limit-upper if kind=='upper' else lower-limit if kind=='lower' else min(lower-limit[0],limit[1]-upper)
        sigma_limit = context.get('sigma_limits', {}).get(response.name)
        sigma_margin = sigma_limit - sigma if sigma_limit is not None else None
        rows.append({'标准差上限':sigma_limit, '波动约束余量':sigma_margin, '响应':response.name,'角色':response.kind,'名义预测':nominal,'鲁棒均值':mu,'传播标准差':sigma,
                     '均值−kσ':lower,'均值+kσ':upper,'约束余量':margin,
                     '判定':'违反' if any(v is not None and v < 0 for v in (margin,sigma_margin)) else '通过' if margin is not None or sigma_margin is not None else '目标'})
    return {'point':dict(point),'k_design':kd,'k_constraint':kc,'feasible':evaluation['infeas']<=1e-12,'responses':rows,
            'note':'μ±kσ 是当前局部矩近似下的稳健带，不是预测置信区间，也不保证指定覆盖概率。输入独立，沿用优化器的对角二阶近似。'}


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
