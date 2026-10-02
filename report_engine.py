"""Versioned, source-grounded reports. No fitting or optimization occurs here."""
import base64
import copy
import hashlib
import html
import io
import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path


def serializable(value):
    if isinstance(value,dict):
        return {str(k):serializable(v) for k,v in value.items() if not callable(v)}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    if hasattr(value,'tolist'):return serializable(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):
        return '+∞' if value==float('inf') else '-∞' if value==float('-inf') else None
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'报告不支持的数据类型：{type(value).__name__}')


def inputs(project):
    return serializable({'factors':[asdict(f) for f in project.factors],
        'responses':[asdict(r) for r in project.responses], 'method':project.design_method,
        'design_params':project.design_params,'matrix':project.doe_matrix,'surrogate_config':project.surrogate_config})


def fingerprint(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def results(project):
    return serializable({'screening':project.screening_result,'taguchi':project.taguchi_result,
        'rsm':project.rsm_result,'surrogate':project.surrogate_result,'workbench':project.workbench_result})


def record_operation(project,stage,status='完成',payload=None,message=''):
    source=inputs(project)
    record={'id':uuid.uuid4().hex,'time':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'stage':stage,'status':status,'message':str(message),'input_sha256':fingerprint(source),
        'input':source,'metadata':serializable(project.report_metadata),
        'payload':serializable(payload or {})}
    record['record_sha256']=fingerprint(record)
    project.report_history.append(copy.deepcopy(record))
    return record


def bundle(project):
    project.ensure_results_current()
    return {'format':'RAMS-report-archive','version':1,'created_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'metadata':serializable(project.report_metadata),'input':inputs(project),
        'results':results(project),'history':copy.deepcopy(project.report_history)}


def save_archive(data,path):
    """Atomic write; an interrupted export leaves the previous archive intact."""
    path=Path(path)
    fd,tmp=tempfile.mkstemp(prefix='.rams-report-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(serializable(data),f,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_archive(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict) or data.get('format')!='RAMS-report-archive' or data.get('version')!=1:
        raise ValueError('不是支持的 RAMS 报告档案（版本 1）')
    if not isinstance(data.get('input'),dict) or not isinstance(data.get('history'),list) or not isinstance(data.get('results'),dict):
        raise ValueError('报告档案缺少输入、结果或操作记录')
    seen=set()
    for rec in data['history']:
        if not isinstance(rec,dict) or not all(k in rec for k in ('id','time','stage','status','input','input_sha256','payload','record_sha256')):
            raise ValueError('操作记录不完整')
        if rec['id'] in seen or fingerprint(rec['input'])!=rec['input_sha256']:
            raise ValueError('操作记录编号重复或输入校验不一致')
        if fingerprint({k:v for k,v in rec.items() if k!='record_sha256'})!=rec['record_sha256']:
            raise ValueError('操作记录内容校验不一致')
        if not isinstance(rec['input'],dict) or not isinstance(rec['payload'],dict):
            raise ValueError('操作记录的输入或结果格式无效')
        seen.add(rec['id'])
    return data


def fmt(value):
    if value is None:return '未提供 / 不可估计'
    if isinstance(value,bool):return '是' if value else '否'
    if isinstance(value,float):return f'{value:.6g}'
    if isinstance(value,(list,dict,tuple)):return json.dumps(serializable(value),ensure_ascii=False)
    return str(value)


def table(title,rows):return {'type':'table','title':title,'rows':serializable(rows)}
def text(value):return {'type':'text','text':str(value)}


def build_report(data,kind='model',record_id=None,include_raw=False):
    history=data.get('history',[])
    if kind=='trial':
        record=next((r for r in history if r['id']==record_id),None)
        if record is None:raise ValueError('请选择一条试验或操作记录')
        source=record['input'];metadata=record.get('metadata',{});payload=record['payload']
        res=payload.get('results',{})
        if 'workbench' in payload:res={**res,'workbench':payload['workbench']}
        selected_history=[record];status=f"{record['stage']}：{record['status']}"
        title='试验与分析记录报告';created=record['time']
    else:
        source=data['input'];metadata=data.get('metadata',{});res=data.get('results',{})
        selected_history=history;status='当前输入版本的模型报告';title='产品稳定性设计模型报告';created=data.get('created_utc','')
    sha=fingerprint(source)
    wb=res.get('workbench') or {}
    optimization=wb.get('optimization') or {}
    best=optimization.get('best') or {}
    sections=[]
    def section(key,title,blocks):sections.append({'id':key,'title':title,'blocks':blocks})
    matrix=source.get('matrix') or [];responses=source.get('responses') or [];factors=source.get('factors') or []
    complete=[]
    for response in responses:
        name=response.get('name','')
        if not name:continue
        vals=[]
        for row in matrix:
            try:v=float(row.get(name))
            except (ValueError,TypeError):continue
            if math.isfinite(v):vals.append(v)
        complete.append({'响应':name,'有效数':len(vals),'总行数':len(matrix),'缺失或无效':len(matrix)-len(vals)})
    analysis_present=bool(wb.get('responses') or wb.get('point_evaluation') or (res.get('rsm') or {}).get('fit') or
        (res.get('surrogate') or {}).get('responses') or (res.get('screening') or {}).get('response_analysis') or
        (res.get('taguchi') or {}).get('response_analysis'))
    risks=['模型计算不等于工程实测确认；是否满足最终产品要求需结合独立确认试验。']
    if not analysis_present:risks.append('当前版本尚无完整分析结果，不生成“验证通过”或最优取值结论。')
    if not matrix:risks.append('尚无 DOE 数据。')
    if any(r['缺失或无效'] for r in complete):risks.append('存在缺失或非有限响应，需补齐后重新分析。')
    if any(r['input_sha256']!=sha for r in history):risks.append('历史记录中存在其他输入版本，仅供追溯，不作为本版取值依据。')
    if kind=='model':
        not_passed=[r for r in history if r['input_sha256']==sha and r['status'] in ('失败','未通过','部分完成')]
        if not_passed:
            risks.append('本输入版本保留的未通过 / 失败记录：'+'；'.join(f"{r['stage']}（{r['id'][:12]}，{r['status']}）" for r in not_passed)+'。请结合后续修正与确认记录解释。')
    if best and (best.get('infeas') or 0)>0:risks.append('优化返回点违反设定约束，不可作为可行设计推荐。')
    if kind=='trial' and record['status']!='完成':risks.append(f"该记录状态为{record['status']}：{record.get('message','')}")
    for family in ('screening','taguchi'):
        for name,error in (res.get(family) or {}).get('errors',{}).items():risks.append(f'{name}：{error}')
    if source.get('surrogate_config') and analysis_present:
        risks.append('代理模型检验使用留出集，不套用 OLS 系数显著性；训练精度不代表整个设计域的预测精度。')
    summary=[{'项目':metadata.get('project_name') or '未填写','报告类型':title,'状态':status,
        '设计方法':source.get('method') or '未设置','因子数':len(factors),'响应数':len(responses),'DOE行数':len(matrix),
        '可追溯记录数':len(selected_history)}]
    section('summary','摘要与结论',[table('报告摘要',summary),text('本报告仅汇集已记录的数据与结果，不在报告页重新拟合或优化。'),
        text('当前有分析结果，工程确认状态：待确认。' if analysis_present else '当前证据不足，报告为待完善状态。')])
    section('purpose','目的与验证条件',[text(metadata.get('purpose') or '目的尚未填写。'),
        table('响应定义',[{'指标':r.get('name'),'角色':r.get('kind'),'目标特征':r.get('feature'),'下界':r.get('lower'),
        '上界':r.get('upper'),'单侧阈值':r.get('robust_limit'),'单位':r.get('unit')} for r in responses])])
    section('materials','材料、对象与输入',[text(metadata.get('materials') or '样机、材料、设备和试验环境尚未填写。'),
        table('因子定义',[{'因子':f.get('name'),'来源':f.get('source'),'不确定性':f.get('uncertainty'),'分布':f.get('distribution'),
        'P1':f.get('param1'),'P2':f.get('param2'),'固定':f.get('is_fixed'),'固定值':f.get('fixed_value'),'单位':f.get('unit')} for f in factors]),
        text('区间因子的 P1/P2 为上下界；概率因子的 P1/P2 为均值/方差。')])
    params=source.get('design_params') or {}
    blocks=[text('设计方法：'+(source.get('method') or '未设置')),
        table('DOE 配置',[{'配置项':k,'值':v} for k,v in (params.get('meta') or params).items() if k!='groups']),
        table('响应完整性',complete)]
    if source.get('surrogate_config'):blocks.append(table('代理模型配置',[{'配置项':k,'值':v} for k,v in source['surrogate_config'].items()]))
    if optimization:blocks.append(table('优化条件',[{'模式':optimization.get('mode'),'均值权重':optimization.get('weight'),
        '输入 k':optimization.get('k_design'),'约束 k':optimization.get('k_constraint'),
        '可行候选数':optimization.get('feasible_count'),'评价数':optimization.get('evaluated_count'),
        '种群数':optimization.get('population_size'),'代数':optimization.get('generations'),'种子':optimization.get('seed')}]))
    if kind=='trial':
        blocks.append(text('操作说明：'+record.get('message','')))
        if payload.get('settings'):blocks.append(table('当时的运行设置',[{'设置':k,'值':v} for k,v in payload['settings'].items()]))
    section('methods','方法、数据与配置',blocks)
    blocks=[];charts=[]
    if wb.get('data'):blocks.append(table('描述统计',wb['data']))
    for name,item in wb.get('responses',{}).items():
        blocks += [table(f'{name} · 拟合与验证统计',item.get('summary',[])),table(f'{name} · 系数与区间',item.get('coefficients',[]))]
        blocks.extend(text(note) for note in item.get('notes',[]))
        observations=item.get('observations',[])
        if observations:
            charts.append({'kind':'prediction','title':f'{name}：预测与实测（{item.get("source","")}）',
                'x':[r['实测'] for r in observations],'y':[r['预测'] for r in observations]})
        if include_raw:blocks.append(table(f'{name} · 逐点诊断',observations))
    if not wb.get('responses'):
        for name,item in ((res.get('rsm') or {}).get('fit') or {}).items():
            blocks.append(table(f'{name} · 二次模型',[{'R²':item.get('r2'),'调整R²':item.get('adj_r2'),'RMSE':item.get('rmse'),'残差自由度':item.get('df_residual')}]))
        for name,item in (res.get('surrogate') or {}).get('responses',{}).items():
            if item.get('error'):blocks.append(text(f'{name}：{item["error"]}'));risks.append(f'{name} 训练失败：{item["error"]}');continue
            blocks.append(table(f'{name} · 留出验证',[{'训练样本':item.get('train_samples'),'验证样本':item.get('test_samples'),'验证R²':item.get('r2_test'),'相对误差%':item.get('mape_test_%')}]))
            if item.get('actual') and len(item['actual'])==len(item.get('predicted',[])):
                charts.append({'kind':'prediction','title':f'{name}：留出集预测与实测','x':item['actual'],'y':item['predicted']})
    for name,item in (res.get('screening') or {}).get('response_analysis',{}).items():
        effects=item.get('effects',[])
        blocks.append(table(f'{name} · 筛选主效应',[{'因子':r['name'],'高减低效应':r['effect'],'效应份额%':r['share'],
            '标准误':r.get('se'),'未校正p':r.get('p')} for r in effects]))
        blocks += [text(item.get('note','')),text(f"中心与角点均值差：{fmt(item.get('curvature'))}")]
        charts.append({'kind':'bar','title':f'{name}：有符号主效应','x':[r['name'] for r in effects],'y':[r['effect'] for r in effects]})
    if res.get('screening'):
        blocks.extend(text(v) for v in res['screening'].get('aliases',[]))
        risks.append('筛选效应份额不等于因果贡献或统计显著性，须结合交互混杂解释。')
    for name,item in (res.get('taguchi') or {}).get('response_analysis',{}).items():
        blocks.append(table(f'{name} · 水平分析',[{'因子':r['factor'],'水平':r['level_value'],'均值':r['mean'],
            '平均组内标准差':r.get('std'),item.get('metric_label','指标'):r['metric'],'候选':r.get('recommended')} for r in item.get('levels',[])]))
        blocks.append(table(f'{name} · 内表观测',[{'内表组':r['inner'],'观测数':r['n'],'均值':r['mean'],'标准差':r.get('std'),'指标':r['metric'],'设置':r['settings']} for r in item.get('runs',[])]))
        blocks.append(text(item.get('note','')))
    if res.get('taguchi'):risks.append('田口候选采用主效应可加假设，必须做确认试验；不同目标有冲突时不能直接合并。')
    if wb.get('anova_text'):blocks.append(text(wb['anova_text']))
    if wb.get('validation_text'):blocks.append(text(wb['validation_text']))
    if not blocks:blocks=[text('尚无分析结果。')]
    blocks += [{'type':'chart',**c} for c in charts]
    section('results','分析结果与推论依据',blocks)
    blocks=[]
    if best:
        blocks.append(table('优化返回参数',[{'因子':k,'取值':v} for k,v in best.get('x',{}).items()]))
        blocks.append(table('优化返回点响应',best.get('responses',[])))
        blocks.append(text('模型约束判定：'+('违反约束，不能作为可行推荐。' if (best.get('infeas') or 0)>0 else '按记录的数值精度满足设定约束；需独立确认。')))
        front=optimization.get('front') or []
        if front and len(front[0].get('obj',[]))>=2:
            blocks.append({'type':'chart','kind':'pareto','title':'可行帕累托前沿：望性损失与归一化波动',
                'x':[r['obj'][0] for r in front],'y':[r['obj'][1] for r in front]})
    for name,candidate in (res.get('taguchi') or {}).get('best_levels_by_response',{}).items():
        blocks.append(table(f'{name} · 待确认水平',[{'因子':k,'水平':v} for k,v in candidate.items()]))
    if (res.get('taguchi') or {}).get('recommendation'):blocks.append(text(res['taguchi']['recommendation']))
    if wb.get('parameter_design',{}).get('text'):blocks.append(text(wb['parameter_design']['text']))
    for name,rows in wb.get('tolerance',{}).items():blocks.append(table(f'{name} · 局部容差贡献',[{'来源':r['name'],'方差':r.get('variance'),'份额%':r['share']} for r in rows]))
    if wb.get('point_evaluation'):
        evaluation=wb['point_evaluation'];blocks += [text('手动核验点：'+fmt(evaluation.get('point'))),table('手动点约束核验',evaluation.get('responses',[])),text(evaluation.get('note',''))]
    if wb.get('local_search'):
        local=wb['local_search'];blocks += [text(local.get('text','')),table('局部探索接受路径',[{'设置':r['x'],'望性':r['mean'],'归一标准差':r['std'],'约束违反':r['violation'],'损失':r['score']} for r in local.get('path',[])])]
    if wb.get('model_text'):blocks.append(text(wb['model_text']))
    if not blocks:blocks=[text('尚无参数取值结果。筛选设计只提供因子筛选结论，不输出连续最优点。')]
    section('settings','参数取值、稳定性与容差',blocks)
    if optimization or wb:risks.append('μ±kσ 是独立输入、对角二阶近似下的稳健带，不是预测区间，不保证概率覆盖。')
    section('risks','风险、未通过项与确认计划',[text(r) for r in dict.fromkeys(risks)]+[
        text('人工结论 / 复核意见：'+(metadata.get('conclusion') or '未填写，未自动判定工程验收通过。')),
        text('确认计划：'+(metadata.get('confirmation') or '建议对候选参数开展独立实测，复核均值、波动和约束；记录未达标原因与修正方案。'))])
    section('history','试验记录与版本追溯',[table('操作记录',[{'记录ID（前12位）':r['id'][:12],'时间UTC':r['time'].replace('T',' ').replace('+00:00',''),'阶段':r['stage'],
        '状态':r['status'],'输入版本':r['input_sha256'][:12],'适用于本版':r['input_sha256']==sha,'说明':r.get('message','')} for r in selected_history]),
        text('记录从本功能启用后开始；此前未记录的操作无法追补。完成表示计算或录入完成，不表示工程验证通过。'),
        text('输入 SHA-256：'+sha)])
    if include_raw or kind=='trial':section('appendix','附录：原始 DOE 数据',[table('DOE 与回填响应',matrix)])
    return {'title':title,'kind':kind,'record_id':record_id,'created_utc':created,'input_sha256':sha,
        'metadata':metadata,'status':status,'sections':sections,'risks':list(dict.fromkeys(risks))}


def chart_png(block):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib
    with matplotlib.rc_context({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False}):
        fig=Figure(figsize=(7.2,3.3),dpi=125,layout='constrained');FigureCanvasAgg(fig);ax=fig.add_subplot()
        x,y=block['x'],block['y']
        if block['kind']=='bar':ax.bar(x,y,color='#245f9e');ax.axhline(0,color='gray',lw=.6);ax.tick_params(axis='x',rotation=25)
        else:
            ax.scatter(x,y,s=20,color='#245f9e')
            if block['kind']=='prediction':
                lo,hi=min(x+y),max(x+y);ax.plot([lo,hi],[lo,hi],'--',color='gray');ax.set(xlabel='实测',ylabel='预测')
            else:ax.set(xlabel='望性损失',ylabel='归一化波动')
        ax.set_title(block['title'],fontsize=11);ax.grid(alpha=.2)
        out=io.BytesIO();fig.savefig(out,format='png');return out.getvalue()


def render_html(report,section_ids=None):
    esc=lambda value:html.escape(fmt(value),quote=True).replace('\n','<br/>')
    selected=set(section_ids) if section_ids is not None else None
    meta=report['metadata']
    chunks=['<!doctype html><html><head><meta charset="utf-8"/><style>body{font-family:"Microsoft YaHei",sans-serif;color:#243449;font-size:10pt;} h1{font-size:22pt;color:#17385e;} h2{font-size:15pt;color:#17385e;margin-top:24px;} h3{font-size:11pt;color:#245f9e;} table{border-collapse:collapse;width:100%;font-size:9pt;} th{background:#e7eef7;} td,th{border:1px solid #cbd5e1;padding:5px;text-align:left;} p{line-height:1.5;} .meta{color:#50657d;} </style></head><body>',
        f'<h1>{esc(report["title"])}</h1>',f'<p class="meta">项目：{esc(meta.get("project_name") or "未填写")}<br/>编制：{esc(meta.get("author") or "未填写")}<br/>时间 UTC：{esc(report["created_utc"])}<br/>输入版本：{esc(report["input_sha256"][:12])}　状态：{esc(report["status"])}</p>']
    for section in report['sections']:
        if selected is not None and section['id'] not in selected:continue
        chunks.append(f'<h2 id="{section["id"]}">{esc(section["title"])}</h2>')
        for block in section['blocks']:
            if block['type']=='text':chunks.append(f'<p>{esc(block["text"])}</p>')
            elif block['type']=='chart':
                encoded=base64.b64encode(chart_png(block)).decode('ascii')
                chunks.append(f'<p><img width="610" height="280" src="data:image/png;base64,{encoded}" alt="{esc(block["title"])}"/></p>')
            else:
                rows=block['rows'];chunks.append(f'<h3>{esc(block["title"])}</h3>')
                if not rows:chunks.append('<p>无可用记录。</p>');continue
                keys=list(dict.fromkeys(k for row in rows for k in row))
                # Long statistical tables are split vertically into field groups, keeping the first identifier.
                groups=[keys] if len(keys)<=8 else [[keys[0]]+keys[i:i+7] for i in range(1,len(keys),7)]
                for columns in groups:
                    chunks.append('<table width="100%" cellspacing="0" cellpadding="4"><thead><tr>'+''.join(f'<th>{esc(k)}</th>' for k in columns)+'</tr></thead><tbody>')
                    for row in rows:chunks.append('<tr>'+''.join(f'<td>{esc(row.get(k))}</td>' for k in columns)+'</tr>')
                    chunks.append('</tbody></table><br/>')
    chunks.append('<p class="meta">RAMS · 可追溯分析报告 · 计算结论需按确认计划复核</p></body></html>')
    return ''.join(chunks)


def export_pdf(html_text,path):
    """Use the application's existing Qt dependency; text remains selectable."""
    from PyQt6.QtCore import QRectF,QMarginsF,Qt
    from PyQt6.QtGui import QPdfWriter,QPageSize,QTextDocument,QPainter,QFont,QAbstractTextDocumentLayout
    writer=QPdfWriter(str(path));writer.setResolution(96);writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(15,15,15,15));writer.setTitle('RAMS 稳定性设计分析报告')
    doc=QTextDocument();doc.setDefaultFont(QFont('Microsoft YaHei',10));doc.documentLayout().setPaintDevice(writer)
    doc.setHtml(html_text);width=writer.width();height=writer.height()-38
    from PyQt6.QtCore import QSizeF
    doc.setPageSize(QSizeF(width,height))
    painter=QPainter(writer)
    if not painter.isActive():raise OSError('PDF 文件无法写入，请检查路径和文件占用')
    try:
        count=doc.pageCount()
        for i in range(count):
            if i and not writer.newPage():raise OSError('PDF 新页面写入失败')
            painter.save();painter.setClipRect(QRectF(0,0,width,height));painter.translate(0,-i*height)
            context=QAbstractTextDocumentLayout.PaintContext();context.clip=QRectF(0,i*height,width,height)
            doc.documentLayout().draw(painter,context);painter.restore()
            painter.setFont(QFont('Microsoft YaHei',8));painter.drawText(QRectF(0,height+12,width,22),Qt.AlignmentFlag.AlignRight,f'RAMS · {i+1} / {count}')
    finally:painter.end()


def build_current_report(data):
    """Render only this calculation, without trial history or authoring workflow."""
    current = dict(data, history=[])
    report = build_report(current)
    report['title'] = '产品稳定性设计结果报告'
    report['status'] = '本次计算结果'
    optimization = (data.get('results', {}).get('workbench') or {}).get('optimization') or {}
    if optimization.get('sigma_limits'):
        section = next(s for s in report['sections'] if s['id'] == 'settings')
        section['blocks'].append(table('独立波动约束（响应原单位）', optimization['best'].get('sigma_constraints', [])))
    report['sections'] = [s for s in report['sections'] if s['id'] != 'history']
    for section in report['sections']:
        if section['id'] == 'risks':
            section['title'] = '结果适用范围与确认建议'
            section['blocks'] = [b for b in section['blocks'] if not b.get('text', '').startswith('人工结论')]
        if section['id'] in ('purpose', 'materials'):
            section['blocks'] = [b for b in section['blocks'] if b['type'] != 'text' or '尚未填写' not in b.get('text', '')]
        for block in section['blocks']:
            if block['type'] == 'table':
                for row in block['rows']:
                    row.pop('可追溯记录数', None)
                    if '报告类型' in row:
                        row['报告类型'] = report['title']
                        row['状态'] = report['status']
    return report
