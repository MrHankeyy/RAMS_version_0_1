"""Response-level diagnostic tables, plots and interactive design evaluation."""
import csv
import hashlib
from datetime import datetime, timezone
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy import stats
from PyQt6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,
    QTabWidget,QTableWidget,QTableWidgetItem,QHeaderView,QTextEdit,QFileDialog,QMessageBox,
    QDoubleSpinBox,QScrollArea,QFormLayout)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import workbench_analysis as wa
import optimizer_engine as oe


class WorkbenchDiagnostics(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.context=None;self.snapshot={};self.tables={};self.controls={}
        layout=QVBoxLayout(self)
        head=QHBoxLayout();head.addWidget(QLabel('响应'))
        self.response=QComboBox();head.addWidget(self.response,1)
        for title,slot in [('导出当前表 CSV',self.export_csv),('导出完整统计 JSON',self.export_json),('导出诊断图',self.export_plot)]:
            button=QPushButton(title);button.clicked.connect(slot);head.addWidget(button)
        layout.addLayout(head)
        self.notice=QLabel('请先执行“仅建模与统计分析”或鲁棒优化。');self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.tabs=QTabWidget();layout.addWidget(self.tabs)
        for key,title in [('data','数据概览'),('summary','拟合统计'),('coefficients','系数与区间'),('observations','逐点诊断')]:
            table=QTableWidget();table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setAlternatingRowColors(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.tables[key]=table;self.tabs.addTab(table,title)
        self.figure=Figure(figsize=(9,6),tight_layout=True)
        self.canvas=FigureCanvasQTAgg(self.figure);self.tabs.addTab(self.canvas,'诊断图')
        self.notes=QTextEdit();self.notes.setReadOnly(True);self.tabs.addTab(self.notes,'统计解释')
        self.point_tab=QWidget();point_layout=QVBoxLayout(self.point_tab)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMaximumHeight(220)
        self.form_widget=QWidget();self.form=QFormLayout(self.form_widget);scroll.setWidget(self.form_widget)
        point_layout.addWidget(scroll)
        bar=QHBoxLayout()
        for title,slot in [('填入推荐点',self.use_recommended),('计算当前设计点',self.evaluate_point)]:
            button=QPushButton(title);button.clicked.connect(slot);bar.addWidget(button)
        point_layout.addLayout(bar)
        self.point_note=QLabel('');self.point_note.setWordWrap(True);point_layout.addWidget(self.point_note)
        self.point_table=QTableWidget();self.point_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.point_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        point_layout.addWidget(self.point_table);self.tabs.addTab(self.point_tab,'设计点与约束核验')
        self.response.currentTextChanged.connect(self.render)

    @staticmethod
    def fill(table,rows):
        headers=list(dict.fromkeys(k for row in rows for k in row))
        table.setColumnCount(len(headers));table.setHorizontalHeaderLabels(headers);table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            for j,key in enumerate(headers):
                value=row.get(key)
                text='—' if value is None else f'{value:.6g}' if isinstance(value,float) else str(value)
                table.setItem(i,j,QTableWidgetItem(text))

    def clear(self):
        self.context=None;self.snapshot={};self.response.clear()
        for table in self.tables.values():table.setRowCount(0)
        self.point_table.setRowCount(0);self.point_note.clear();self.notes.clear()
        self.figure.clear();self.canvas.draw()
        self.notice.setText('输入或模型已更新，请重新计算统计分析。')

    def set_context(self,context,robust=None,kd=6,kc=6):
        self.context=context;self.robust=robust or {};self.kd=kd;self.kc=kc
        self.snapshot={'created_utc':datetime.now(timezone.utc).isoformat(),
                       'input_sha256':hashlib.sha256(context['_project'].analysis_signature().encode('utf-8')).hexdigest(),
                       'inputs':[asdict(f) for f in context['inputs']],
                       'response_definitions':[asdict(r) for r in context['_responses']],
                       'model_types':{name:m.get('model_type','二次回归') for name,m in context['models'].items()},
                       'method':context['_project'].design_method,'k_design':kd,'k_constraint':kc,
                       'data':wa.descriptive(context['_project']),
                       'responses':{name:wa.diagnostics(context,name) for name in context['models']}}
        self.fill(self.tables['data'],self.snapshot['data'])
        self.response.blockSignals(True);self.response.clear();self.response.addItems(list(context['models']));self.response.blockSignals(False)
        while self.form.rowCount():self.form.removeRow(0)
        self.controls={}
        for f in oe.design_inputs(context):
            lo,hi=oe.de.factor_limits(f);spin=QDoubleSpinBox();spin.setDecimals(8);spin.setRange(lo,hi)
            spin.setSingleStep((hi-lo)/100);spin.setValue((hi+lo)/2)
            self.controls[f.name]=spin;self.form.addRow(f'{f.name} [{lo:g}, {hi:g}]',spin)
        self.point_table.setRowCount(0);self.point_note.clear()
        self.notice.setText(f'共 {len(context["models"])} 个响应模型。统计基于当前已训练模型；设计点评估使用 k输入={kd:g}，k约束={kc:g}。')
        self.render()

    def render(self,*args):
        item=self.snapshot.get('responses',{}).get(self.response.currentText())
        if not item:return
        for key in ('summary','coefficients','observations'):self.fill(self.tables[key],item[key])
        self.notes.setPlainText('\n\n'.join(item['notes']))
        self.figure.clear()
        rows=item['observations']
        if rows:
            y=np.array([r['实测'] for r in rows]);pred=np.array([r['预测'] for r in rows]);res=y-pred
            ax=self.figure.add_subplot(221);ax.scatter(y,pred,s=18)
            low=min(y.min(),pred.min());high=max(y.max(),pred.max());ax.plot([low,high],[low,high],'--',color='gray')
            ax.set(xlabel='实测',ylabel='预测',title='预测 vs 实测')
            ax=self.figure.add_subplot(222);ax.scatter(pred,res,s=18);ax.axhline(0,color='gray',ls='--')
            ax.set(xlabel='预测',ylabel='残差',title='残差 vs 预测')
            ax=self.figure.add_subplot(223)
            if len(res)>1:stats.probplot(res,dist='norm',plot=ax)
            ax.set(title='残差正态 Q-Q',xlabel='理论分位数',ylabel='观测分位数')
            ax=self.figure.add_subplot(224);ax.plot([r['序号'] for r in rows],res,'o-',ms=3)
            ax.axhline(0,color='gray',ls='--');ax.set(title='残差序列（非时间检验）',xlabel='数据序号',ylabel='残差')
        self.canvas.draw()

    def use_recommended(self):
        point=self.robust.get('best',{}).get('x',{}) if self.context else {}
        if not point:
            self.point_note.setText('尚无优化推荐点，可手动输入设计值。');return
        for name,spin in self.controls.items():spin.setValue(point.get(name,spin.value()))
        self.evaluate_point()

    def evaluate_point(self):
        if not self.context:return
        try:
            result=wa.point_analysis(self.context,{k:v.value() for k,v in self.controls.items()},self.kd,self.kc)
            self.snapshot['point_evaluation']=result
            import report_engine as reports
            project=self.context['_project']
            project.workbench_result.update(point_evaluation=reports.serializable(result))
            project.record_operation('设计点核验', '完成' if result['feasible'] else '未通过', {'workbench':reports.serializable(self.snapshot)}, '模型计算，非实测确认')
            self.fill(self.point_table,result['responses'])
            self.point_note.setText(('当前点满足设定约束。' if result['feasible'] else '当前点违反设定约束。')+'\n'+result['note'])
        except Exception as exc:
            self.point_table.setRowCount(0);self.snapshot.pop('point_evaluation',None)
            self.point_note.setText(str(exc))
            self.context["_project"].record_operation("设计点核验", "失败", message=str(exc))

    def export_csv(self):
        table=self.tabs.currentWidget()
        if table is self.point_tab:table=self.point_table
        if not isinstance(table,QTableWidget) or not table.rowCount():
            Notice.information(self,'暂无表格','请切换到有结果的统计表后导出。');return
        path,_=QFileDialog.getSaveFileName(self,'导出当前统计表','','CSV (*.csv)')
        if not path:return
        try:
            with open(path,'w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow([table.horizontalHeaderItem(i).text() for i in range(table.columnCount())])
                writer.writerows([[table.item(i,j).text() if table.item(i,j) else '' for j in range(table.columnCount())] for i in range(table.rowCount())])
        except OSError as exc:Notice.warning(self,'导出失败',str(exc))
        else:Notice.information(self,'导出成功',path)

    def export_json(self):
        if not self.context:return
        path,_=QFileDialog.getSaveFileName(self,'导出完整统计','','JSON (*.json)')
        if not path:return
        try:Path(path).write_text(json.dumps(wa.clean_json(self.snapshot),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        except OSError as exc:Notice.warning(self,'导出失败',str(exc))
        else:Notice.information(self,'导出成功',path)

    def export_plot(self):
        if not self.context:return
        path,_=QFileDialog.getSaveFileName(self,'导出当前响应的诊断图','','PNG (*.png);;SVG (*.svg)')
        if not path:return
        try:self.figure.savefig(path,dpi=180,bbox_inches='tight')
        except (OSError,ValueError) as exc:Notice.warning(self,'导出失败',str(exc))
        else:Notice.information(self,'导出成功',path)

from notifications import Notice
