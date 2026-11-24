"""Operator view: select a machine, confirm six targets, export four radar mounts."""
from pathlib import Path
from itertools import combinations
from queue import Empty, Queue
from threading import Event, Thread
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import yaml
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from v02_results import Results
from v02_batch import (MACHINE_FILE, PARAMS, ROI_FILE, SLOTS, calibrate, corners, discover,
                       preview, valid_bounds, write_yaml)

POSES = {'F3d':'正前 3 m','0.3L':'左偏 0.3 m','0.3R':'右偏 0.3 m'}
FIELDS = ('内点阈值 m','最少内点','最多平面','迭代次数','最小夹角 °','最小跨度 m','最大跨度 m')
EDGES=[(i,j) for i,j in combinations(range(8),2) if (i^j) in (1,2,4)]


def draw(ax, p, bounds, axes=(0,1)):
    ax.clear(); a,b=axes
    points=p.points; ax.scatter(points[:,a],points[:,b],s=1,c='#496478',linewidths=0)
    for vertices,color,style in [(corners(p.search),'#9cabb2','--'),(p.pose.apply(corners(bounds)),'#008267','-')]:
        xy=vertices[:,[a,b]];ax.add_collection(LineCollection([xy[[i,j]] for i,j in EDGES],colors=color,linewidths=1.5,linestyles=style))
    center=p.search.mean(1); width=max(3.6,np.ptp(p.search,axis=1).max()+.5)
    for axis,dim in ((ax.set_xlim,a),(ax.set_ylim,b)):axis(center[dim]-width/2,center[dim]+width/2)
    ax.set_aspect('equal');ax.tick_params(labelsize=7);ax.grid(alpha=.18)
    ax.set_xlabel('XYZ'[a]+' (m)',fontsize=8);ax.set_ylabel('XYZ'[b]+' (m)',fontsize=8)


class BatchViewer:
    def __init__(self, root):
        self.root=root;root.title('LiDAR Cube Viewer V02');root.geometry('1160x800')
        self.roi_window = None
        self.parent=None;self.models={};self.notes={};self.output=None;self.busy=False
        self.cache={};self.events=Queue();self.stop=Event();self.axes={};self.cancelled=False
        top=ttk.Frame(root,padding=8);top.pack(fill='x');ttk.Button(top,text='选择数据目录…',command=self.choose).pack(side='left')
        self.folder=tk.StringVar(value='请选择含六个 bag 的数据目录');ttk.Label(top,textvariable=self.folder).pack(side='left',padx=8)
        self.figure=Figure(figsize=(10,5),dpi=85);self.figure.subplots_adjust(left=.06,right=.98,top=.90,bottom=.10,hspace=.46,wspace=.17)
        self.canvas=FigureCanvasTkAgg(self.figure,master=root);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.canvas.mpl_connect('button_press_event',self.click)
        for idx,k in enumerate(SLOTS):self.axes[k]=self.figure.add_subplot(2,3,idx+1)
        ttk.Label(root,text='点击预览可调整框或更换包。虚线为搜索范围，绿色实线为累计范围。').pack(anchor='w',padx=12)
        self.params={k:tk.StringVar(value=str(v)) for k,v in PARAMS.items()};self.param_area=ttk.Frame(root);self.fields=ttk.Frame(self.param_area)
        self.param_area.pack(fill='x',padx=8);ttk.Button(self.param_area,text='平面拟合参数（六包共用）',command=self.toggle_params).pack(anchor='w')
        for i,((key,var),label) in enumerate(zip(self.params.items(),FIELDS)):
            ttk.Label(self.fields,text=label).grid(row=0,column=i,padx=3);ttk.Entry(self.fields,textvariable=var,width=10).grid(row=1,column=i,padx=3)
        bottom=ttk.Frame(root,padding=8);bottom.pack(fill='x');self.start_button=ttk.Button(bottom,text='确认全部并开始标定',command=self.start);self.start_button.pack(side='left')
        for title,item in [('打开结果目录',''),('打开最终外参','suggested_extrinsics.yaml')]:
            ttk.Button(bottom,text=title,command=lambda item=item:self.open_output(item)).pack(side='left',padx=4)
        self.status=tk.StringVar(value='等待选择数据目录');ttk.Label(root,textvariable=self.status,wraplength=1050).pack(anchor='w',padx=10,pady=5)
        self.results = Results(root)
        self.results.pack(side='bottom', fill='x', padx=8, pady=(0, 6))
        self.run_started = False
        root.protocol('WM_DELETE_WINDOW',self.close);self.refresh();root.after(100,self.poll)

    def report(self, message):
        self.status.set(message)

    def refresh(self):
        for k,ax in self.axes.items():
            p=self.models.get(k);ax.clear();ax.set_axis_on()
            if p:draw(ax,p,p.bounds)
            else:ax.set_axis_off()
            group,pose=k.split('_');ax.set_title(group+'  /  '+pose+'   '+('!' if (not p or p.manual) else 'OK'),loc='left',fontsize=11)
        self.canvas.draw_idle()

    def toggle_params(self):
        if self.fields.winfo_manager():self.fields.pack_forget()
        else:self.fields.pack(anchor='w',pady=4)

    def choose(self):
        if self.busy or self.raise_roi_editor():return
        path=filedialog.askdirectory(parent=self.root,initialdir=self.parent or Path.home(),title='选择数据目录')
        if path:self.load_machine(Path(path))

    def load_machine(self,path):
        if self.busy:return
        self.cache.clear();self.parent=path.resolve();self.folder.set(str(self.parent));self.models={};self.notes={};self.output=None;self.refresh()
        try:
            self.results.new(self.parent.name)
            self.run_started = False
            self.report(f'已选择数据目录　{self.parent}')
            state=yaml.safe_load((path/MACHINE_FILE).read_text()) if (path/MACHINE_FILE).is_file() else {}
            found=discover(path); selected={}
            for k,v in self.params.items():v.set(str(state.get('params',{}).get(k,PARAMS[k])))
            for k in SLOTS:
                stored=state.get('bags',{}).get(k,'');candidate=path/stored
                if stored and candidate.parent==path and candidate.is_dir():selected[k]=candidate
                elif len(found[k])==1:selected[k]=found[k][0]
                else:self.notes[k]='缺包或重名，点击补选'
            self.load_slots(selected)
        except Exception as exc:self.report(str(exc))

    def load_slots(self,selected):
        def work():
            for k,path in selected.items():
                if self.stop.is_set():return
                self.events.put(('progress',f'正在准备 {k} 前雷达预览'))
                try:self.events.put(('preview',(k,preview(path,k),'')))
                except Exception as exc:self.events.put(('preview',(k,None,str(exc))))
            self.events.put(('ready',None))
        self.busy=True;Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            while True:
                key,value=self.events.get_nowait()
                if key=='progress':self.status.set(value)
                elif key=='bag':self.results.current.bag(*value)
                elif key=='mount':self.results.current.mount(*value)
                elif key=='preview':
                    k,p,msg=value; self.notes[k]=msg
                    if p:
                        self.models[k]=p
                        self.report(f'{k} · {p.note}　{p.bag.name}')
                    else:
                        self.models.pop(k,None)
                        self.report(f'{k} · 预览失败　{msg}')
                    self.refresh()
                elif key=='ready':
                    self.busy=False;need=[k for k in SLOTS if k not in self.models or self.models[k].manual];self.report('请点击检查 '+', '.join(need) if need else '六个摆位已就绪，请核对绿色累计框。')
                elif key=='done':
                    self.busy=False;self.output,fail=value;self.report('已导出四雷达外参  '+str(self.output) if not fail else '复验失败，请调整后重试  '+', '.join(fail)+'　结果目录 '+str(self.output))
                elif key=='error':self.busy=False;self.report(str(value))
        except Empty:pass
        self.root.after(100,self.poll)

    def click(self,event):
        if self.busy or self.parent is None:return
        for k,ax in self.axes.items():
            if ax==event.inaxes:self.edit(k);break

    def raise_roi_editor(self):
        if self.roi_window is None or not self.roi_window.winfo_exists():
            return False
        self.roi_window.deiconify()
        self.roi_window.lift()
        return True

    def edit(self,k):
        if self.raise_roi_editor():
            return self.roi_window
        p = self.models.get(k)
        win = tk.Toplevel(self.root)
        self.roi_window = win
        win.title(k + ' ROI')
        win.transient(self.root)

        ttk.Label(win,text=(str(p.bag)+'  ·  '+p.note if p else self.notes.get(k,'请选择 bag'))).pack(padx=8,pady=5)
        def replace():
            path=filedialog.askdirectory(parent=win,initialdir=self.parent,title=k+' 选择数据包目录')
            if not path:return
            bag=Path(path).resolve()
            if bag.parent!=self.parent or any(x.bag==bag for key,x in self.models.items() if key!=k):
                self.report('请选择本机目录下未被其他格使用的数据包');return
            win.destroy();self.output=None;self.load_slots({k:bag})
        ttk.Button(win,text='更换此包…',command=replace).pack(anchor='e',padx=8)
        if not p:return win
        bounds=p.bounds.copy();fig=Figure(figsize=(8,4.4));fig.subplots_adjust(bottom=.16);canvas=FigureCanvasTkAgg(fig,master=win);canvas.get_tk_widget().pack(fill='both',expand=True)
        top,side=fig.subplots(1,2);drag={};step=tk.StringVar(value='0.05')
        def redraw():
            for ax,idx,label in ((top,(0,1),'XY'),(side,(0,2),'XZ')):draw(ax,p,bounds,idx);ax.set_title(label)
            canvas.draw_idle()
        def press(ev):
            if ev.inaxes in (top,side) and ev.button==1:drag.update(ax=ev.inaxes,start=(ev.xdata,ev.ydata),original=bounds.copy())
        def move(ev):
            if not drag or ev.inaxes!=drag['ax']:return
            dx,dy=np.array([ev.xdata,ev.ydata])-drag['start'];d=[dx,dy,0] if ev.inaxes==top else [dx,0,dy]
            bounds[:]=drag['original']+p.pose.rotation.inv().apply(d)[:,None];redraw()
        canvas.mpl_connect('button_press_event',press);canvas.mpl_connect('motion_notify_event',move);canvas.mpl_connect('button_release_event',lambda _:drag.clear());redraw()
        bar=ttk.Frame(win,padding=6);bar.pack(fill='x');ttk.Label(bar,text='拖动平移框；每侧调整（m）').pack(side='left');ttk.Entry(bar,textvariable=step,width=6).pack(side='left')
        def resize(sign):
            try:
                n=float(step.get());assert np.isfinite(n) and n>0;new=valid_bounds(bounds+sign*n*np.array([-1,1]));bounds[:]=new;redraw()
            except (ValueError,AssertionError):self.report('调整量无效或 ROI 尺寸太小')
        for text,sign in [('扩大',1),('缩小',-1)]:ttk.Button(bar,text=text,command=lambda s=sign:resize(s)).pack(side='left')
        def confirm():
            p.bounds=bounds.copy();p.manual=False;p.note='已人工确认';self.save_roi(p)
            self.report(f'{k} · 已人工确认并保存 ROI');self.output=None;self.refresh();win.destroy()
        ttk.Button(bar,text='确认此 ROI',command=confirm).pack(side='right');return win

    @staticmethod
    def save_roi(p):write_yaml(p.bag/ROI_FILE,dict(slot=p.slot,bag=p.bag.name,bounds=p.bounds.tolist()))

    def start(self):
        if self.busy or self.raise_roi_editor():return
        need=[k for k in SLOTS if k not in self.models or self.models[k].manual]
        if need:self.report('请先点击并确认 '+', '.join(need));return
        previews=[self.models[k] for k in SLOTS]
        try:
            params={k:type(v)(self.params[k].get()) for k,v in PARAMS.items()};validate_params(params)
            for p in previews:self.save_roi(p)
            write_yaml(self.parent/MACHINE_FILE,dict(params=params,bags={p.slot:p.bag.name for p in previews}))
        except Exception as exc:self.report('无法开始：'+str(exc));return
        if self.run_started:
            self.results.new(self.parent.name)
        self.run_started = True
        self.output=None;self.busy=True;self.report(f'开始新一轮标定 · 六个 ROI 已保存，内点阈值 {params["plane_threshold"]} m，迭代 {params["plane_max_iterations"]} 次')
        def work():
            try:self.events.put(('done',calibrate(previews,params,self.parent,lambda m:self.events.put(('progress',m)),self.stop.is_set,self.cache,on_result=lambda kind, payload:self.events.put((kind,payload)))))
            except Exception as exc:self.events.put(('error',str(exc)))
        Thread(target=work,daemon=True).start()

    def open_output(self,item):
        if self.output and (self.output/item).exists():subprocess.Popen(['xdg-open',str(self.output/item)])
        else:self.report('尚无可打开的结果')

    def close(self):self.stop.set();self.root.destroy()


def validate_params(params):
    if not all(np.isfinite(v) for v in params.values()) or not (params['plane_threshold']>0 and params['plane_min_inliers']>=3 and params['plane_max_planes']==3 and params['plane_max_iterations']>0 and 0<params['plane_min_angle']<90 and 0<=params['plane_min_span']<params['plane_max_span']):
        raise ValueError('请检查拟合参数；三面体需要 3 个面，最小跨度应小于最大跨度。')


def main():
    root=tk.Tk();BatchViewer(root);root.mainloop()
