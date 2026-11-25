"""V02 运行结果表。每轮保留独立的界面记录。"""
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from v02_charts import MetricChart, chart_rows, initial_deviation, match_detail
from v02_batch import RADAR_NAMES, SLOTS


def number(value, scale=1):
    return '—' if value is None else f'{value * scale:.2f}'


class RunTables(ttk.Notebook):
    def __init__(self, parent):
        super().__init__(parent)
        self.records = {}
        self.issues = []
        self.row_reasons = {}
        self.detail_window = None
        self.pages = []
        self.charts = {}
        self.planes = self.table('平面拟合', (
            '摆位', '雷达', '阶段', '累计帧数', 'ROI 点数', '滤波点数', '拟合面数', '最大直角差 °', '最大 RMS mm', '评级'))
        self.pairs = self.table('面面匹配', ('摆位', '配对', '阶段', '交点距离 cm', '平均法向夹角 °', '评级'))
        self.mounts = self.table('安装修正 · 相对 /tf_static', ('雷达', 'ΔX cm', 'ΔY cm', 'ΔZ cm', '位移 cm', '转角 °'))

    def table(self, title, columns):
        tab = ttk.Frame(self)
        self.add(tab, text=title)
        body = ttk.Frame(tab)
        body.pack(fill='both', expand=True)
        self.pages.append((tab, body))
        view = ttk.Treeview(body, columns=columns, show='headings', height=6)
        for col in columns:
            view.heading(col, text=col)
            view.column(col, width=100, minwidth=50, anchor='center')
        if '评级' in columns:
            view.bind('<Double-1>', lambda event: self.open_row_reason(view, event))
        for tag, color in [('PASS', '#176347'), ('WARNING', '#915b00'), ('FAIL', '#ad2626')]:
            view.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(body, command=view.yview)
        view.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        view.pack(fill='both', expand=True)
        return view

    def render(self, page, chart_mode, slot):
        tab, body = self.pages[page]
        if chart_mode and page < 2:
            body.pack_forget()
            if page not in self.charts:
                self.charts[page] = MetricChart(tab)
            chart = self.charts[page]
            chart.pack(fill='both', expand=True)
            chart.display(page, chart_rows(self.records, page, slot))
        else:
            if page in self.charts:
                self.charts[page].pack_forget()
            body.pack(fill='both', expand=True)

    def bag(self, slot, stage, stats, pairs, frames):
        for side, sensor in stats.items():
            fit = sensor['summary']
            if fit['quality'] == 'FAIL':
                self.add_issue(f"{slot} · {stage} · {RADAR_NAMES[side]}雷达拟合：{fit['reason']}")
            item = self.planes.insert('', 'end', tags=(fit['quality'],), values=(
                slot, RADAR_NAMES[side], stage, frames[side]['used_frames'], frames[side]['roi_points'],
                sensor['filtered_points'], fit['plane_count'], number(fit['max_right_angle_deviation_deg']),
                number(fit['max_plane_rms_m'], 1000), 'FAIL · 详情' if fit['quality']=='FAIL' else fit['quality']))
            if fit['quality'] == 'FAIL':
                self.row_reasons[self.planes, item] = self.issues[-1]
        for key, match in pairs.items():
            name = '–'.join(RADAR_NAMES[s] for s in key.split('-'))
            quality = match['quality']
            rating = quality
            tag = quality
            if quality == 'FAIL':
                detail = f'{slot} · {stage} · {name}匹配：' + match_detail(key, match, stage)
                if initial_deviation(stage, match):
                    rating, tag = '初始偏差 · 详情', 'WARNING'
                else:
                    rating = 'FAIL · 详情'
                    self.add_issue(detail)
            item = self.pairs.insert('', 'end', tags=(tag,), values=(
                slot, name, stage, number(match.get('corner_distance_m'), 100),
                number(match.get('normal_error_deg')), rating))
            if quality == 'FAIL':
                self.row_reasons[self.pairs, item] = detail

        self.records[slot, stage] = stats, pairs
        if self.master.displayed is self:
            self.master.update_issues()

        if self.master.displayed is self and self.master.mode.get() == '柱状图':
            self.master.refresh_view()

    def open_row_reason(self, view, event):
        reason = self.row_reasons.get((view, view.identify_row(event.y)))
        if reason:
            from tkinter import messagebox
            messagebox.showinfo('本行指标详情', reason, parent=self)

    def add_issue(self, message):
        self.issues.append(message)
        if self.detail_window is not None and self.detail_window.winfo_exists():
            self.detail_text.configure(state='normal')
            self.detail_text.insert('end', message + '\n\n')
            self.detail_text.configure(state='disabled')

    def show_issues(self):
        if self.detail_window is not None and self.detail_window.winfo_exists():
            self.detail_window.lift()
            return
        self.detail_window = tk.Toplevel(self)
        self.detail_window.title('本轮标定 · 失败原因')
        self.detail_text = ScrolledText(self.detail_window, wrap='word', width=80, height=18)
        self.detail_text.pack(fill='both', expand=True)
        self.detail_text.insert('end', '\n\n'.join(self.issues) + '\n\n')
        self.detail_text.configure(state='disabled')

    def mount(self, side, delta, rotation):
        name = '前 · 锁定' if side == 'front' else RADAR_NAMES[side]
        length = sum(x * x for x in delta) ** .5
        self.mounts.insert('', 'end', values=(name, *(number(x) for x in delta), number(length), number(rotation)))


class Results(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        bar = ttk.Frame(self)
        bar.pack(fill='x')
        ttk.Label(bar, text='运行结果').pack(side='left', padx=4)
        self.selector = ttk.Combobox(bar, state='readonly', width=42)
        self.selector.pack(side='left', padx=4)
        self.selector.bind('<<ComboboxSelected>>', lambda _: self.show(self.selector.current()))
        ttk.Label(bar, text='展示方式').pack(side='left', padx=(12, 3))
        self.mode = ttk.Combobox(bar, state='readonly', values=('数据表', '柱状图'), width=7)
        self.preferred_mode = '数据表'
        self.mode.current(0)
        self.mode.pack(side='left')
        self.mode.bind('<<ComboboxSelected>>', self.refresh_view)
        self.slot_controls = ttk.Frame(bar)
        ttk.Label(self.slot_controls, text='摆位').pack(side='left', padx=(12, 3))
        self.slot = ttk.Combobox(self.slot_controls, state='readonly', values=SLOTS, width=13)
        self.slot.current(0)
        self.slot.pack(side='left')
        self.slot.bind('<<ComboboxSelected>>', self.refresh_view)
        self.issues_button = ttk.Button(bar, text='失败原因（0）', state='disabled',
                                       command=lambda: self.displayed.show_issues())
        self.issues_button.pack(side='right', padx=4)
        self.runs = []
        self.labels = []
        self.current = self.displayed = self.make_run()
        self.displayed.pack(fill='both', expand=True)

    def update_issues(self):
        count = len(self.displayed.issues)
        self.issues_button.configure(text=f'失败原因（{count}）', state='normal' if count else 'disabled')

    def make_run(self):
        run = RunTables(self)
        run.bind('<<NotebookTabChanged>>', self.refresh_view)
        return run

    def refresh_view(self, event=None):
        if isinstance(getattr(event, 'widget', None), RunTables) and event.widget is not self.displayed:
            return
        page = self.displayed.index('current')
        if getattr(event, 'widget', None) is self.mode:
            self.preferred_mode = self.mode.get()
        self.mode.set('数据表' if page == 2 else self.preferred_mode)
        chart_mode = self.mode.get() == '柱状图'
        if page == 2:
            self.mode.state(['disabled'])
        else:
            self.mode.state(['!disabled'])
        if chart_mode:
            self.slot_controls.pack(side='left')
        else:
            self.slot_controls.pack_forget()
        self.displayed.render(page, chart_mode, self.slot.get())

    def new(self, machine):
        if self.runs:
            self.current = self.make_run()
        self.runs.append(self.current)
        self.labels.append(f'{len(self.runs)}  ·  {machine}  ·  {datetime.now():%H:%M:%S}')
        self.selector.configure(values=self.labels)
        self.selector.current(len(self.runs) - 1)
        self.show(len(self.runs) - 1)

    def show(self, index):
        self.displayed.pack_forget()
        self.displayed = self.runs[index]
        self.update_issues()
        self.displayed.pack(fill='both', expand=True)
        self.refresh_view()
