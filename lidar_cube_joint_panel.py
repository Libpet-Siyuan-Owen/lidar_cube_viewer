"""三标定包面板。Tk 控件只在主线程更新。"""
from pathlib import Path
from queue import Queue, Empty
import shutil
import subprocess
from threading import Event, Thread
from tkinter import StringVar, ttk
import numpy as np
import yaml

from lidar_cube_joint import CONFIG, JOINT_CONFIG, LABEL, load_joint_config, report_directories, read_json, run_joint
from lidar_cube_core import WORKSPACE


class JointPanel:
    title = '三标定包联合拟合'
    case_count = 3
    mount_count = 3
    joint_label = '三包联合外参 N'
    def __init__(self, viewer, parent, load_settings):
        self.viewer = viewer
        self.load_settings = load_settings
        self.events = Queue()
        self.stop = Event()
        self.running = False
        self.output = None
        self.inputs = []
        self.snapshots = [(None, None)] * 3
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=4)
        self.toggle = ttk.Button(section, text='▸ ' + self.title, command=self.expand)
        self.toggle.pack(fill='x')
        self.body = ttk.Frame(section, padding=4)
        load_button = ttk.Button(self.body, text='加载联合拟合配置…', command=self.pick_config)
        load_button.pack(fill='x')
        self.inputs.append(load_button)
        self.folders, self.choices, self.notes, self.selectors = [], [], [], []
        for i in range(3):
            folder, choice, note = (StringVar(master=section) for _ in range(3))
            self.folders.append(folder)
            self.choices.append(choice)
            self.notes.append(note)
            row = ttk.Frame(self.body)
            row.pack(fill='x', pady=(4, 0))
            button = ttk.Button(row, text=f'情况{i+1}…', width=8, command=lambda i=i: self.pick(i))
            button.pack(side='left')
            ttk.Entry(row, width=12, textvariable=folder, state='readonly').pack(side='left', fill='x', expand=True)
            combo = ttk.Combobox(self.body, width=16, state='readonly', textvariable=choice)
            combo.pack(fill='x')
            combo.bind('<<ComboboxSelected>>', lambda _, i=i: self.change(i))
            self.inputs.append(button)
            self.selectors.append(combo)
            ttk.Label(self.body, textvariable=note, wraplength=330).pack(anchor='w')
        self.output_parent = ''
        self.output_display = StringVar(master=section)
        destination = ttk.Frame(self.body)
        destination.pack(fill='x', pady=(6, 0))
        ttk.Label(destination, text='结果保存位置').pack(side='left')
        choose = ttk.Button(destination, text='选择目录…', command=self.choose_output_parent)
        choose.pack(side='right')
        reset = ttk.Button(destination, text='跟随情况1', command=lambda: self.update_output_parent(''))
        reset.pack(side='right', padx=3)
        self.inputs.extend((choose, reset))
        ttk.Entry(self.body, textvariable=self.output_display, state='readonly').pack(fill='x')
        self.set_output_parent('')
        buttons = ttk.Frame(self.body)
        buttons.pack(fill='x', pady=4)
        self.start_button = ttk.Button(buttons, text='开始联合拟合', command=self.start)
        self.start_button.pack(side='left')
        self.open_button = ttk.Button(buttons, text='打开结果目录', command=self.open_output, state='disabled')
        self.open_button.pack(side='left', padx=4)
        self.message = StringVar(master=section, value='请选择三个不同的标定包。')
        ttk.Label(self.body, textvariable=self.message, wraplength=340, justify='left').pack(anchor='w')
        cols = ('固定使用的外参', '情况1', '情况2', '情况3', '最差情况 ↓')
        self.validation = self.table('各情况的平均角点距离（cm）', cols, 4)
        for col, width in zip(cols, (113, 49, 49, 49, 82)):
            self.validation.column(col, width=width, minwidth=width, anchor='w' if col == cols[0] else 'center')
        self.validation.tag_configure('joint', foreground='#174c83')
        self.mounts = self.table('导出外参 N 相对 /tf_static 的安装修正',
                                 ('雷达','ΔX cm','ΔY cm','ΔZ cm','位移 cm','转角 °'), 3)

    def table(self, title, cols, height):
        ttk.Label(self.body, text=title).pack(anchor='w', pady=(4, 0))
        table = ttk.Treeview(self.body, columns=cols, show='headings', height=height, selectmode='none')
        for c in cols:
            table.heading(c, text=c)
            table.column(c, width=42, minwidth=30, stretch=True, anchor='center')
        table.pack(fill='x')
        return table

    def expand(self):
        if self.body.winfo_manager():
            self.body.pack_forget()
            self.toggle.configure(text='▸ ' + self.title)
        else:
            self.body.pack(fill='x')
            self.toggle.configure(text='▾ ' + self.title)

    def clear_result(self):
        for table in (self.validation, self.mounts):
            table.delete(*table.get_children())
        self.output = None
        self.open_button.state(['disabled'])

    def set_output_parent(self, path):
        self.output_parent = str(path) if path else ''
        default = self.folders[0].get() or '请选择第一个标定包'
        self.output_display.set(self.output_parent or f'跟随情况1 · {default}')

    def update_output_parent(self, path):
        if path != self.output_parent:
            self.set_output_parent(path)
            self.viewer.autosave_current_settings()

    def choose_output_parent(self):
        initial = Path(self.output_parent or self.folders[0].get() or WORKSPACE/'data')
        path = self.viewer.dialog.askdirectory(parent=self.viewer.root, title='选择联合拟合结果保存目录',
                                               initialdir=initial if initial.is_dir() else WORKSPACE/'data',
                                               mustexist=True)
        if path:
            self.update_output_parent(path)

    def pick_config(self):
        path = self.viewer.dialog.askopenfilename(parent=self.viewer.root, title='加载联合拟合配置',
                     initialdir=self.output or self.output_parent or self.folders[0].get() or WORKSPACE/'data',
                     filetypes=[('联合拟合配置', JOINT_CONFIG), ('YAML', '*.yaml')])
        if not path:
            return
        try:
            self.restore(Path(path))
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, yaml.YAMLError) as exc:
            self.message.set(f'加载联合配置失败：{exc}')

    def restore(self, path):
        config, result = load_joint_config(path)
        rows = self.table_rows(result)
        output = path.resolve().parent
        for i, case in enumerate(config['cases']):
            name = Path(case['report']).name
            self.folders[i].set(case['bag'])
            self.choices[i].set(name)
            self.selectors[i].configure(values=[name])
            self.snapshots[i] = (output / case['config_snapshot'],
                                 output / case['single_snapshot'] if 'single_snapshot' in case else None)
            self.notes[i].set('已恢复本次使用的配置快照')
        self.set_output_parent(output.parent)
        self.clear_result()
        self.output = output
        self.open_button.state(['!disabled'])
        self.render(result, rows)
        self.viewer.status.set(f"已加载联合拟合结果（{result['status']}）：{output}")

    def pick(self, i):
        path = self.viewer.dialog.askdirectory(parent=self.viewer.root, title='Select calibration bag directory',
                                               initialdir=Path(self.folders[i].get()).parent if self.folders[i].get() else WORKSPACE/'data')
        if path:
            self.set_folder(i, Path(path))

    def set_folder(self, i, path):
        names = [p.name for p in report_directories(path)]
        self.folders[i].set(str(path))
        if i == 0:
            self.set_output_parent(self.output_parent)
        self.selectors[i].configure(values=names)
        self.choices[i].set(names[0] if names else '')
        self.change(i)

    def change(self, i):
        self.snapshots[i] = (None, None)
        self.clear_result()
        self.message.set('选择三个包及对应的单包报告后开始拟合。')
        try:
            folder = Path(self.folders[i].get())
            if not self.choices[i].get():
                raise ValueError('未找到单包报告，请先保存平面拟合结果')
            meta = read_json(folder/self.choices[i].get()/'validation.json')
            cfg = yaml.safe_load((folder/CONFIG).read_text(encoding='utf-8'))
            report_roi, config_roi = meta['bounds_xyz'], cfg['bounds']
            note = '' if np.allclose(report_roi, config_roi, atol=1e-9, rtol=0) else f'ROI 不同：拟合 {report_roi}；回代 {config_roi}'
        except Exception as exc:
            note = str(exc)
        self.notes[i].set(note)

    def busy(self, value):
        self.running = value
        for button in self.inputs + [self.start_button]:
            button.state(['disabled' if value else '!disabled'])
        for combo in self.selectors:
            combo.configure(state='disabled' if value else 'readonly')

    def start(self):
        if self.running:
            return
        self.clear_result()
        if not all(var.get() for var in self.folders+self.choices):
            self.message.set('请为三个不同的数据包各选择一份单包报告。')
            return
        selections = [(Path(b.get()), Path(b.get())/r.get()) for b,r in zip(self.folders,self.choices)]
        output_parent = self.output_parent or None
        snapshots = list(self.snapshots)
        self.busy(True)
        self.message.set('正在检查三个标定包……')
        self.stop.clear()
        def worker():
            try:
                result = run_joint(selections, self.load_settings,
                                   lambda m: self.events.put(('progress', m)), self.stop.is_set,
                                   output_parent=output_parent, compare_singles=True, snapshots=snapshots)
                self.events.put(('done', result))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        Thread(target=worker, daemon=True).start()
        self.viewer._schedule_after(80, self.poll)

    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'progress':
                    self.message.set(data)
                    self.viewer.status.set(data)
                elif kind == 'error':
                    self.busy(False)
                    self.message.set('联合拟合失败：' + data)
                    self.viewer.status.set(self.message.get())
                else:
                    self.busy(False)
                    self.output, result = data
                    self.open_button.state(['!disabled'])
                    self.render(result)
        except Empty:
            if self.running:
                self.viewer._schedule_after(100, self.poll)

    def table_rows(self, result):
        validation, mounts = [], []
        if not isinstance(result['status'], str) or not result['corner_comparison'] or len(result['mounts']) != self.mount_count:
            raise ValueError('结果报告缺少状态或指标表。')
        for row in result['corner_comparison']:
            if len(row['cases_cm']) != self.case_count:
                raise ValueError('结果表缺少情况数据。')
            cells = ['—' if value is None else f'{value:.2f}' for value in (*row['cases_cm'], row['worst_cm'])]
            tags = ('joint',) if row['label'] == self.joint_label else ()
            validation.append(((row['label'], *cells), tags))
        for side, m in result['mounts'].items():
            values = [*m['delta_xyz_cm'],m['displacement_cm'],m['rotation_deg']]
            mounts.append([('前·锁定' if side=='front' else LABEL[side]),
                                                  *[f'{v:.2f}' for v in values]])
        return validation, mounts

    def render(self, result, rows=None):
        validation, mounts = rows or self.table_rows(result)
        for values, tags in validation:
            self.validation.insert('', 'end', values=values, tags=tags)
        for values in mounts:
            self.mounts.insert('', 'end', values=values)
        self.message.set('' if result['status'] == '验证完成' else '部分结果缺失，详见 metrics.log。')
        self.viewer.status.set(f"{result['status']}，结果已保存到 {self.output}")

    def open_output(self):
        if not self.output:
            return
        try:
            if shutil.which('explorer.exe') and shutil.which('wslpath'):
                target = subprocess.check_output(['wslpath','-w',str(self.output)], text=True).strip()
                subprocess.Popen(['explorer.exe',target])
            else:
                subprocess.Popen(['xdg-open',str(self.output)])
        except (OSError, subprocess.SubprocessError) as exc:
            self.message.set(f'请打开 {self.output}（{exc}）')
