"""六包四雷达面板，共用三包面板的结果表、任务队列和目录操作。"""
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from tkinter import StringVar, ttk

from lidar_cube_core import WORKSPACE
from lidar_cube_joint import JOINT_CONFIG
from lidar_cube_joint_panel import JointPanel
from lidar_cube_six import GROUPS, LABELS, SIX_CONFIG, load_six, read_group, run_six


class SixPanel(JointPanel):
    title = '六包四雷达联合解'
    case_count = 6
    mount_count = 4
    joint_label = LABELS[-1]

    def __init__(self, viewer, parent, load_settings):
        self.viewer, self.load_settings = viewer, load_settings
        self.events, self.stop = Queue(), Event()
        self.running, self.output = False, None
        self.output_parent = ''
        self.inputs, self.selectors = [], []
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=4)
        self.toggle = ttk.Button(section, text='▸ ' + self.title, command=self.expand)
        self.toggle.pack(fill='x')
        self.body = ttk.Frame(section, padding=4)
        self.paths = [StringVar(section), StringVar(section)]
        self.output_display = StringVar(section, value='默认保存到机器数据目录')
        self.message = StringVar(section)
        self.button(self.body, '加载六包结果配置…', self.pick_config).pack(fill='x')
        for i, name in enumerate(('FLT', 'FRT')):
            row = ttk.Frame(self.body)
            row.pack(fill='x', pady=2)
            self.button(row, f'选择 {name} 配置…', lambda i=i: self.pick_group(i)).pack(side='left')
            ttk.Entry(row, width=15, textvariable=self.paths[i], state='readonly').pack(side='left', expand=True, fill='x')
        row = ttk.Frame(self.body)
        row.pack(fill='x')
        self.button(row, '结果保存位置…', self.choose_output_parent).pack(side='left')
        ttk.Entry(row, width=15, textvariable=self.output_display, state='readonly').pack(side='left', expand=True, fill='x')
        row = ttk.Frame(self.body)
        row.pack(fill='x', pady=4)
        self.start_button = ttk.Button(row, text='开始六包联合拟合', command=self.start)
        self.start_button.pack(side='left')
        self.open_button = ttk.Button(row, text='打开结果目录', state='disabled', command=self.open_output)
        self.open_button.pack(side='left', padx=4)
        ttk.Label(self.body, textvariable=self.message, wraplength=340).pack(anchor='w')
        cols = ('固定使用的外参', '左1','左2','左3','右1','右2','右3','最差 ↓')
        self.validation = self.table('各情况的平均角点距离（cm）', cols, 4)
        for col, width in zip(cols, (105,33,33,33,33,33,33,52)):
            self.validation.column(col, width=width, minwidth=width, anchor='w' if col==cols[0] else 'center')
        self.validation.tag_configure('joint', foreground='#174c83')
        self.mounts = self.table('六包联合外参 N 相对 /tf_static 的安装修正',
                                 ('雷达','ΔX cm','ΔY cm','ΔZ cm','位移 cm','转角 °'), 4)

    def button(self, parent, text, command):
        widget = ttk.Button(parent, text=text, command=command)
        self.inputs.append(widget)
        return widget

    def choose_output_parent(self):
        folder = self.viewer.dialog.askdirectory(parent=self.viewer.root, title='选择六包结果保存位置',
                         initialdir=self.output_parent or WORKSPACE/'data', mustexist=True)
        if folder:
            self.output_parent = folder
            self.output_display.set(folder)

    def select_config(self, filename, title):
        initial = self.output or self.output_parent
        if not initial:
            selected = next((v.get() for v in self.paths if v.get()), '')
            initial = Path(selected).parent.parent if selected else WORKSPACE/'data'
        return self.viewer.dialog.askopenfilename(parent=self.viewer.root, title=title,
                          initialdir=initial, filetypes=[('拟合配置', filename), ('YAML', '*.yaml')])

    def pick_group(self, index):
        path = self.select_config(JOINT_CONFIG, '选择对应的三包联合配置')
        if path:
            try:
                cases, _ = read_group(path, GROUPS[index])
                self.paths[index].set(path)
                self.clear_result()
                self.message.set('')
                if not self.output_parent:
                    self.output_display.set(str(Path(cases[0]['bag']).parent))
            except Exception as exc:
                self.message.set(f'无法加载三包配置：{exc}')

    def pick_config(self):
        path = self.select_config(SIX_CONFIG, '加载六包结果配置')
        if path:
            try:
                config, result = load_six(path)
                rows = self.table_rows(result)
                self.clear_result()
                self.output = Path(path).resolve().parent
                self.output_parent = str(self.output.parent)
                self.output_display.set(self.output_parent)
                for var, source in zip(self.paths, config['group_configs']):
                    var.set(source)
                self.open_button.state(['!disabled'])
                self.render(result, rows)
            except Exception as exc:
                self.message.set(f'加载六包结果失败：{exc}')

    def start(self):
        if self.running:
            return
        paths = [v.get() for v in self.paths]
        if not all(paths):
            self.message.set('请分别选择 FLT 和 FRT 的三包联合配置。')
            return
        parent = self.output_parent
        self.clear_result()
        self.busy(True)
        self.stop.clear()
        self.message.set('正在检查六个标定包……')

        def worker():
            try:
                data = run_six(paths, self.load_settings, lambda m: self.events.put(('progress', m)),
                               self.stop.is_set, parent)
                self.events.put(('done', data))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        Thread(target=worker, daemon=True).start()
        self.viewer._schedule_after(80, self.poll)
