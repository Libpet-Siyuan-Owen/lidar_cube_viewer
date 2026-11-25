"""Interactive single-placement cloud inspection in the batch window."""
from pathlib import Path
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lidar_cube_core import COLORS
from v02_batch import RADAR_NAMES, SLOTS
from v02_results import Results


class ScenePanel(ttk.Frame):
    standard = ('本次联合外参', '/tf_static')
    browse = '选择 YAML 文件…'

    def __init__(self, parent, events):
        super().__init__(parent)
        self.events = events
        self.cancel = Event()
        self.report = self.scene = self.slot = self.results = None
        self.custom = None
        self.source = 'joint'
        bar = ttk.Frame(self, padding=4)
        bar.pack(fill='x')
        ttk.Label(bar, text='标定包').pack(side='left')
        self.bags = ttk.Combobox(bar, state='readonly', width=48)
        self.bags.pack(side='left', padx=4)
        self.bags.bind('<<ComboboxSelected>>', lambda _: self.load())
        ttk.Label(bar, text='外参').pack(side='left', padx=(12, 0))
        self.sources = ttk.Combobox(bar, state='readonly', width=25)
        self.sources.pack(side='left', padx=4)
        self.sources.bind('<<ComboboxSelected>>', self.choose_source)
        self.planes = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text='显示拟合平面', variable=self.planes, command=self.draw).pack(side='left')
        figure = Figure(figsize=(9, 3.7), dpi=90)
        self.ax = figure.add_subplot(111, projection='3d')
        figure.subplots_adjust(left=.03, right=.97, top=.98, bottom=.10)
        self.canvas = FigureCanvasTkAgg(figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.mpl_connect('scroll_event', self.zoom)
        self.note = tk.StringVar()
        ttk.Label(self, textvariable=self.note, wraplength=1050).pack(anchor='w', padx=8)
        self.caption = ttk.Label(self)
        self.caption.pack(anchor='w', padx=8)

    def open(self, report):
        self.report = report
        self.slot = self.custom = None
        self.source = 'joint'
        self.planes.set(True)
        self.bags.configure(values=[f'{s}  ·  {name}' for s,(name,_,_) in report.cases.items()])
        self.bags.current(0)
        self.sources.configure(values=(*self.standard, self.browse))
        self.sources.current(0)
        if self.results:
            self.results.destroy()
        self.results = Results(self)
        self.results.pack(fill='x', padx=6)
        self.results.new(report.path.name)
        self.results.selector.configure(state='disabled')
        self.caption.configure(text=f'下方指标取自原标定报告　{report.path.name}')
        for stage, field in [('初始','initial'), ('复验','validation')]:
            for slot, row in zip(SLOTS, report.data[field]):
                frames = row.get('frames') or {s: dict(used_frames='—',roi_points='—') for s in row['sensors']}
                self.results.current.bag(slot, stage, row['sensors'], row['pairs'], frames)
        for side, (delta, angle) in report.data.get('mount_changes', {}).items():
            self.results.current.mount(side, delta, angle)
        if 'mount_changes' not in report.data:
            self.results.current.mounts.insert('', 'end', values=('旧报告未记录', *(['—']*5)))
        self.load()

    def choose_source(self, _=None):
        choice = self.sources.get()
        if choice == self.browse:
            path = filedialog.askopenfilename(parent=self, initialdir=self.report.path,
                    filetypes=[('外参 YAML', '*.yaml *.yml'), ('全部文件','*')])
            if not path:
                self.sources.current(0 if self.source=='joint' else 1 if self.source=='initial' else 2)
                return
            self.custom = Path(path)
            self.sources.configure(values=(*self.standard, self.custom.name, self.browse))
            self.sources.current(2)
        self.source = ('joint','initial',self.custom)[self.sources.current()]
        self.load()

    def load(self):
        self.cancel.set()
        token = self.cancel = Event()
        slot = SLOTS[self.bags.current()]
        if self.slot != slot:
            self.slot = slot
            self.ax.clear()
            self.ax.view_init(elev=22, azim=-58)
            bounds = self.report.cases[slot][1]
            mid, radius = bounds.mean(axis=1), np.ptp(bounds, axis=1).max()*.52
            for axis, c in zip('xyz', mid):
                getattr(self.ax, 'set_'+axis+'lim')(c-radius,c+radius)
            self.ax.set_box_aspect((1,1,1))
            self.ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        self.scene = None
        self.draw()
        self.note.set('正在读取点云与拟合面；缺少缓存时将从原包重新累计…')
        report, source = self.report, self.source
        def worker():
            try:
                scene = report.load(slot, source, token.is_set)
                value = (token, scene, '')
            except Exception as exc:
                value = (token, None, str(exc))
            if not token.is_set():
                self.events.put(('scene', value))
        Thread(target=worker, daemon=True).start()

    def receive(self, token, scene, error):
        if token is not self.cancel or token.is_set():
            return
        self.scene = scene
        if error:
            self.note.set(f'{self.slot}　{error}')
        else:
            self.note.set('　　'.join(f'{RADAR_NAMES[s]}雷达 {len(p):,} 点 · {len(scene.metadata["faces"][s])}/3 面'
                                      for s,p in scene.points.items()))
        self.draw()

    def draw(self):
        for artist in list(self.ax.collections):
            artist.remove()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        if self.scene:
            handles = []
            for side, points in self.scene.points.items():
                color = COLORS[side]
                handles.append(self.ax.scatter(*points.T, s=1, c=color, alpha=.65, depthshade=False, label=side))
                if self.planes.get():
                    faces = [f['vertices'] for f in self.scene.metadata['faces'][side] if f['vertices']]
                    if faces:
                        self.ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=color,
                                                                  linewidths=1.3, alpha=.24))
            self.ax.legend(handles=handles, loc='upper left', markerscale=4)
        self.canvas.draw_idle()

    def zoom(self, event):
        if event.inaxes is not self.ax:
            return
        for axis in 'xyz':
            a,b = getattr(self.ax,'get_'+axis+'lim')()
            mid = (a+b)/2
            radius = (b-a)/2 * (.85 if event.step>0 else 1/.85)
            getattr(self.ax,'set_'+axis+'lim')(mid-radius,mid+radius)
        self.canvas.draw_idle()
