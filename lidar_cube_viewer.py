#!/usr/bin/env python3
"""静态多雷达标定查看器。读取 bag 并累计 ROI，拟合三面体、验证和保存外参。

本模块管理 Tk 界面、配置和后台任务。点云累计、平面计算和外参求解见同目录 README。
"""
import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
from threading import Event, Thread
from datetime import datetime

import numpy as np
import yaml
from lidar_cube_correction import attach_extrinsic_report
from lidar_cube_core import (
    AccumulationRequest, AccumulationResult, COLORS, DATA_DIR,
    ALL_SIDES, KIT, MAPPING_SIDES, accumulate,
    LEGACY_CALIBRATION_TARGETS, normalize_pose_rows, thin,
)

# Independent map controls include the top radar.  The core's
# ``SIDES`` remains the legacy front/left/right calibration set for scripts
# that still report the original three-radar metrics.
SIDES = MAPPING_SIDES
from lidar_cube_processing import (
    FilterConfig, compute_extrinsic_corrections, evaluate_metrics, extract_planes,
    plane_spans, filter_points,
)

DEFAULT_CONFIG = Path.home() / '.config/lidar_cube_viewer/settings.yaml'
LAST_CONFIG_POINTER = DEFAULT_CONFIG.with_name('last_config_path.txt')
# Keep top-level labels in ASCII so they render correctly on X11 setups where
# the native window title or Matplotlib canvas does not have a CJK font.
SOURCE_LABELS = {'bag': 'Bag initial /tf_static', 'yaml': 'Calibrated extrinsics YAML'}
@dataclass
class ViewerSettings:
    request: AccumulationRequest
    front_map: str = ''
    map_bundle_dir: str = ''
    map_source: str = 'build'
    mapping_sensors: list = field(default_factory=lambda: ['front'])
    static_mapping: bool = True
    display_voxel: float = .01
    analysis_voxel: float = .01
    filter_radius: float = .08
    filter_min_neighbors: int = 3
    plane_threshold: float = .015
    plane_min_inliers: int = 100
    plane_max_planes: int = 3
    plane_max_iterations: int = 500
    plane_min_angle: float = 30.0
    plane_max_span: float = .70
    plane_min_span: float = 0.0
    visible: dict = field(default_factory=lambda: dict.fromkeys(ALL_SIDES, True))
    # Legacy aliases are kept for old configuration files.  The UI now stores
    # visibility per independent radar map/trajectory.
    show_map: bool = True
    show_trajectory: bool = True
    show_independent_maps: dict = field(
        default_factory=lambda: {'front': True, 'left': False, 'right': False, 'top': False})
    show_independent_trajectories: dict = field(
        default_factory=lambda: {'front': True, 'left': False, 'right': False, 'top': False})

    # 空值表示联合结果跟随面板的第一个标定包。
    joint_output_parent: str = ''

    def validate(self):
        self.request.validate()
        if (not self.mapping_sensors or any(side not in SIDES for side in self.mapping_sensors)
                or 'front' not in self.mapping_sensors):
            raise ValueError('独立建图必须包含前雷达；左、右、顶部雷达可以按需追加。')
        if not np.isfinite(self.display_voxel) or self.display_voxel < 0:
            raise ValueError('Display voxel must be nonnegative; 0 displays all ROI points.')
        if set(self.visible) != set(ALL_SIDES) or any(type(v) is not bool for v in self.visible.values()):
            raise ValueError('Visible layers need front/left/right/top boolean values.')
        for name, values in (('independent maps', self.show_independent_maps),
                             ('independent trajectories', self.show_independent_trajectories)):
            if set(values) != set(SIDES) or any(type(v) is not bool for v in values.values()):
                raise ValueError(f'{name} need front/left/right boolean values.')
        if (self.analysis_voxel <= 0 or self.filter_radius <= 0
                or self.filter_min_neighbors <= 0 or self.plane_threshold <= 0
                or self.plane_min_inliers <= 0 or self.plane_max_planes <= 0
                or self.plane_max_iterations <= 0
                or not 0 < self.plane_min_angle < 90 or self.plane_max_span <= 0
                or self.plane_min_span < 0):
            raise ValueError('Processing parameters must be positive.')

    def as_dict(self):
        """配置保存输入与显示设置，累计点云独立保留在结果对象中。"""
        self.validate()
        return dict(settings_version=15, bag=self.request.bag, trajectory='',
                    front_map=self.front_map,
                    map_bundle_dir=self.map_bundle_dir,
                    joint_output_parent=self.joint_output_parent,
                    map_source=self.map_source,
                    mapping_sensors=list(self.mapping_sensors),
                    static_mapping=True,
                    show_map=self.show_map, show_trajectory=self.show_trajectory,
                    show_independent_maps=dict(self.show_independent_maps),
                    show_independent_trajectories=dict(self.show_independent_trajectories),
                    extrinsics_source=self.request.extrinsics_source,
                    extrinsics=self.request.extrinsics, bounds=self.request.bounds.tolist(),
                    calibration_sides=list(self.request.active_sides()),
                    display_voxel=self.display_voxel, visible=self.visible,
                    analysis_voxel=self.analysis_voxel, filter_radius=self.filter_radius,
                    filter_min_neighbors=self.filter_min_neighbors,
                    plane_threshold=self.plane_threshold,
                    plane_min_inliers=self.plane_min_inliers,
                    plane_max_planes=self.plane_max_planes,
                    plane_max_iterations=self.plane_max_iterations,
                    plane_min_angle=self.plane_min_angle,
                    plane_max_span=self.plane_max_span,
                    plane_min_span=self.plane_min_span,
                    max_frames=self.request.max_frames)


def default_settings():
    return ViewerSettings(AccumulationRequest(
        '', '',
        np.array([[1.8, 3.5], [4.9, 6.5], [.08, .70]]),
        calibration_sides=('front', 'left'),
        static_mapping=True),
        front_map='')


def load_settings(path):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('settings_version') not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        raise ValueError('请选择有效的 viewer 配置文件。')

    def input_path(key):
        value = data.get(key, '')
        if not isinstance(value, str):
            raise ValueError(f'{key} must be a path string.')
        return str((path.parent / Path(value).expanduser()).resolve()) if value else ''

    legacy_show_map = bool(data.get('show_map', True))
    legacy_show_trajectory = bool(data.get('show_trajectory', True))
    independent_maps = data.get('show_independent_maps', {
        'front': legacy_show_map, 'left': False, 'right': False, 'top': False})
    independent_trajectories = data.get('show_independent_trajectories', {
        'front': legacy_show_trajectory, 'left': False, 'right': False, 'top': False})
    independent_maps = dict(independent_maps)
    independent_trajectories = dict(independent_trajectories)
    independent_maps.setdefault('top', False)
    independent_trajectories.setdefault('top', False)
    visible = dict(data.get('visible', dict.fromkeys(ALL_SIDES, True)))
    visible.setdefault('top', True)
    legacy_target = data.get('calibration_target', 'front-left-right')
    calibration_sides = data.get('calibration_sides')
    if calibration_sides is None:
        calibration_sides = LEGACY_CALIBRATION_TARGETS.get(
            legacy_target, LEGACY_CALIBRATION_TARGETS['front-left-right'])
    request = AccumulationRequest(
        input_path('bag'), input_path('trajectory'),
        np.asarray(data['bounds'], dtype=float),
        data.get('extrinsics_source', 'bag'), input_path('extrinsics'),
        legacy_target, tuple(calibration_sides), int(data.get('max_frames', 0)))
    settings = ViewerSettings(request,
                              front_map=input_path('front_map'),
                              map_bundle_dir=input_path('map_bundle_dir'),
                              joint_output_parent=input_path('joint_output_parent'),
                              map_source=data.get(
                                  'map_source',
                                  'existing' if (data.get('map_bundle_dir') or
                                                 (data.get('front_map') and data.get('trajectory'))) else 'build'),
                              mapping_sensors=list(data.get('mapping_sensors', ['front'])),
                              # Dynamic NDT mapping is retired.  Treat old
                              # configurations as static captures as well.
                              static_mapping=True,
                              display_voxel=float(data.get('display_voxel', .01)),
                              analysis_voxel=float(data.get('analysis_voxel', data.get('filter_voxel', .01))),
                              filter_radius=float(data.get('filter_radius', .08)),
                              filter_min_neighbors=int(data.get('filter_min_neighbors', 3)),
                              plane_threshold=float(data.get('plane_threshold', .015)),
                              plane_min_inliers=int(data.get('plane_min_inliers', 100)),
                              plane_max_planes=int(data.get('plane_max_planes', 3)),
                              plane_max_iterations=int(data.get('plane_max_iterations', 500)),
                              plane_min_angle=float(data.get('plane_min_angle', 30.0)),
                              plane_max_span=float(data.get('plane_max_span', .70)),
                              plane_min_span=float(data.get('plane_min_span', 0.0)),
                              visible=visible,
                              show_map=legacy_show_map,
                              show_trajectory=legacy_show_trajectory,
                              show_independent_maps=independent_maps,
                              show_independent_trajectories=independent_trajectories,
                              )
    settings.validate()
    return settings


def save_settings(path, settings):
    """同目录原子替换，写入失败时保留原来的配置。"""
    data = settings.as_dict()
    for name in ('bag', 'trajectory', 'extrinsics', 'front_map', 'map_bundle_dir'):
        if data[name]:
            data[name] = str(Path(data[name]).expanduser().resolve())
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Viewer:
    def __init__(self, root, settings=None, config_path=DEFAULT_CONFIG):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from tkinter import font
        import matplotlib
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.tk, self.ttk, self.dialog, self.messagebox = tk, ttk, filedialog, messagebox
        self.root, self.config_path = root, Path(config_path)
        # All generated artifacts are kept beside the selected bag/map bundle.
        # This is intentionally separate from ``config_path`` so loading the
        # legacy global settings file does not make the workspace/data folder
        # the output directory for a newly selected dataset.
        self.data_root = None
        # WSL often has no CJK font by default. Configure Tk and message boxes
        # together so Chinese labels are rendered instead of replacement glyphs.
        families = set(font.families(root))
        if 'WenQuanYi Zen Hei' in families:
            chinese_font = 'WenQuanYi Zen Hei'
        elif 'Noto Sans CJK SC' in families:
            chinese_font = 'Noto Sans CJK SC'
        else:
            chinese_font = 'TkDefaultFont'
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont',
                     'TkCaptionFont', 'TkSmallCaptionFont', 'TkTooltipFont'):
            try:
                font.nametofont(name).configure(family=chinese_font)
            except tk.TclError:
                pass
        tk_font_spec = f'{{{chinese_font}}} 10'
        root.option_add('*Font', tk_font_spec)
        root.option_add('*TkFDialog*Font', tk_font_spec)
        root.option_add('*Dialog*Font', tk_font_spec)
        matplotlib.rcParams['font.sans-serif'] = [chinese_font, 'Noto Sans CJK SC', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        self.result: AccumulationResult | None = None
        self.processed_clouds = None
        self.filter_display = False
        self.plane_results = None
        self.metrics_results = None
        self.report_directory = None
        self.plane_markers = []
        self.plane_inlier_markers = []
        self.Poly3DCollection = Poly3DCollection
        self.processing = False
        self.front_map_points = np.empty((0, 3), dtype=np.float32)
        self.trajectory_points = np.empty((0, 3), dtype=np.float32)
        self.trajectory_rows = []
        self.front_map_path = ''
        self.independent_map_points = {
            side: np.empty((0, 3), dtype=np.float32) for side in SIDES
        }
        self.independent_trajectory_points = {
            side: np.empty((0, 3), dtype=np.float32) for side in SIDES
        }
        self.independent_trajectory_rows = {side: [] for side in SIDES}
        self.independent_map_paths = {}
        self.show_independent_maps = {
            side: tk.BooleanVar(value=(side == 'front')) for side in SIDES
        }
        self.show_independent_trajectories = {
            side: tk.BooleanVar(value=(side == 'front')) for side in SIDES
        }
        self.independent_map_markers = []
        self.independent_trajectory_markers = []
        self.map_markers = []
        self.trajectory_marker = None
        self.map_ready = False
        self.roi_confirmed = False
        self.build_sides = []
        self.build_output = None
        self.events, self.filter_events, self.plane_events, self.cancelled = (
            Queue(), Queue(), Queue(), Event())
        self.pending, self.autosave = False, True
        # Tk 的 after 回调可能在文件选择对话框关闭、或主窗口关闭后仍被
        # 调度。统一登记并取消这些回调，避免回调访问已销毁的控件。
        self._closing = False
        self._after_ids = set()
        self.view_key, self.markers = None, []
        self.roi_markers = []
        self.frame_step = 1
        self.frame_index = 0
        self.frame_mode = tk.StringVar(value='single')
        self.build_process = None
        warning = ''
        if settings is None:
            try:
                settings = load_settings(self.config_path) if self.config_path.exists() else default_settings()
            except Exception as error:
                settings = default_settings()
                warning = f'Settings could not be restored: {error}'
                self.autosave = False  # 损坏文件留给用户处理，不自动覆盖。
        root.title('LiDAR Cube Validation Viewer')
        root.geometry('1500x950')
        root.minsize(1180, 820)
        style = ttk.Style(root)
        try:
            style.configure('Action.TButton', padding=(8, 5))
            style.configure('Primary.TButton', padding=(8, 7), font=('TkDefaultFont', 10, 'bold'))
            style.configure('Status.TLabel', padding=(4, 3))
        except tk.TclError:
            pass
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill='both', expand=True)
        # Use a real paned window so the sidebar can be widened when paths are long.
        self.paned = ttk.Panedwindow(outer, orient='horizontal')
        self.paned.pack(fill='both', expand=True)
        panel_container = ttk.Frame(self.paned, width=460)
        panel_container.pack_propagate(False)
        panel_canvas = tk.Canvas(panel_container, highlightthickness=0, borderwidth=0)
        panel_scroll = ttk.Scrollbar(panel_container, orient='vertical', command=panel_canvas.yview)
        panel_canvas.configure(yscrollcommand=panel_scroll.set)
        panel_canvas.pack(side='left', fill='both', expand=True)
        panel_scroll.pack(side='right', fill='y')
        panel = ttk.Frame(panel_canvas, padding=(0, 0, 10, 0), width=440)
        panel_window = panel_canvas.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', lambda event: panel_canvas.configure(
            scrollregion=panel_canvas.bbox('all')))
        panel_canvas.bind('<Configure>', lambda event: panel_canvas.itemconfigure(
            panel_window, width=event.width))
        plot = ttk.Frame(self.paned)
        # Keep the left pane at its requested width. The sash remains draggable.
        self.paned.add(panel_container, weight=0)
        self.paned.add(plot, weight=1)
        # Keep configuration actions at the top so they are always visible.
        controls = ttk.Frame(panel)
        controls.pack(fill='x', pady=(0, 6))
        ttk.Button(controls, text='加载配置', command=self.import_settings,
                   style='Action.TButton').pack(
            side='left', expand=True, fill='x')
        ttk.Button(controls, text='保存配置', command=self.export_settings,
                   style='Action.TButton').pack(
            side='left', expand=True, fill='x')
        ttk.Label(panel, text='LiDAR ROI / Calibration',
                  font=('TkDefaultFont', 13, 'bold')).pack(anchor='w')
        ttk.Label(panel, text='先建立或加载地图，再确认 ROI，最后执行选定雷达累计。',
                  wraplength=420).pack(anchor='w')
        self.map_source = tk.StringVar(value='build')
        self.mapping_sensor_vars = {side: tk.BooleanVar(value=(side == 'front')) for side in SIDES}
        self.mapping_sensor_label = tk.StringVar(value='前雷达')
        self.accumulation_vars = {
            side: tk.BooleanVar(value=(side in ('front', 'left')))
            for side in ALL_SIDES
        }
        self.accumulation_selection_label = tk.StringVar(value='前、左雷达')
        # Static capture is now the only supported mode.  Keep this internal
        # variable for compatibility with saved settings and worker code, but
        # do not expose a dynamic-mode checkbox in the UI.
        self.static_mapping_var = tk.BooleanVar(value=True)
        source_box = ttk.LabelFrame(panel, text='独立地图来源', padding=4)
        source_box.pack(fill='x', pady=(7, 3))
        ttk.Radiobutton(source_box, text='从原始 bag 建图', variable=self.map_source,
                        value='build', command=self.update_map_source).pack(anchor='w')
        ttk.Radiobutton(source_box, text='使用已有静态地图', variable=self.map_source,
                        value='existing', command=self.update_map_source).pack(anchor='w')
        ttk.Label(source_box, text='独立建图雷达（必须包含前雷达）').pack(anchor='w', pady=(6, 0))
        self.mapping_menu_button = ttk.Menubutton(source_box, textvariable=self.mapping_sensor_label)
        self.mapping_menu = tk.Menu(self.mapping_menu_button, tearoff=False)
        for side, label in {'front': '前雷达', 'left': '左雷达', 'right': '右雷达',
                            'top': '顶部雷达'}.items():
            self.mapping_menu.add_checkbutton(
                label=label, variable=self.mapping_sensor_vars[side],
                command=self.update_mapping_sensor_label)
        self.mapping_menu.add_separator()
        self.mapping_menu.add_command(label='全选', command=lambda: self.set_mapping_sensors(SIDES))
        self.mapping_menu.add_command(label='清空', command=lambda: self.set_mapping_sensors(()))
        self.mapping_menu_button['menu'] = self.mapping_menu
        self.mapping_menu_button.pack(fill='x')
        self.bag = self.path_entry(panel, '原始 ROS bag', self.pick_bag)
        self.map_bundle_dir = self.path_entry(
            panel, '已有静态地图目录', self.pick_map_bundle_dir)
        self.map_bundle_widgets = self.last_path_widgets
        # These two variables remain as internal compatibility paths.  The user
        # selects one directory; the front files are discovered automatically.
        self.trajectory = tk.StringVar()
        self.front_map_entry = tk.StringVar()
        self.build_button = ttk.Button(panel, text='建立选定雷达独立地图', command=self.build_map,
                                       style='Primary.TButton')
        self.build_button.pack(fill='x', pady=(5, 3))
        self.save_map_button = ttk.Button(
            panel, text='保存选定独立雷达地图',
            command=self.save_selected_independent_maps, style='Action.TButton')
        self.save_map_button.pack(fill='x', pady=(0, 3))
        ttk.Label(panel, text='请将所有雷达静态地图放在同一目录下。',
                  wraplength=370).pack(anchor='w', pady=(0, 3))
        # 外参只用于后续选定雷达的 ROI 累计；独立建图阶段始终使用各自坐标系。
        extrinsic_box = ttk.LabelFrame(panel, text='外参设置（用于 ROI 累计）', padding=5)
        extrinsic_box.pack(fill='x', pady=(8, 4))
        ttk.Label(extrinsic_box, text='外参来源').pack(anchor='w')
        self.source = tk.StringVar()
        sources = ttk.Combobox(extrinsic_box, textvariable=self.source, state='readonly',
                              values=tuple(SOURCE_LABELS.values()))
        sources.pack(fill='x')
        sources.bind('<<ComboboxSelected>>', lambda _: self.update_source())
        self.extrinsics = self.path_entry(extrinsic_box, '标定外参 YAML', self.pick_extrinsics)
        self.extrinsics_widgets = self.last_path_widgets
        region = ttk.LabelFrame(panel, text='前雷达地图坐标系（世界坐标）中的累计区域 [m]', padding=6)
        region.pack(fill='x', pady=(12, 4))
        self.bounds = [self.numbers(region, axis + ' min / max', 2) for axis in 'XYZ']
        display = ttk.LabelFrame(panel, text='显示设置', padding=6)
        display.pack(fill='x', pady=4)
        self.voxel = self.numbers(display, '显示体素 [m]', 1)[0]
        ttk.Button(display, text='应用显示设置', command=self.redraw,
                   style='Action.TButton').pack(fill='x', pady=3)
        self.accumulation_box = ttk.LabelFrame(
            display, text='参与累计和标定的雷达', padding=4)
        self.accumulation_box.pack(fill='x', pady=(3, 3))
        for side in ALL_SIDES:
            ttk.Checkbutton(
                self.accumulation_box,
                text={'front': '前雷达', 'left': '左雷达', 'right': '右雷达',
                      'top': '顶部雷达'}[side],
                variable=self.accumulation_vars[side],
                command=self.update_calibration_selection
            ).pack(side='left', padx=(0, 8))
        ttk.Label(
            self.accumulation_box, textvariable=self.accumulation_selection_label,
            foreground='#5e6d82').pack(anchor='w', pady=(2, 0))

        self.layers = {side: tk.BooleanVar(value=True) for side in ALL_SIDES}
        layer_box = ttk.LabelFrame(display, text='显示累计点云', padding=4)
        layer_box.pack(fill='x', pady=(3, 3))
        for side in ALL_SIDES:
            ttk.Checkbutton(
                layer_box, text={'front': '前雷达', 'left': '左雷达', 'right': '右雷达',
                                 'top': '顶部雷达'}[side],
                variable=self.layers[side], command=self.redraw).pack(side='left', padx=(0, 8))

        map_box = ttk.LabelFrame(display, text='独立雷达地图', padding=4)
        map_box.pack(fill='x', pady=(2, 3))
        for column, side in enumerate(SIDES):
            ttk.Checkbutton(
                map_box,
                text={'front': '前雷达地图', 'left': '左雷达地图', 'right': '右雷达地图',
                      'top': '顶部雷达地图'}[side],
                variable=self.show_independent_maps[side], command=self.redraw
            ).grid(row=0, column=column, sticky='w', padx=(0, 10))

        # Static capture has no motion trajectory layer to display.
        # Compatibility aliases for older methods and configuration handling.
        self.show_map = self.show_independent_maps['front']
        self.show_trajectory = self.show_independent_trajectories['front']
        self.confirm_roi_button = ttk.Button(panel, text='确定 ROI', command=self.confirm_roi,
                                             style='Primary.TButton')
        self.confirm_roi_button.pack(fill='x', pady=(8, 2))
        self.max_frames = self.numbers(panel, '累计帧数（0=全部）', 1)[0]
        self.max_frames.set('0')
        ttk.Label(panel, text='每个已勾选雷达最多读取前 N 帧；0 表示使用全部帧。',
                  wraplength=420).pack(anchor='w', pady=(0, 2))
        self.accumulate_button = ttk.Button(
            panel, text='使用外参累计：前、左雷达', command=self.load, style='Primary.TButton')
        self.accumulate_button.pack(fill='x', pady=(0, 4))
        # Compatibility alias used by the existing async build/accumulation code.
        self.load_button = self.accumulate_button
        self.confirm_roi_button.state(['disabled'])
        self.accumulate_button.state(['disabled'])
        processing = ttk.LabelFrame(panel, text='点云处理设置', padding=6)
        processing.pack(fill='x', pady=4)
        ttk.Label(processing, text='滤波参数').pack(anchor='w')
        self.analysis_voxel = self.numbers(processing, '处理体素 [m]', 1)[0]
        self.filter_radius = self.numbers(processing, '邻域半径 [m]', 1)[0]
        self.filter_neighbors = self.numbers(processing, '最少邻居数', 1)[0]
        ttk.Label(processing, text='平面拟合参数').pack(anchor='w', pady=(5, 0))
        self.plane_threshold = self.numbers(processing, '平面内点阈值 [m]', 1)[0]
        self.plane_min_inliers = self.numbers(processing, '最少平面内点数', 1)[0]
        self.plane_max_planes = self.numbers(processing, '最多提取平面数', 1)[0]
        self.plane_max_iterations = self.numbers(processing, 'RANSAC 最大迭代次数', 1)[0]
        self.plane_min_angle = self.numbers(processing, '平面最小夹角 [deg]', 1)[0]
        self.plane_max_span = self.numbers(processing, '平面最大跨度 [m]', 1)[0]
        self.plane_min_span = self.numbers(processing, '平面最小跨度 [m]', 1)[0]
        ttk.Label(processing, text='平面最小夹角用于避免把同一个面误拆成多个平面。').pack(anchor='w')
        ttk.Label(processing, text='平面最大跨度用于拒绝大于目标面的拉伸点云。').pack(anchor='w')
        ttk.Label(processing, text='平面最小跨度用于拒绝过窄的条带或背景平面；0 表示不限制。').pack(anchor='w')
        ttk.Label(processing, text='拟合默认使用滤波后的点云；没有执行滤波时使用原始 ROI。').pack(anchor='w')
        self.filter_button = ttk.Button(processing, text='执行滤波', command=self.filter_result,
                                        style='Action.TButton')
        self.filter_button.pack(fill='x', pady=(5, 2))
        actions = ttk.Frame(processing)
        actions.pack(fill='x', pady=2)
        actions.columnconfigure((0, 1), weight=1, uniform='plane_actions')
        self.plane_button = ttk.Button(actions, text='执行平面拟合和匹配', command=self.fit_planes,
                                       style='Action.TButton')
        self.plane_button.grid(row=0, column=0, sticky='ew', padx=(0, 2))
        self.save_result_button = ttk.Button(actions, text='修正并保存外参指标', command=self.save_validation,
                                             style='Action.TButton', state='disabled')
        self.save_result_button.grid(row=0, column=1, sticky='ew', padx=(2, 0))
        from lidar_cube_joint_panel import JointPanel
        self.joint_panel = JointPanel(self, panel, load_settings)
        from lidar_cube_six_panel import SixPanel
        self.six_panel = SixPanel(self, panel, load_settings)
        self.config_label = tk.StringVar()
        ttk.Label(panel, textvariable=self.config_label, wraplength=370).pack(anchor='w', pady=6)
        self.status = tk.StringVar(value=warning or '请先建立或加载前雷达地图。')
        status_box = ttk.LabelFrame(panel, text='运行状态', padding=5)
        status_box.pack(fill='x', pady=6)
        ttk.Label(status_box, textvariable=self.status, wraplength=420,
                  justify='left', anchor='nw').pack(fill='x')
        # 指标单独显示，避免运行状态更新时把已经生成的指标覆盖掉。
        self.metrics_summary = tk.StringVar(value='请先执行平面拟合和匹配。')
        metrics_box = ttk.LabelFrame(panel, text='标定评价指标', padding=5)
        metrics_box.pack(fill='x', pady=6)
        ttk.Label(metrics_box, textvariable=self.metrics_summary, wraplength=420,
                  justify='left', anchor='nw').pack(fill='x')
        ttk.Label(panel, text='操作提示：拖动鼠标旋转，滚轮缩放。\n独立地图按各自坐标显示；累计结果统一到前雷达地图坐标系。\n颜色：蓝色前雷达，绿色左雷达，橙色右雷达。',
                  wraplength=420).pack(anchor='w', pady=6)
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.axes = self.figure.add_subplot(projection='3d')
        self.axes.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
        self.figure.subplots_adjust(left=.03, right=.97, bottom=.03, top=.95)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        NavigationToolbar2Tk(self.canvas, plot)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        frame_bar = ttk.Frame(plot, padding=(4, 3))
        frame_bar.pack(fill='x', side='bottom')
        ttk.Button(frame_bar, text='◀', width=4, command=lambda: self.step_frame(-1)).pack(side='left')
        self.frame_counter = tk.StringVar(value='Frame -- / --')
        ttk.Label(frame_bar, textvariable=self.frame_counter, width=18, anchor='center').pack(side='left')
        ttk.Button(frame_bar, text='▶', width=4, command=lambda: self.step_frame(1)).pack(side='left')
        ttk.Label(frame_bar, text='  步进帧数').pack(side='left')
        self.frame_step_var = tk.StringVar(value='1')
        ttk.Entry(frame_bar, textvariable=self.frame_step_var, width=6).pack(side='left')
        ttk.Radiobutton(frame_bar, text='单帧', variable=self.frame_mode,
                        value='single', command=self.redraw).pack(side='left', padx=(12, 2))
        ttk.Radiobutton(frame_bar, text='累计', variable=self.frame_mode,
                        value='cumulative', command=self.redraw).pack(side='left')
        self.bottom_status = tk.StringVar(value='状态：等待建立或加载前雷达地图。')
        ttk.Label(plot, textvariable=self.bottom_status, anchor='w',
                  style='Status.TLabel').pack(fill='x', side='bottom', padx=6)
        self.canvas.mpl_connect('scroll_event', self.zoom)
        self.apply_settings(settings)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _schedule_after(self, delay_ms, callback):
        """安全地调度 Tk 回调，并在窗口关闭时自动取消。"""
        if self._closing:
            return None
        callback_id = None

        def invoke():
            self._after_ids.discard(callback_id)
            if self._closing:
                return
            try:
                callback()
            except self.tk.TclError:
                # 文件对话框/主窗口关闭时，Tk 可能已经销毁部分控件。
                return
            except Exception as error:
                # 后台结果回调中的绘图异常不能让 Tk 主循环进入异常状态；
                # 将错误留在状态栏，窗口仍可继续操作和重新执行。
                if not self._closing:
                    try:
                        self.status.set(f'界面刷新失败：{error}')
                    except self.tk.TclError:
                        pass

        callback_id = self.root.after(delay_ms, invoke)
        self._after_ids.add(callback_id)
        return callback_id

    def path_entry(self, parent, label, browse):
        self.ttk.Label(parent, text=label).pack(anchor='w', pady=(7, 0))
        row = self.ttk.Frame(parent)
        row.pack(fill='x')
        variable = self.tk.StringVar()
        entry = self.ttk.Entry(row, textvariable=variable, width=43)
        entry.pack(side='left', fill='x', expand=True)
        button = self.ttk.Button(row, text='选择', command=browse)
        button.pack(side='left')
        self.last_path_widgets = (entry, button)
        return variable

    @staticmethod
    def _data_root_from_path(path):
        """Return the dataset directory represented by a file or directory.

        A user may select a rosbag metadata/db3 file, the dataset directory,
        or a generated ``mapYYYY...`` directory.  All three forms resolve to
        the dataset directory that owns ``Cube_Viewer_config.yaml``.
        """
        root = Path(path).expanduser().resolve()
        if root.is_file():
            root = root.parent
        if root.name.startswith('map') and root.parent.is_dir():
            root = root.parent
        return root

    def set_data_root(self, path):
        """Switch all future output/configuration to one dataset directory."""
        if not path:
            return
        root = self._data_root_from_path(path)
        root.mkdir(parents=True, exist_ok=True)
        if self.data_root is not None and Path(self.data_root) != root:
            # Do not let a map output from the previous dataset be reused
            # when the user switches to another bag/map directory.
            self.build_output = None
        self.data_root = root
        self.config_path = root / 'Cube_Viewer_config.yaml'
        if hasattr(self, 'config_label'):
            self.config_label.set(f'配置保存位置\n{self.config_path}')
        self.remember_config_path()

    def get_data_root(self):
        """Resolve the current dataset directory, with a safe workspace fallback."""
        if self.data_root is not None:
            return Path(self.data_root)
        for variable in (self.bag, self.map_bundle_dir, self.front_map_entry):
            value = variable.get().strip()
            if value:
                self.set_data_root(value)
                return Path(self.data_root)
        raise ValueError('请先选择一个标定包。')

    def autosave_current_settings(self):
        """Persist the current UI state in the selected dataset directory."""
        try:
            save_settings(self.config_path, self.collect_settings())
            self.autosave = True
        except Exception as error:
            self.status.set(f'配置自动保存失败：{error}')

    def numbers(self, parent, label, count):
        row = self.ttk.Frame(parent)
        row.pack(fill='x', pady=2)
        self.ttk.Label(row, text=label, width=21).pack(side='left')
        variables = [self.tk.StringVar() for _ in range(count)]
        for variable in variables:
            self.ttk.Entry(row, textvariable=variable, width=9).pack(side='left', padx=2)
        return variables

    def update_source(self):
        state = 'normal' if self.source.get() == SOURCE_LABELS['yaml'] else 'disabled'
        for widget in self.extrinsics_widgets:
            widget.configure(state=state)

    def selected_calibration_sides(self):
        """返回当前勾选、将参与 ROI 累计和标定的雷达。"""
        return tuple(side for side in ALL_SIDES if self.accumulation_vars[side].get())

    @staticmethod
    def calibration_sides_text(sides):
        labels = {'front': '前雷达', 'left': '左雷达', 'right': '右雷达',
                  'top': '顶部雷达'}
        return '、'.join(labels[side] for side in sides) if sides else '未选择雷达'

    @staticmethod
    def calibration_pair_names(sides):
        """Return every unordered pair among the selected calibration radars."""
        return tuple(
            f'{sides[index]}-{sides[other]}'
            for index in range(len(sides))
            for other in range(index + 1, len(sides))
        )

    def update_calibration_selection(self):
        """更新参与累计的雷达摘要和按钮文字。"""
        sides = self.selected_calibration_sides()
        text = self.calibration_sides_text(sides)
        self.accumulation_selection_label.set(text)
        self.accumulate_button.configure(text=f'使用外参累计：{text}')
        if 'front' not in sides:
            self.status.set('参与累计和标定的雷达必须包含前雷达。')
        elif len(sides) < 2:
            self.status.set('请至少选择前雷达和一个其他雷达。')
        else:
            pairs = '、'.join(self.calibration_pair_names(sides))
            self.status.set(f'当前参与标定：{text}；实际计算配对：{pairs}')

    def accumulation_target_text(self):
        """兼容旧状态提示，返回当前参与累计的雷达名称。"""
        return self.calibration_sides_text(self.selected_calibration_sides())

    def result_sides(self, clouds=None):
        """返回当前累计结果中的雷达。"""
        if clouds is None:
            clouds = self.result.clouds if self.result is not None else {}
        return tuple(side for side in ALL_SIDES if side in clouds)

    def selected_mapping_sides(self):
        return [side for side in SIDES if self.mapping_sensor_vars[side].get()]

    def update_mapping_sensor_label(self):
        labels = {'front': '前雷达', 'left': '左雷达', 'right': '右雷达',
                  'top': '顶部雷达'}
        selected = [labels[side] for side in self.selected_mapping_sides()]
        self.mapping_sensor_label.set('、'.join(selected) if selected else '未选择雷达')

    def set_mapping_sensors(self, sides):
        selected = set(sides)
        for side in SIDES:
            self.mapping_sensor_vars[side].set(side in selected)
        self.update_mapping_sensor_label()

    def update_map_source(self):
        use_existing = self.map_source.get() == 'existing'
        state = 'normal' if use_existing else 'disabled'
        for entry, button in (self.map_bundle_widgets,):
            entry.configure(state=state)
            button.configure(state=state)
        self.build_button.configure(state='disabled' if use_existing else 'normal')
        if not use_existing:
            self.map_ready = False
            self.roi_confirmed = False
            self.status.set('当前模式会从原始 bag 建立前雷达地图。')
        else:
            self.status.set('请选择包含所有雷达静态地图的同一目录。')
        self.update_roi_buttons()

    def update_roi_buttons(self):
        """根据地图、轨迹和累计状态控制 ROI 两步操作按钮。"""
        if not hasattr(self, 'confirm_roi_button'):
            return
        self.confirm_roi_button.configure(
            text='ROI 已确定 ✓' if self.roi_confirmed else '确定 ROI')
        busy = self.pending or self.build_process is not None
        has_map_and_trajectory = self.map_ready and (
            self.static_mapping_var.get() or len(self.trajectory_rows) >= 2)
        if busy:
            self.confirm_roi_button.state(['disabled'])
            self.accumulate_button.state(['disabled'])
        else:
            self.confirm_roi_button.state(['!disabled'] if has_map_and_trajectory else ['disabled'])
            self.accumulate_button.state(['!disabled'] if self.roi_confirmed else ['disabled'])

    def confirm_roi(self):
        """只确认当前 ROI 范围，不读取外参，也不启动点云累计。"""
        if self.pending or self.build_process is not None:
            return
        selected = self.selected_calibration_sides()
        if 'front' not in selected:
            self.status.set('参与累计和标定的雷达必须包含前雷达。')
            return
        if len(selected) < 2:
            self.status.set('请至少选择前雷达和一个其他雷达。')
            return
        if not self.map_ready or (not self.static_mapping_var.get()
                                  and len(self.trajectory_rows) < 2):
            self.status.set('请先建立或加载前雷达地图。')
            return
        if self.map_source.get() == 'existing' and not self.static_mapping_var.get():
            try:
                self.validate_input_pair()
            except Exception as error:
                self.status.set(str(error))
                return
        try:
            bounds = np.array([[float(value.get()) for value in pair] for pair in self.bounds])
        except (TypeError, ValueError):
            self.status.set('ROI 范围必须填写数字。')
            return
        if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])):
            self.status.set('ROI 范围必须满足每个轴 min < max。')
            return
        self.roi_confirmed = True
        self.set_bounds(bounds)
        self.status.set(
            f'ROI 已确定。当前参与标定：{self.calibration_sides_text(selected)}；'
            f'实际计算配对：{"、".join(self.calibration_pair_names(selected))}。')
        self.bottom_status.set('Status: ROI confirmed; waiting for extrinsic accumulation.')
        self.update_roi_buttons()
        self.redraw()
        self.root.update_idletasks()

    def _clear_roi_markers(self):
        """删除上一次绘制的 ROI 边框。"""
        for marker in self.roi_markers:
            try:
                marker.remove()
            except Exception:
                pass
        self.roi_markers.clear()

    def _draw_roi_box(self):
        """在当前 3D 坐标系中绘制已确认的 XYZ ROI 立方框。"""
        if not self.roi_confirmed:
            return
        try:
            bounds = np.array([[float(value.get()) for value in pair] for pair in self.bounds])
        except (TypeError, ValueError):
            return
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
            return
        x0, x1 = bounds[0]
        y0, y1 = bounds[1]
        z0, z1 = bounds[2]
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = ((0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7))
        for start, end in edges:
            marker, = self.axes.plot(
                corners[[start, end], 0], corners[[start, end], 1], corners[[start, end], 2],
                color='#00bcd4', linestyle='--', linewidth=1.2,
                alpha=0.9, label='_nolegend_')
            self.roi_markers.append(marker)

    def _crop_points_to_roi(self, points):
        """按当前 XYZ 范围筛选用于显示的点云，不修改原始地图数据。"""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            return points[:0]
        try:
            bounds = np.array([[float(value.get()) for value in pair] for pair in self.bounds])
        except (TypeError, ValueError):
            return points[:0]
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
            return points[:0]
        mask = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
        return points[mask]

    def validate_input_pair(self):
        bag_name = Path(self.bag.get().strip()).expanduser().resolve()
        if bag_name.name == 'metadata.yaml':
            bag_name = bag_name.parent
        elif bag_name.is_file() and bag_name.suffix == '.db3':
            # 允许用户直接选择单个 SQLite bag 文件；其数据身份仍然是
            # 所在 rosbag 目录，而不是 xxx_0.db3 文件名。
            bag_name = bag_name.parent
        expected = bag_name.name
        trajectory_text = self.trajectory.get().strip()
        if not trajectory_text:
            raise ValueError('已有地图模式缺少前雷达轨迹，请选择包含 front_trajectory.json 的地图目录。')
        trajectory_path = Path(trajectory_text).expanduser().resolve()
        if not trajectory_path.is_file():
            raise ValueError(f'前雷达轨迹文件不存在或不是文件：{trajectory_path}')
        trajectory_data = json.loads(trajectory_path.read_text(encoding='utf-8'))
        trajectory_bag = trajectory_data.get('bag')
        map_path = Path(self.front_map_entry.get().strip()).expanduser().resolve()
        if not map_path.is_file():
            raise ValueError(f'前雷达地图文件不存在或不是文件：{map_path}')
        meta_path = map_path.with_name('front_map_meta.json')
        map_bag = None
        if meta_path.is_file():
            map_bag = json.loads(meta_path.read_text(encoding='utf-8')).get('bag')
        if trajectory_bag and trajectory_bag != expected:
            raise ValueError(f'输入不匹配：轨迹属于 {trajectory_bag}，当前 bag 是 {expected}。')
        if map_bag and map_bag != expected:
            raise ValueError(f'输入不匹配：地图属于 {map_bag}，当前 bag 是 {expected}。')
        if map_bag and trajectory_bag and map_bag != trajectory_bag:
            raise ValueError('输入不匹配：地图和轨迹来自不同的 bag。')

    def pick_bag(self):
        path = self.dialog.askopenfilename(title='Select ROS 2 bag metadata.yaml or .db3',
                                          initialdir=DATA_DIR,
                                          filetypes=[('ROS 2 SQLite bag', '*.db3 *.yaml')],
                                          parent=self.root)
        if path:
            self.bag.set(path)
            self.set_data_root(path)
            self.autosave_current_settings()

    def pick_map_bundle_dir(self):
        """选择包含各雷达静态地图的统一目录。"""
        path = self.dialog.askdirectory(
            title='选择包含所有雷达静态地图的目录',
            initialdir=DATA_DIR,
            mustexist=True,
            parent=self.root)
        if path:
            try:
                self.map_source.set('existing')
                self.update_map_source()
                self.map_bundle_dir.set(path)
                self.set_data_root(path)
                self.load_map_bundle_directory(path)
                self.autosave_current_settings()
            except Exception as error:
                self.status.set(f'加载地图目录失败：{error}')
                try:
                    self.messagebox.showerror('加载失败', str(error), parent=self.root)
                except Exception:
                    pass

    def _clear_independent_maps(self):
        for side in SIDES:
            self.independent_map_points[side] = np.empty((0, 3), dtype=np.float32)
            self.independent_trajectory_points[side] = np.empty((0, 3), dtype=np.float32)
            self.independent_trajectory_rows[side] = []
        self.independent_map_paths.clear()
        self.front_map_points = np.empty((0, 3), dtype=np.float32)
        self.trajectory_points = np.empty((0, 3), dtype=np.float32)
        self.trajectory_rows = []
        self.front_map_path = ''
        self.map_ready = False
        self.roi_confirmed = False

    def load_map_bundle_directory(self, directory):
        """从同一目录自动加载各雷达的 *_map.npy 和 *_trajectory.json。"""
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError('请选择目录，而不是单个文件。')
        # Loading an existing bundle must not reuse a previous build output
        # when the save button is pressed later.
        self.build_output = None
        # 用户有时会选择 rosbag 父目录；自动进入其中的地图目录，
        # 避免把 bag 目录误当成地图目录。一个 bag 目录可能保留多次建图
        # 结果，此时优先选择名称最新的目录（通常是最新时间戳）。
        required = ('front_map.npy',)
        selected_note = ''
        if not all((directory / name).is_file() for name in required):
            candidates = [child for child in sorted(directory.iterdir())
                          if child.is_dir()
                          and all((child / name).is_file() for name in required)]
            if not candidates:
                raise ValueError(
                    f'目录中没有找到 front_map.npy：{directory}')
            if len(candidates) > 1:
                # Prefer the new ``mapYYYY...`` naming over legacy
                # ``mapping_YYYY...`` directories, then choose the newest
                # directory by timestamp/mtime.
                candidates.sort(key=lambda child: (
                    child.name.startswith('map') and not child.name.startswith('mapping_'),
                    child.stat().st_mtime, child.name),
                    reverse=True)
                selected_note = (
                    f'；检测到 {len(candidates)} 个地图目录，已自动选择最新目录：'
                    f'{candidates[0].name}')
            directory = candidates[0]
        self._clear_independent_maps()
        loaded, missing, errors = [], [], []
        for side in SIDES:
            map_path = directory / f'{side}_map.npy'
            trajectory_path = directory / f'{side}_trajectory.json'
            if not map_path.is_file() and not trajectory_path.is_file():
                missing.append(side)
                continue
            if not map_path.is_file():
                errors.append(f'{side} 缺少地图文件')
                continue
            try:
                self.load_independent_map(
                    side, map_path, trajectory_path if trajectory_path.is_file() else None)
                if side == 'front':
                    self.front_map_entry.set(str(map_path))
                    self.trajectory.set(str(trajectory_path) if trajectory_path.is_file() else '')
                    self.load_front_map(map_path)
                    if trajectory_path.is_file():
                        self.load_trajectory(trajectory_path)
                loaded.append(side)
            except Exception as error:
                errors.append(f'{side}：{error}')
        if 'front' not in loaded:
            raise ValueError('目录中必须包含 front_map.npy。')
        self.map_bundle_dir.set(str(directory))
        self.map_source.set('existing')
        self.update_map_source()
        detail = '已加载：' + '、'.join(loaded) + selected_note
        if missing:
            detail += '；未找到：' + '、'.join(missing)
        if errors:
            detail += '；问题：' + '；'.join(errors)
        self.status.set(detail)
        self.bottom_status.set('Status: loaded independent LiDAR maps from one directory.')
        self.redraw()

    def load_front_map(self, path):
        path = Path(path).expanduser().resolve()
        self.front_map_entry.set(str(path))
        points = np.asarray(np.load(path, mmap_mode='r'), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Front map must be a finite Nx3 NumPy point cloud.')
        if len(points) < 100:
            raise ValueError('Front map contains too few points.')
        self.front_map_points = points
        self.front_map_path = str(path)
        self.independent_map_points['front'] = points
        self.independent_map_paths['front'] = self.front_map_path
        self.show_independent_maps['front'].set(True)
        self.map_ready = True
        self.roi_confirmed = False
        if self.result is None:
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            margin = np.maximum((maximum - minimum) * .05, .1)
            self.set_bounds(np.column_stack((minimum - margin, maximum + margin)))
        self.redraw()
        self.update_roi_buttons()

    def load_independent_map(self, side, map_path, trajectory_path=None):
        """加载独立坐标系地图；静态地图不要求运动轨迹文件。"""
        points = np.asarray(np.load(Path(map_path), mmap_mode='r'), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f'{side} map must be a finite Nx3 NumPy point cloud.')
        if trajectory_path and Path(trajectory_path).is_file():
            data = json.loads(Path(trajectory_path).read_text(encoding='utf-8'))
            rows = normalize_pose_rows(data.get('poses', []))
            if len(rows) < 2:
                raise ValueError(f'{side} trajectory needs at least two pose samples.')
        else:
            # Display-only fixed poses keep the existing trajectory widgets
            # compatible without pretending that a moving trajectory exists.
            rows = normalize_pose_rows([
                [0, 0., 0., 0., 0., 0., 0., 1.],
                [1, 0., 0., 0., 0., 0., 0., 1.],
            ])
        self.independent_map_points[side] = points
        self.independent_trajectory_rows[side] = rows
        self.independent_trajectory_points[side] = np.asarray(
            [row[1:4] for row in rows], dtype=np.float32)
        self.independent_map_paths[side] = str(Path(map_path).expanduser().resolve())
        self.show_independent_maps[side].set(True)
        self.show_independent_trajectories[side].set(True)
        self.redraw()
        self.update_roi_buttons()

    def load_trajectory(self, path):
        data = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
        normalized_rows = normalize_pose_rows(data.get('poses', []))
        rows = np.asarray(normalized_rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) < 2:
            raise ValueError('Front trajectory needs at least two pose samples.')
        if not np.isfinite(rows[:, 1:4]).all():
            raise ValueError('Front trajectory contains non-finite positions.')
        self.trajectory_points = rows[:, 1:4]
        self.trajectory_rows = normalized_rows
        self.independent_trajectory_rows['front'] = normalized_rows
        self.independent_trajectory_points['front'] = self.trajectory_points
        self.show_independent_trajectories['front'].set(True)
        self.redraw()
        self.update_roi_buttons()

    def save_selected_independent_maps(self):
        """保存选中的独立雷达静态地图和元数据。

        每个雷达保留自己的地图坐标系；这里不会使用标定外参把地图互相对齐。
        """
        selected_sides = self.selected_mapping_sides()
        if not selected_sides or 'front' not in selected_sides:
            message = '保存独立地图时必须包含前雷达；左、右、顶部雷达可以按需追加。'
            self.status.set(message)
            self.bottom_status.set('Status: save blocked; front map is required.')
            try:
                self.messagebox.showwarning('无法保存', message, parent=self.root)
            except Exception:
                pass
            return
        missing = []
        for side in selected_sides:
            if len(self.independent_map_points[side]) == 0:
                missing.append(f'{side} 地图')
        if missing:
            message = '无法保存，以下数据尚未加载：' + '、'.join(missing)
            self.status.set(message)
            self.bottom_status.set('Status: save blocked; map is missing.')
            try:
                self.messagebox.showwarning('无法保存', message, parent=self.root)
            except Exception:
                pass
            return
        try:
            data_root = self.get_data_root()
            # Reuse the just-built map directory when it already contains the
            # selected static maps.  Maps loaded from an older directory are
            # copied into a fresh timestamped map directory instead.
            output = Path(self.build_output) if self.build_output else None
            if output is None or not all(
                    (output / f'{side}_map.npy').is_file() for side in selected_sides):
                output = data_root / (
                    'map' + datetime.now().strftime('%Y%m%d_%H%M%S'))
            output.mkdir(parents=True, exist_ok=True)
            records = []
            for side in selected_sides:
                points = np.asarray(self.independent_map_points[side], dtype=np.float32)
                map_name = f'{side}_map.npy'
                meta_name = f'{side}_map_meta.json'
                np.save(output / map_name, points)
                metadata = {
                    'sensor_side': side,
                    'map_points': int(len(points)),
                    'poses': 0,
                    'mapping_mode': 'static',
                    'source_map': self.independent_map_paths.get(side, ''),
                    'coordinate_frame': f'{side}_map',
                    'world_frame': 'front_map' if side == 'front' else None,
                }
                (output / meta_name).write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
                records.append(metadata)
                if side == 'front':
                    self.front_map_entry.set(str((output / map_name).resolve()))
                    self.trajectory.set('')
                    self.front_map_path = str((output / map_name).resolve())
            (output / 'independent_maps_meta.json').write_text(json.dumps({
                'schema_version': 1,
                'source': 'lidar_cube_viewer/manual_save',
                'selected_sides': selected_sides,
                'maps': records,
            }, indent=2, ensure_ascii=False), encoding='utf-8')
            self.map_bundle_dir.set(str(output.resolve()))
            self.status.set('选定独立雷达静态地图已保存到\n' + str(output))
            self.bottom_status.set('Status: selected static maps saved.')
            self.autosave_current_settings()
        except Exception as error:
            message = f'保存独立雷达静态地图失败：{error}'
            self.status.set(message)
            self.bottom_status.set('Status: save failed.')
            try:
                self.messagebox.showerror('保存失败', message, parent=self.root)
            except Exception:
                pass

    def build_map(self):
        if self.pending or self.build_process is not None:
            return
        bag = self.bag.get().strip()
        if not bag:
            self.status.set('Select the original bag before building the front map.')
            return
        # 当前查看器只支持静态采集。旧配置中的动态模式标志会被忽略。
        self.static_mapping_var.set(True)
        sides = self.selected_mapping_sides()
        if not sides:
            self.status.set('独立建图必须包含前雷达；左、右、顶部雷达可以按需追加。')
            return
        if 'front' not in sides:
            self.status.set('独立建图必须包含前雷达，前雷达地图将作为世界坐标。')
            return
        # 独立地图默认保存到当前 bag 所在目录，避免所有 bag 的结果混在
        # workspace/data/lidar_cube_independent_maps 下。
        bag_path = Path(bag).expanduser().resolve()
        self.set_data_root(bag_path)
        output = self.get_data_root() / (
            'map' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.map_ready = False
        self.roi_confirmed = False
        self.build_sides = list(sides)
        self.build_output = output
        self.build_process = True
        self.update_roi_buttons()
        self.status.set('正在建立选定雷达的静态独立地图（不使用 NDT 运动轨迹），请保持窗口打开……')
        self.bottom_status.set('状态：正在执行静态点云累计。')
        Thread(target=self.build_worker, args=(output, sides, bag), daemon=True).start()
        self._schedule_after(200, self.receive)

    def build_worker(self, output, sides, bag):
        """顺序建立多个独立地图，避免多个 mapping 节点争用同一 ROS 服务。"""
        builder = Path(__file__).with_name('front_map_builder.py')
        output.mkdir(parents=True, exist_ok=True)
        labels = {'front': '前', 'left': '左', 'right': '右', 'top': '顶部'}
        for index, side in enumerate(sides, start=1):
            self.events.put(('progress',
                             f'[{index}/{len(sides)}] 开始建立{labels.get(side, side)}雷达独立地图……'))
            command = [sys.executable, str(builder), '--static', '--sensor-side', side,
                       '--bag', bag, '--output', str(output)]
            try:
                process = subprocess.Popen(command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                self.build_process = process
            except Exception as error:
                self.events.put(('error', f'{side} 雷达建图无法启动：{error}'))
                self.events.put(('build_done', None))
                return
            output_lines = []
            if process.stdout is not None:
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        output_lines.append(line)
                        self.events.put(('progress',
                                         f'[{index}/{len(sides)}] {labels.get(side, side)}雷达建图：' + line))
            process.wait()
            if process.returncode != 0:
                self.events.put(('error', f'{side} 雷达建图失败，请查看日志：'
                                 f'{output / f"{side}_mapping.launch.log"}'))
                self.events.put(('build_done', None))
                return
            self.events.put(('progress',
                             f'[{index}/{len(sides)}] {labels.get(side, side)}雷达静态累计完成，正在加载地图……'))
            try:
                meta = json.loads(output_lines[-1])
                self.events.put(('map', (side, Path(meta['output']))))
            except Exception as error:
                self.events.put(('error', f'{side} 雷达建图返回结果无效：{error}'))
                self.events.put(('build_done', None))
                return
        self.events.put(('build_done', (Path(output), list(sides))))

    def pick_extrinsics(self):
        path = self.dialog.askopenfilename(title='Select calibrated extrinsics YAML',
                                          initialdir=DATA_DIR,
                                          filetypes=[('Extrinsics YAML', '*.yaml *.yml')],
                                          parent=self.root)
        if path:
            self.extrinsics.set(path)

    def apply_settings(self, settings):
        request = settings.request
        self.bag.set(request.bag)
        self.trajectory.set(request.trajectory)
        self.front_map_entry.set(settings.front_map)
        self.map_bundle_dir.set(settings.map_bundle_dir)
        self.joint_panel.set_output_parent(settings.joint_output_parent)
        self.build_output = None
        data_root_candidate = settings.map_bundle_dir or request.bag or settings.front_map
        if data_root_candidate:
            self.set_data_root(data_root_candidate)
        self.map_source.set(settings.map_source)
        self.set_mapping_sensors(settings.mapping_sensors)
        self.static_mapping_var.set(True)
        self.extrinsics.set(request.extrinsics)
        self.max_frames.set(str(request.max_frames))
        self.source.set(SOURCE_LABELS[request.extrinsics_source])
        selected = set(request.active_sides())
        for side in ALL_SIDES:
            self.accumulation_vars[side].set(side in selected)
        self.update_calibration_selection()
        for variables, values in zip(self.bounds, request.bounds):
            for variable, value in zip(variables, values):
                variable.set(str(value))
        self.voxel.set(str(settings.display_voxel))
        self.analysis_voxel.set(str(settings.analysis_voxel))
        self.filter_radius.set(str(settings.filter_radius))
        self.filter_neighbors.set(str(settings.filter_min_neighbors))
        self.plane_threshold.set(str(settings.plane_threshold))
        self.plane_min_inliers.set(str(settings.plane_min_inliers))
        self.plane_max_planes.set(str(settings.plane_max_planes))
        self.plane_max_iterations.set(str(settings.plane_max_iterations))
        self.plane_min_angle.set(str(settings.plane_min_angle))
        self.plane_max_span.set(str(settings.plane_max_span))
        self.plane_min_span.set(str(settings.plane_min_span))
        for side in ALL_SIDES:
            self.layers[side].set(settings.visible[side])
        for side in SIDES:
            self.show_independent_maps[side].set(settings.show_independent_maps[side])
            self.show_independent_trajectories[side].set(
                settings.show_independent_trajectories[side])
        self.update_source()
        self.update_map_source()
        self.config_label.set(f'配置保存位置\n{self.config_path}')
        if settings.map_bundle_dir and Path(settings.map_bundle_dir).is_dir():
            try:
                self.load_map_bundle_directory(settings.map_bundle_dir)
            except Exception as error:
                self.status.set(f'地图目录无法加载：{error}')
        elif settings.front_map and Path(settings.front_map).is_file():
            try:
                self.load_front_map(settings.front_map)
            except Exception as error:
                self.status.set(f'Front map could not be loaded: {error}')
        if request.trajectory and Path(request.trajectory).is_file():
            try:
                self.load_trajectory(request.trajectory)
            except Exception as error:
                self.status.set(f'Front trajectory could not be loaded: {error}')

    def collect_settings(self):
        source = {label: key for key, label in SOURCE_LABELS.items()}[self.source.get()]
        use_existing = self.map_source.get() == 'existing'
        request = AccumulationRequest(
            bag=self.bag.get().strip(),
            trajectory='',
            bounds=np.array([[float(v.get()) for v in pair] for pair in self.bounds]),
            extrinsics_source=source,
            extrinsics=self.extrinsics.get().strip(),
            calibration_sides=self.selected_calibration_sides(),
            max_frames=int(self.max_frames.get()),
            static_mapping=True)
        settings = ViewerSettings(
            request,
            front_map=self.front_map_entry.get().strip(),
            map_bundle_dir=self.map_bundle_dir.get().strip(),
            joint_output_parent=self.joint_panel.output_parent,
            map_source=self.map_source.get(),
            mapping_sensors=self.selected_mapping_sides(),
            static_mapping=True,
            display_voxel=float(self.voxel.get()),
            visible={side: var.get() for side, var in self.layers.items()},
            show_map=self.show_map.get(),
            show_trajectory=self.show_trajectory.get(),
            show_independent_maps={side: var.get()
                                   for side, var in self.show_independent_maps.items()},
            show_independent_trajectories={side: var.get()
                                           for side, var in self.show_independent_trajectories.items()},
            analysis_voxel=float(self.analysis_voxel.get()),
            filter_radius=float(self.filter_radius.get()),
            filter_min_neighbors=int(self.filter_neighbors.get()),
            plane_threshold=float(self.plane_threshold.get()),
             plane_min_inliers=int(self.plane_min_inliers.get()),
             plane_max_planes=int(self.plane_max_planes.get()),
             plane_max_iterations=int(self.plane_max_iterations.get()),
             plane_min_angle=float(self.plane_min_angle.get()),
             plane_max_span=float(self.plane_max_span.get()),
             plane_min_span=float(self.plane_min_span.get()))
        settings.validate()
        return settings

    def import_settings(self):
        if self.pending:
            self.status.set('Wait for the current accumulation to finish before loading settings.')
            return
        path = self.dialog.askopenfilename(
            title='Load viewer settings', initialdir=self.config_path.parent,
            filetypes=[('Viewer settings YAML', '*.yaml *.yml')],
            parent=self.root)
        if path:
            try:
                settings = load_settings(path)
                self.config_path = Path(path)
                self.remember_config_path()
                self.apply_settings(settings)
                self.autosave = True
                # 加载配置只恢复界面参数和地图/轨迹，不自动执行 ROI 累计。
                # 用户需要先确认当前 ROI，再明确点击当前选定雷达的累计按钮。
                self.roi_confirmed = False
                self.update_roi_buttons()
                self.status.set('配置已加载。请先点击“确定 ROI”，再开始外参累计。')
            except Exception as error:
                self.status.set(f'Could not load settings: {error}')

    def export_settings(self):
        try:
            settings = self.collect_settings()
            self.config_path = self.get_data_root() / 'Cube_Viewer_config.yaml'
            save_settings(self.config_path, settings)
            self.remember_config_path()
            self.autosave = True
            self.config_label.set(f'配置保存位置\n{self.config_path}')
            self.status.set(f'配置已保存到\n{self.config_path}')
        except Exception as error:
            self.status.set(f'配置保存失败：{error}')

    def remember_config_path(self):
        LAST_CONFIG_POINTER.parent.mkdir(parents=True, exist_ok=True)
        LAST_CONFIG_POINTER.write_text(str(self.config_path.resolve()), encoding='utf-8')

    def load(self):
        if self.pending or self.processing:
            return
        if not self.roi_confirmed:
            self.status.set('请先点击“确定 ROI”，确认范围后再使用外参累计。')
            self.bottom_status.set('Status: ROI has not been confirmed.')
            self.update_roi_buttons()
            return
        if self.map_source.get() == 'existing' and not self.static_mapping_var.get():
            try:
                self.validate_input_pair()
            except Exception as error:
                self.status.set(str(error))
                return
        if not self.map_ready:
            self.status.set('Build or load the front global map before confirming ROI.')
            return
        if not self.static_mapping_var.get() and not self.trajectory.get().strip():
            self.status.set('动态模式需要前雷达轨迹；当前程序仅支持静态采集。')
            return
        try:
            settings = self.collect_settings()
        except Exception as error:
            self.status.set(str(error))
            return
        # 新一轮 ROI 累计开始时立即移除旧结果，避免累计过程中继续显示上一轮点云。
        self.save_result_button.state(['disabled'])
        self.report_directory = None
        self.result = None
        self.processed_clouds = None
        self.filter_display = False
        self.plane_results = None
        self.metrics_results = None
        self.metrics_summary.set('请先执行平面拟合和匹配。')
        self.frame_index = 0
        self.view_key = None
        self._clear_plane_markers()
        self.redraw()
        self.pending = True
        self.cancelled.clear()
        self.load_button.state(['disabled'])
        try:
            frame_limit = int(settings.request.max_frames)
        except (TypeError, ValueError):
            frame_limit = 0
        frame_text = '全部帧' if frame_limit == 0 else f'前 {frame_limit} 帧'
        self.status.set(f'正在把每个选定雷达的{frame_text}累计到选定区域，请稍候……')
        self.bottom_status.set(
            f'Status: accumulating {frame_text} from {self.accumulation_target_text()}...')
        Thread(target=self.worker, args=(settings,), daemon=True).start()
        self._schedule_after(100, self.receive)

    def worker(self, settings):
        """计算线程只写队列；Tk 控件始终由主线程操作。"""
        try:
            result = accumulate(settings.request, progress=lambda text: self.events.put(('progress', text)),
                                cancelled=self.cancelled.is_set)
            self.events.put(('result', (result, settings)))
        except Exception as error:
            self.events.put(('error', str(error)))

    def receive(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == 'progress':
                    self.status.set(payload)
                    self.bottom_status.set('Status: ' + payload)
                    continue
                if kind == 'build_done':
                    self.build_process = None
                    self.update_roi_buttons()
                    output, sides = payload if payload else (self.build_output, self.build_sides)
                    if output:
                        self.map_bundle_dir.set(str(Path(output).resolve()))
                        self.set_data_root(output)
                        self.autosave_current_settings()
                    labels = {'front': '前', 'left': '左', 'right': '右', 'top': '顶部'}
                    names = '、'.join(labels.get(side, side) + '雷达' for side in sides)
                    self.status.set(
                        f'独立建图全部完成：{names}。\n'
                        f'静态地图已保存到：{output}\n可以继续确认 ROI。')
                    self.bottom_status.set('Status: all selected static LiDAR maps are ready.')
                    continue
                if kind == 'map':
                    side, payload = payload
                    try:
                        map_path = payload / f'{side}_map.npy'
                        trajectory_path = payload / f'{side}_trajectory.json'
                        self.load_independent_map(
                            side, map_path,
                            trajectory_path if trajectory_path.is_file() else None)
                        if side == 'front':
                            self.map_source.set('existing')
                            self.trajectory.set(str(trajectory_path) if trajectory_path.is_file() else '')
                            self.front_map_entry.set(str(map_path))
                            if trajectory_path.is_file():
                                self.load_trajectory(trajectory_path)
                            self.load_front_map(map_path)
                            self.update_map_source()
                        self.status.set(f'{side} 雷达静态独立地图已保存：{payload}')
                        self.bottom_status.set('Status: static independent map ready.')
                    except Exception as error:
                        self.status.set(f'{side} 雷达地图已生成，但自动加载失败：{error}')
                    continue
                self.pending = False
                self.update_roi_buttons()
                if kind == 'error':
                    self.build_process = None
                    self.update_roi_buttons()
                    prefix = 'Previous result remains displayed.\n' if self.result else ''
                    self.status.set(prefix + payload)
                    return
                self.result, settings = payload
                self.processed_clouds = None
                self.filter_display = False
                self.plane_results = None
                self.metrics_results = None
                self.report_directory = None
                self.save_result_button.state(['disabled'])
                self.metrics_summary.set('请先执行平面拟合和匹配。')
                self.roi_confirmed = True
                request = self.result.request
                key = (request.bag, request.trajectory, tuple(request.bounds.ravel()), self.result.frame)
                if key != self.view_key:
                    self.set_bounds(request.bounds)
                    self.axes.view_init(elev=28, azim=-55)
                    self.view_key = key
                self.redraw()
                if self.autosave:
                    try:
                        save_settings(self.config_path, settings)
                    except Exception as error:
                        self.status.set(self.status.get() + f'\nSettings save failed: {error}')
                return
        except Empty:
            self._schedule_after(100, self.receive)

    def set_bounds(self, bounds):
        for setter, limits in zip((self.axes.set_xlim, self.axes.set_ylim, self.axes.set_zlim), bounds):
            setter(*limits)
        self.axes.set_box_aspect(bounds[:, 1] - bounds[:, 0])

    def _clear_plane_markers(self):
        """删除上一次拟合结果的面片和内点标记。"""
        for marker in self.plane_markers + self.plane_inlier_markers:
            try:
                marker.remove()
            except (AttributeError, ValueError):
                pass
        self.plane_markers.clear()
        self.plane_inlier_markers.clear()

    def _plane_corners(self, points, normal):
        """根据平面内点范围生成一个用于显示的矩形面片。"""
        points = np.asarray(points, dtype=float)
        normal = np.asarray(normal, dtype=float)
        length = np.linalg.norm(normal)
        if len(points) < 3 or length < 1e-12:
            return None
        normal = normal / length
        # 选择一个与法向量不平行的坐标轴，构造平面内的两个正交方向。
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        u = np.cross(normal, axis)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        center = points.mean(axis=0)
        pu = (points - center) @ u
        pv = (points - center) @ v
        margin_u = max((pu.max() - pu.min()) * .03, .01)
        margin_v = max((pv.max() - pv.min()) * .03, .01)
        lo_u, hi_u = pu.min() - margin_u, pu.max() + margin_u
        lo_v, hi_v = pv.min() - margin_v, pv.max() + margin_v
        return np.array([
            center + u * lo_u + v * lo_v,
            center + u * hi_u + v * lo_v,
            center + u * hi_u + v * hi_v,
            center + u * lo_u + v * hi_v,
        ])

    def _draw_planes(self):
        """在当前三维坐标轴中画出各雷达拟合得到的平面。"""
        if not self.plane_results:
            return
        for side in self.result_sides(self.result.clouds if self.result else {}):
            if not self.layers[side].get():
                continue
            for index, plane in enumerate(self.plane_results.get(side, []), start=1):
                points = getattr(plane, 'points', None)
                if points is None or len(points) < 3:
                    continue
                corners = self._plane_corners(points, plane.normal)
                if corners is None:
                    continue
                color = COLORS[side]
                patch = self.Poly3DCollection(
                    [corners], facecolors=color, edgecolors=color,
                    linewidths=1.4, alpha=.28,
                    # Poly3DCollection 在部分 Matplotlib 版本中不能直接
                    # 作为 legend 句柄，否则会访问不存在的 _facecolors2d。
                    label='_nolegend_')
                self.axes.add_collection3d(patch)
                self.plane_markers.append(patch)
                # 用较小透明点标出真正参与拟合的内点，便于判断面片是否覆盖点云。
                self.plane_inlier_markers.append(self.axes.scatter(
                    *np.asarray(points).T, s=7, c=color, alpha=.18,
                    depthshade=False, label='_nolegend_'))

    def _update_legend(self):
        """只把 Matplotlib 支持的句柄加入 3D 图例。"""
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()
        handles, labels = self.axes.get_legend_handles_labels()
        safe_handles, safe_labels = [], []
        for handle, label in zip(handles, labels):
            if not label or label.startswith('_'):
                continue
            if isinstance(handle, self.Poly3DCollection):
                continue
            safe_handles.append(handle)
            safe_labels.append(label)
        if safe_handles:
            self.axes.legend(safe_handles, safe_labels, loc='upper right')

    @staticmethod
    def _plane_metrics(plane):
        """返回拟合面的 RMS 距离和两个面内跨度，便于判断结果是否可信。"""
        points = getattr(plane, 'points', None)
        if points is None or len(points) < 3:
            return 0.0, (0.0, 0.0)
        points = np.asarray(points, dtype=float)
        residuals = np.abs(points @ plane.normal + plane.offset)
        return float(np.sqrt(np.mean(residuals ** 2))), plane_spans(points)

    def _metrics_status_lines(self):
        metrics = self.metrics_results or {}
        labels = {'front': '前', 'left': '左', 'right': '右', 'top': '顶'}
        lines = []
        corrected = 'extrinsic_corrections' in metrics
        if not corrected:
            for side, fit in metrics.get('sensor_summary', {}).items():
                angle = fit.get('max_right_angle_deviation_deg')
                detail = (f'最大直角偏差 {angle:.2f}°' if angle is not None else
                          f"拟合 {fit['plane_count']} 面，需要 3 面")
                lines.append(f"{labels.get(side, side)}雷达　{fit['quality']}　{detail}")
        if metrics.get('pairs'):
            lines.append('新外参 · 对应面匹配' if corrected else '当前外参 · 对应面匹配')
        for pair, current in metrics.get('pairs', {}).items():
            name = '–'.join(labels.get(side, side) for side in pair.split('-'))
            score = metrics.get('saved_pair_validation', {}).get(pair) if corrected else current
            if not score:
                lines.append(f'{name}　无可用的修正后匹配指标')
                continue
            corner, angle = score.get('corner_distance_m'), score.get('mean_normal_error_deg')
            distance = '—' if corner is None else f'{corner * 100:.2f} cm'
            normal = '—' if angle is None else f'{angle:.2f}°'
            lines.append(f"{name}　{score['quality']}　交点距离 {distance}　平均法向夹角 {normal}")
        if corrected and not metrics.get('saved_pair_validation'):
            reason = (metrics.get('extrinsics', {}).get('error') or
                      metrics.get('three_radar_optimization', {}).get('reason') or
                      next((c['reason'] for c in metrics['extrinsic_corrections'].values() if c.get('reason')), ''))
            if reason:
                lines.append(reason)
        if corrected and self.report_directory:
            lines.append(f'结果目录\n{self.report_directory}')
        return lines

    @staticmethod
    def _metrics_log(metrics):
        """按阶段保存逐面诊断，便于追查一次验收为什么通过或失败。"""
        lines = []
        for side in metrics.get('calibration_sides', list(metrics.get('per_sensor', {}))):
            fit = metrics.get('sensor_summary', {}).get(side, {})
            lines.append(f"{side} 平面拟合 {fit.get('quality', '—')} {fit.get('reason', '')}".rstrip())
            for pair in fit.get('plane_angles', []):
                lines.append(f"  P{pair['plane_a_index']} ↔ P{pair['plane_b_index']}  "
                             f"无向夹角 {pair['angle_deg']:.2f}°  直角偏差 {pair['deviation_deg']:.2f}°")
            for plane in metrics.get('per_sensor', {}).get(side, []):
                long_span, short_span = plane['span_m']
                lines.append(f"  P{plane['plane_index']} {plane['quality']}  "
                             f"内点 {plane['inlier_count']}（{plane['inlier_ratio']:.2%}）  "
                             f"跨度 {long_span:.3f} × {short_span:.3f} m  "
                             f"RMS {1000*plane['rms_m']:.2f} mm  "
                             f"P95 {1000*plane['p95_point_plane_distance_m']:.2f} mm")
        joint = metrics.get('three_radar_optimization', {})
        for pair, current in metrics.get('pairs', {}).items():
            correction = metrics.get('extrinsic_corrections', {}).get(pair, {})
            for label, score in [('当前', current), ('候选', correction.get('candidate_validation', {})),
                                 ('联合复验', joint.get('final_validation', {}).get(pair, {}))]:
                if not score:
                    continue
                lines.append(f"{pair} {label} {score['quality']} {score.get('reason', '')}".rstrip())
                for match in score.get('matches', []):
                    lines.append(f"  P{match['plane_a_index']} ↔ P{match['plane_b_index']}  {match['quality']}  "
                                 f"夹角 {match['normal_error_deg']:.2f}°  "
                                 f"双向 RMS {100*match['point_to_plane_rms_m']:.2f} cm  "
                                 f"P95 {100*match['point_to_plane_p95_m']:.2f} cm")
            if correction.get('reason'):
                lines.append(correction['reason'])
        if joint:
            lines.append(f"联合验收 {joint.get('quality', '未执行')} {joint.get('reason', '')}".rstrip())
        return '\n'.join(lines or [metrics.get('message', '尚无评价指标。')]) + '\n'

    def redraw(self):
        has_independent_map = any(len(points) for points in self.independent_map_points.values())
        if self.result is None and not self.map_ready and not has_independent_map:
            return
        try:
            voxel = float(self.voxel.get())
            if not np.isfinite(voxel) or voxel < 0:
                raise ValueError
        except ValueError:
            self.status.set('显示体素必须大于等于 0，填 0 表示显示全部 ROI 点。')
            return
        for marker in self.markers:
            marker.remove()
        self.markers.clear()
        for marker in self.map_markers:
            marker.remove()
        self.map_markers.clear()
        for marker in self.independent_map_markers:
            marker.remove()
        self.independent_map_markers.clear()
        for marker in self.independent_trajectory_markers:
            marker.remove()
        self.independent_trajectory_markers.clear()
        self._clear_roi_markers()
        self._clear_plane_markers()
        if self.trajectory_marker is not None:
            self.trajectory_marker.remove()
            self.trajectory_marker = None
        independent_roi_counts = {}
        for side in SIDES:
            points = self.independent_map_points[side]
            if self.show_independent_maps[side].get() and len(points):
                if self.roi_confirmed:
                    points = self._crop_points_to_roi(points)
                independent_roi_counts[side] = len(points)
                if not len(points):
                    continue
                # 独立地图显示严格使用界面设置的显示体素；不再额外强制
                # 5 cm 下限，否则用户填写 0.01 m 仍会看到 0.05 m 的稀疏结果。
                shown = thin(points, voxel)
                self.independent_map_markers.append(self.axes.scatter(
                    *shown.T, s=1, c=COLORS[side], alpha=.30,
                    depthshade=False,
                    label=f'{side.title()} independent map'))
        # Static capture has no motion trajectory to draw.
        self._draw_roi_box()
        if self.result is None:
            if self.roi_confirmed:
                try:
                    bounds_text = '；'.join(
                        f'{axis}[{float(pair[0].get()):.2f}, {float(pair[1].get()):.2f}]'
                        for axis, pair in zip('XYZ', self.bounds))
                except (TypeError, ValueError):
                    bounds_text = 'XYZ 范围读取失败，请重新确认。'
                self.axes.set_title('ROI 已确定 | 等待外参累计', fontsize=10)
                count_text = '，'.join(
                    f'{side} {independent_roi_counts.get(side, 0):,} 点'
                    for side in SIDES if self.show_independent_maps[side].get())
                self.status.set(
                    f'ROI 已确定：{bounds_text}\n'
                    f'ROI 内显示点数：{count_text or "未勾选独立地图"}\n'
                    '右侧显示的是各独立雷达落在 ROI 内的点云；青色虚线框为当前累计区域。\n'
                    f'现在可以点击“使用外参累计{self.accumulation_target_text()}”。')
            else:
                self.axes.set_title('Front map frame = world | select ROI', fontsize=10)
                self.status.set(
                    f'前雷达静态地图：{len(self.front_map_points):,} 点。'
                    '请输入 XYZ 范围并确认 ROI。')
            self.canvas.draw_idle()
            return
        lines = []
        result_sides = self.result_sides()
        frame_total = max((len(self.result.frame_clouds.get(side, []))
                           for side in result_sides), default=0)
        if frame_total:
            self.frame_index = min(max(self.frame_index, 0), frame_total - 1)
            self.frame_counter.set(f'Frame {self.frame_index + 1} / {frame_total}')
        else:
            self.frame_counter.set('Frame -- / --')
        for side in result_sides:
            points = self.result.clouds[side]
            if self.filter_display and self.processed_clouds is not None:
                points = self.processed_clouds[side]
            elif self.frame_mode.get() == 'single' and frame_total:
                per_frame = self.result.frame_clouds.get(side, [])
                points = per_frame[self.frame_index] if self.frame_index < len(per_frame) else points[:0]
            elif self.frame_mode.get() == 'cumulative' and frame_total:
                per_frame = self.result.frame_clouds.get(side, [])[:self.frame_index + 1]
                points = np.concatenate(per_frame) if per_frame else points[:0]
            elif self.processed_clouds is not None:
                points = self.processed_clouds[side]
            shown = thin(points, voxel) if self.layers[side].get() else points[:0]
            if len(shown):
                self.markers.append(self.axes.scatter(*shown.T, s=4, c=COLORS[side],
                                                      depthshade=False, label=side.title()))
            stats = self.result.stats[side]
            lines.append(f'{side}: ROI {len(points):,}, shown {len(shown):,}\n'
                         f'  frames {stats["used_frames"]}/{stats["total_frames"]}, '
                         f'ROI frames {stats["roi_frames"]}, pose skips {stats["pose_skips"]}')
        self._draw_planes()
        metric_lines = self._metrics_status_lines()
        if self.metrics_results is None:
            self.metrics_summary.set('请先执行平面拟合和匹配。')
        elif metric_lines:
            self.metrics_summary.set('\n'.join(metric_lines))
        else:
            self.metrics_summary.set('指标报告为空，请重新执行多平面拟合。')
        self._update_legend()
        # 标记使用已完成结果的来源，输入栏尚未应用时也不会误标点云。
        self.axes.set_title(f'{self.result.frame}\n{SOURCE_LABELS[self.result.request.extrinsics_source]}', fontsize=10)
        self.status.set('\n'.join(lines) + '\n\nROI 全部点云已保留，可供后续评估。\n'
                        '当前使用扫描时间戳进行整帧位姿变换。')
        self.bottom_status.set('Status: ' + ' | '.join(
            f'{side} {len(self.result.frame_clouds.get(side, []))} frames'
            for side in result_sides))
        self.canvas.draw_idle()

    def step_frame(self, direction):
        if self.result is None:
            return
        try:
            step = max(1, int(self.frame_step_var.get()))
        except (TypeError, ValueError):
            self.frame_step_var.set('1')
            step = 1
        result_sides = self.result_sides()
        total = max((len(self.result.frame_clouds.get(side, []))
                     for side in result_sides), default=0)
        if total:
            self.frame_index = min(max(self.frame_index + direction * step, 0), total - 1)
            self.redraw()

    def filter_result(self):
        """在后台处理当前 ROI，保留 result.clouds 作为原始评估输入。"""
        if self.result is None or self.processing:
            if self.result is None:
                self.status.set(
                    f'请先确认 ROI 并累计{self.accumulation_target_text()}。')
            return
        self.processing = True
        self.save_result_button.state(['disabled'])
        self.filter_button.state(['disabled'])
        self.status.set('正在执行体素降采样和邻域滤波，请稍候……')
        self.bottom_status.set('Status: filtering ROI clouds...')
        try:
            config = FilterConfig(
                voxel_size=float(self.analysis_voxel.get()),
                neighbor_radius=float(self.filter_radius.get()),
                min_neighbors=int(self.filter_neighbors.get()))
        except (TypeError, ValueError):
            self.processing = False
            self.filter_button.state(['!disabled'])
            if self.metrics_results:
                self.save_result_button.state(['!disabled'])
            self.status.set('滤波参数无效，请检查体素、邻域半径和邻居数。')
            return
        Thread(target=self.filter_worker, args=(config,), daemon=True).start()
        self._schedule_after(100, self.receive_filter)

    def fit_planes(self):
        """对当前选定雷达的 ROI 点云执行多平面 RANSAC，并绘制拟合面。"""
        if self.result is None or self.processing:
            if self.result is None:
                self.status.set(
                    f'请先确认 ROI 并累计{self.accumulation_target_text()}。')
            return
        try:
            threshold = float(self.plane_threshold.get())
            min_inliers = int(self.plane_min_inliers.get())
            max_planes = int(self.plane_max_planes.get())
            max_iterations = int(self.plane_max_iterations.get())
            min_angle = float(self.plane_min_angle.get())
            max_span = float(self.plane_max_span.get())
            min_span = float(self.plane_min_span.get())
            if (threshold <= 0 or min_inliers <= 0 or max_planes <= 0
                    or max_iterations <= 0 or not 0 < min_angle < 90
                    or max_span <= 0 or min_span < 0):
                raise ValueError
        except (TypeError, ValueError):
            self.status.set('平面拟合参数无效，请检查阈值、内点数、平面数、迭代次数、夹角和跨度。')
            return
        clouds = self.processed_clouds if self.filter_display and self.processed_clouds is not None else self.result.clouds
        source_name = '滤波后' if self.filter_display and self.processed_clouds is not None else '原始'
        self.processing = True
        self.save_result_button.state(['disabled'])
        self.filter_button.state(['disabled'])
        self.plane_button.state(['disabled'])
        self.plane_results = self.metrics_results = self.report_directory = None
        self.metrics_summary.set('正在拟合平面和匹配当前外参……')
        self.status.set(f'正在使用{source_name}点云拟合平面和匹配……')
        self.bottom_status.set('Status: fitting multiple planes...')
        Thread(target=self.plane_worker,
               args=(clouds, threshold, min_inliers, max_planes, max_iterations,
                     min_angle, max_span, min_span),
               daemon=True).start()
        self._schedule_after(100, self.receive_plane)

    def plane_worker(self, clouds, threshold, min_inliers, max_planes, max_iterations,
                     min_angle, max_span, min_span):
        try:
            results = {}
            for side in self.result_sides(clouds):
                points = np.asarray(clouds.get(side, np.empty((0, 3))), dtype=float)
                if len(points) < max(3, min_inliers):
                    results[side] = []
                    continue
                planes, _ = extract_planes(
                    points,
                    distance_threshold=threshold,
                    max_iterations=max_iterations,
                    min_inliers=min_inliers,
                    max_planes=max_planes,
                    random_seed=0,
                    min_plane_angle_deg=min_angle,
                    max_plane_span=max_span,
                    min_plane_span=min_span,
                )
                results[side] = planes
            metrics = evaluate_metrics(
                clouds, results,
                calibration_sides=self.result.request.active_sides(), include_corrections=False)
            self.plane_events.put(('planes', (results, metrics)))
        except Exception as error:
            self.plane_events.put(('plane_error', str(error)))

    def receive_plane(self):
        try:
            while True:
                kind, payload = self.plane_events.get_nowait()
                self.processing = False
                self.filter_button.state(['!disabled'])
                self.plane_button.state(['!disabled'])
                if kind == 'plane_error':
                    self.metrics_results = None
                    self.metrics_summary.set('指标未生成：多平面拟合失败。\n' + payload)
                    self.status.set('多平面拟合失败：' + payload)
                    self.bottom_status.set('Status: plane fitting failed | ' + payload)
                    return
                self.plane_results, self.metrics_results = payload
                if not isinstance(self.metrics_results, dict):
                    self.metrics_results = {'version': 1, 'message': '指标报告格式无效。'}
                self.save_result_button.state(['!disabled'])
                counts = ' | '.join(
                    f'{side}: {len(self.plane_results.get(side, []))} 个平面'
                    for side in self.result_sides())
                warnings = ['FAIL · ' + summary['reason']
                            for summary in self.metrics_results.get('sensor_summary', {}).values()
                            if summary['quality'] == 'FAIL']
                self.redraw()
                message = '平面拟合和当前外参匹配完成：' + counts
                if warnings:
                    message += '\n' + '\n'.join(warnings)
                self.status.set(message)
                self.bottom_status.set('Status: plane fitting complete | ' + counts)
                return
        except Empty:
            self._schedule_after(100, self.receive_plane)

    def filter_worker(self, config):
        try:
            filtered = {side: filter_points(points, config)
                        for side, points in self.result.clouds.items()}
            self.filter_events.put(('filtered', filtered))
        except Exception as error:
            self.filter_events.put(('filter_error', str(error)))

    def receive_filter(self):
        try:
            while True:
                kind, payload = self.filter_events.get_nowait()
                if kind == 'filtered':
                    self.processed_clouds = payload
                    self.filter_display = True
                    self.plane_results = None
                    self.metrics_results = None
                    self.report_directory = None
                    self.save_result_button.state(['disabled'])
                    self.metrics_summary.set('请先执行平面拟合和匹配。')
                    self.processing = False
                    self.filter_button.state(['!disabled'])
                    counts = ' | '.join(
                        f'{side}: {len(self.result.clouds[side]):,} -> {len(payload[side]):,}'
                        for side in self.result_sides())
                    self.status.set('滤波完成，点数变化：' + counts)
                    self.bottom_status.set('Status: filtering complete | ' + counts)
                    self.redraw()
                    return
                if kind == 'filter_error':
                    self.processing = False
                    self.filter_button.state(['!disabled'])
                    if self.metrics_results:
                        self.save_result_button.state(['!disabled'])
                    self.status.set('滤波失败：' + payload)
                    self.bottom_status.set('Status: filtering failed | ' + payload)
                    return
        except Empty:
            self._schedule_after(100, self.receive_filter)

    def save_validation(self):
        if self.processing or self.pending:
            return
        if self.result is None or not self.metrics_results or not self.plane_results:
            self.status.set('请先执行平面拟合和匹配。')
            return
        self.processing = True
        for button in (self.filter_button, self.plane_button, self.save_result_button):
            button.state(['disabled'])
        self.status.set('正在修正外参、复验面面匹配并保存报告……')
        self.bottom_status.set('Status: validating corrected extrinsics...')
        # 快照与当前累计绑定。后台线程只处理数据，所有 Tk 更新留在主线程。
        metrics = {k: self.metrics_results[k] for k in (
            'version', 'calibration_sides', 'pairing_rule', 'thresholds', 'per_sensor', 'sensor_summary', 'pairs')}
        result, planes, front_map = self.result, self.plane_results, self.front_map_path
        parent = self._data_root_from_path(result.request.bag)
        def worker():
            try:
                metrics['extrinsic_corrections'] = compute_extrinsic_corrections(planes, metrics)
                metrics['saved_pair_validation'] = {}
                front_kit = attach_extrinsic_report(result, planes, metrics)
                with tempfile.TemporaryDirectory(dir=parent, prefix='.cube_report_') as stage:
                    staged = Path(stage) / 'report'
                    staged.mkdir()
                    generated = self._write_validation(staged, result, planes, metrics, front_kit, front_map)
                    output = parent / ('result_report' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                    staged.rename(output)
                self.plane_events.put(('saved', (output, metrics, generated)))
            except Exception as error:
                self.plane_events.put(('save_error', str(error)))
        Thread(target=worker, daemon=True).start()
        self._schedule_after(100, self.receive_correction)

    def receive_correction(self):
        try:
            kind, payload = self.plane_events.get_nowait()
        except Empty:
            self._schedule_after(100, self.receive_correction)
            return
        self.processing = False
        for button in (self.filter_button, self.plane_button, self.save_result_button):
            button.state(['!disabled'])
        if kind == 'save_error':
            self.status.set('修正或保存失败：' + payload)
            self.bottom_status.set('Status: saving correction failed.')
            return
        self.report_directory, self.metrics_results, generated = payload
        message = '已生成 suggested_extrinsics.yaml' if generated else '未生成新外参，已保存诊断报告'
        self.autosave_current_settings()
        self.metrics_summary.set('\n'.join(self._metrics_status_lines()))
        self.status.set(f'{message}\n结果目录：{self.report_directory}')
        self.bottom_status.set('Status: correction report saved.')

    @classmethod
    def _write_validation(cls, output, result, planes, metrics, front_kit, front_map):
        for side in result.request.active_sides():
            np.save(output / f'{side}_roi.npy', result.clouds[side])
        plane_metadata = {}
        if planes:
            for side in result.request.active_sides():
                entries = []
                for index, plane in enumerate(planes.get(side, []), start=1):
                    points = np.asarray(plane.points if plane.points is not None else np.empty((0, 3)),
                                        dtype=np.float32)
                    np.save(output / f'{side}_plane_{index}_points.npy', points)
                    rms, spans = cls._plane_metrics(plane)
                    entries.append({
                        'normal': np.asarray(plane.normal, dtype=float).tolist(),
                        'offset': float(plane.offset),
                        'inlier_count': int(len(points)),
                        'rms_m': rms,
                        'spans_m': list(spans),
                        'points_file': f'{side}_plane_{index}_points.npy',
                    })
                plane_metadata[side] = entries
        metadata = {
            'schema_version': 1,
            'bag': result.request.bag,
            'trajectory': result.request.trajectory,
            'front_map': front_map,
            'extrinsics_source': result.request.extrinsics_source,
            'extrinsics': result.request.extrinsics,
            'calibration_sides': list(result.request.active_sides()),
            'pairing_rule': 'all_selected_pairs',
            'bounds_xyz': result.request.bounds.tolist(),
            'stats': result.stats,
            'provenance': result.provenance,
            'planes': plane_metadata,
            'metrics': metrics,
        }
        (output / 'validation.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        (output / 'metrics_report.json').write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
        (output / 'metrics.log').write_text(cls._metrics_log(metrics), encoding='utf-8')
        (output / 'metrics_report.yaml').write_text(
            yaml.safe_dump(metrics, sort_keys=False, allow_unicode=True), encoding='utf-8')
        # 如果三面体条件满足，额外输出可直接对照 calibration_results.yaml
        # 的建议外参。未通过质量门槛的雷达不会写入，避免误用。
        suggested = {}
        # 前雷达是优化参考，但它相对 sensor_kit_base_link 的静态外参
        # 仍然必须写入完整文件，避免建议 YAML 只有左右雷达而无法独立使用。
        joint_required = len(result.request.active_sides()) == 3
        joint_ok = metrics.get('three_radar_optimization', {}).get('available', False)
        for pair, entry in (metrics.get('extrinsics', {}) or {}).items():
            if joint_required and not joint_ok:
                break
            if not isinstance(entry, dict) or not entry.get('available'):
                continue
            new_kit = entry.get('new_sensor_kit')
            child = entry.get('child_frame')
            if not new_kit or not child:
                continue
            suggested[child] = {
                key: float(new_kit[key])
                for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
            }
        if suggested and front_kit:
            front = front_kit
            suggested[front['child_frame']] = {
                key: float(front[key])
                for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
            }
        if suggested:
            (output / 'suggested_extrinsics.yaml').write_text(
                yaml.safe_dump({KIT: suggested}, sort_keys=False, allow_unicode=True),
                encoding='utf-8')
        return bool(suggested)

    def zoom(self, event):
        if event.inaxes is not self.axes or not event.step:
            return
        bounds = np.array([self.axes.get_xlim(), self.axes.get_ylim(), self.axes.get_zlim()])
        center = bounds.mean(axis=1)
        scale = .86 if event.step > 0 else 1 / .86
        self.set_bounds(center[:, None] + (bounds - center[:, None]) * scale)
        self.canvas.draw_idle()

    def close(self):
        self._closing = True
        self.joint_panel.stop.set()
        self.six_panel.stop.set()
        for callback_id in list(self._after_ids):
            try:
                self.root.after_cancel(callback_id)
            except self.tk.TclError:
                pass
        self._after_ids.clear()
        self.cancelled.set()
        if self.autosave:
            try:
                save_settings(self.config_path, self.collect_settings())
            except Exception as error:
                self.messagebox.showwarning('Settings were not saved', str(error), parent=self.root)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Settings YAML restored at startup')
    parser.add_argument('--check', action='store_true', help='Accumulate the full bag without a window')
    args = parser.parse_args()
    if args.config == DEFAULT_CONFIG and LAST_CONFIG_POINTER.is_file():
        remembered = LAST_CONFIG_POINTER.read_text(encoding='utf-8').strip()
        if remembered:
            args.config = Path(remembered).expanduser()
    if args.check:
        settings = load_settings(args.config) if args.config.exists() else default_settings()
        result = accumulate(settings.request, progress=print)
        print(result.frame)
        print(json.dumps(result.stats, indent=2))
        return
    import tkinter as tk
    root = tk.Tk()
    viewer = Viewer(root, config_path=args.config)
    # 恢复设置后由用户启动，避免打开窗口时立即读取多 GB 文件。
    root.mainloop()


if __name__ == '__main__':
    main()
