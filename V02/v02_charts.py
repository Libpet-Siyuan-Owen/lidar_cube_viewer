"""Grouped bars for the operator's result notebook."""
from itertools import combinations
from math import ceil, floor, isfinite, log10
import tkinter as tk
from tkinter import ttk

from lidar_cube_processing import PAIR_WARNING_LIMITS
from v02_batch import RADAR_NAMES, sides_for

STAGES = ('初始', '复验')
LEGEND = ('修正前 · /tf_static', '修正后 · 联合外参')
COLORS = ('#919ca8', '#286db1')
TITLES = (('最大直角偏差（°）', '最大单面拟合 RMS（mm）'),
          ('角点距离（cm）', '对应面平均法向夹角（°）'))


def match_detail(key, match):
    details = [match['quality']]
    if match.get('reason'):
        details.append(match['reason'])
    a, b = (RADAR_NAMES[s] for s in key.split('-'))
    for face in match.get('matches', []):
        if face['quality'] != 'FAIL':
            continue
        issues = []
        for field, label, scale, unit, limit in zip(
                ('normal_error_deg', 'point_to_plane_rms_m', 'point_to_plane_p95_m'),
                ('法向夹角', '点面 RMS', '点面 P95'), (1, 100, 100), ('°', ' cm', ' cm'), PAIR_WARNING_LIMITS):
            value = face.get(field)
            if value is None or not isfinite(value):
                issues.append(f'{label}无有效值')
            elif value > limit:
                issues.append(f'{label} {value*scale:.2f}{unit} > {limit*scale:g}{unit}')
        details.append(f"{a}第 {face['plane_a_index']} 面 ↔ {b}第 {face['plane_b_index']} 面：" + '，'.join(issues))
    return '\n'.join(details)


def chart_rows(records, page, slot):
    """Return a label and two stage readings (left value, right value, tooltip) per group."""
    sides = sides_for(slot)
    keys = sides if page == 0 else [f'{a}-{b}' for a, b in combinations(sides, 2)]
    rows = []
    for key in keys:
        label = '–'.join(RADAR_NAMES[s] for s in key.split('-'))
        readings = []
        for stage in STAGES:
            record = records.get((slot, stage))
            if record is None:
                readings.append(None)
                continue
            stats, pairs = record
            if page == 0:
                fit = stats[key]['summary']
                rms = fit['max_plane_rms_m']
                readings.append((fit['max_right_angle_deviation_deg'], None if rms is None else rms * 1000,
                                 '\n'.join(filter(None, [fit['quality'], fit.get('reason')]))))
            else:
                pair = pairs.get(key)
                if pair is None:
                    readings.append((None, None, '缺少配对结果'))
                else:
                    corner = pair.get('corner_distance_m')
                    readings.append((None if corner is None else corner * 100, pair.get('normal_error_deg'), match_detail(key, pair)))
        rows.append((label, readings))
    return rows


class MetricChart(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, width=1, height=188, background='white', highlightthickness=0)
        self.rows = []
        self.page = 0
        self.hits = {}
        self.tip = None
        self.active_item = None
        self.bind('<Configure>', lambda _: self.draw())
        self.bind('<Motion>', self.hover)
        for event in ('<Leave>', '<Unmap>'):
            self.bind(event, lambda _: self.hide_tip())

    def display(self, page, rows):
        self.page, self.rows = page, rows
        self.draw()

    def draw(self):
        self.hide_tip()
        self.delete('all')
        self.hits.clear()
        width, height = self.winfo_width(), self.winfo_height()
        if not self.rows or width < 100 or height < 80:
            return
        for metric, title in enumerate(TITLES[self.page]):
            x = metric * width / 2
            self.create_text(x + 10, 13, text=title, anchor='w', fill='#24374a', font='TkHeadingFont')
            self.draw_metric(metric, x + 37, x + width/2 - 10, 43, height - 23)
        # Use space beside the right-hand title for the shared legend.
        cursor = width - 10
        for text, color in reversed(list(zip(LEGEND, COLORS))):
            item = self.create_text(cursor, 13, text=text, anchor='e', font='TkSmallCaptionFont')
            x = self.bbox(item)[0] - 4
            self.create_rectangle(x-9, 9, x, 17, fill=color, outline='')
            cursor = x-22

    def draw_metric(self, metric, left, right, top, bottom):
        vals = [r[metric] for _, stages in self.rows for r in stages if r and r[metric] is not None and isfinite(r[metric])]
        peak = max(vals, default=0) or 1
        step = 10 ** floor(log10(peak)) / 2
        maximum = ceil(peak / step) * step
        for fraction in (0, .5, 1):
            y = bottom - fraction * (bottom - top)
            self.create_line(left, y, right, y, fill='#c7d0d9' if fraction == 0 else '#e6ebef')
            self.create_text(left-4, y, text=f'{maximum*fraction:g}', anchor='e', font='TkSmallCaptionFont')
        stride = (right-left)/len(self.rows)
        half = min(25, stride*.225)
        for i, (label, readings) in enumerate(self.rows):
            center = left + (i+.5)*stride
            self.create_text(center, bottom+9, text=label, font='TkSmallCaptionFont')
            for stage, reading in enumerate(readings):
                cx = center + (2*stage-1)*half
                value = reading[metric] if reading else None
                hint = f'{label}　{LEGEND[stage]}\n'
                if value is None or not isfinite(value):
                    text = '待计算' if reading is None else '无有效值'
                    # Split the two labels vertically when both stages have no bar.
                    item = self.create_text(cx, bottom-9-stage*16, text=text, font='TkSmallCaptionFont', fill=COLORS[stage])
                    self.hits[item] = hint + text + ('\n'+reading[2] if reading else '')
                    continue
                y = bottom - value/maximum*(bottom-top)
                bar = self.create_rectangle(cx-half*.81, y, cx+half*.81, bottom, fill=COLORS[stage], outline='')
                item = self.create_text(cx, y-9, text=f'{value:.2f}', font='TkSmallCaptionFont', fill='#293e51')
                self.hits[bar] = self.hits[item] = hint + f'{TITLES[self.page][metric]}　{value:.2f}\n' + reading[2]

    def hide_tip(self):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None
        self.active_item = None

    def hover(self, event):
        items = self.find_withtag('current')
        item = items[0] if items else None
        if item == self.active_item:
            return
        self.hide_tip()
        if item not in self.hits:
            return
        self.active_item = item
        self.tip = tk.Toplevel(self)
        self.tip.withdraw()
        self.tip.overrideredirect(True)
        ttk.Label(self.tip, text=self.hits[item], padding=6, relief='solid', wraplength=420).pack()
        self.tip.update_idletasks()
        x = min(event.x_root+14, self.winfo_screenwidth()-self.tip.winfo_reqwidth())
        y = event.y_root+16
        if y+self.tip.winfo_reqheight() > self.winfo_screenheight():
            y = event.y_root-self.tip.winfo_reqheight()-4
        self.tip.geometry(f'+{max(0,x)}+{max(0,y)}')
        self.tip.deiconify()
