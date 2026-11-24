#!/usr/bin/env python3
"""离线验证的数据层与累计层，不依赖 Tk/Matplotlib 或正在运行的 ROS 节点。

公开入口 accumulate(request) -> AccumulationResult。后续评估模块可以直接
读取 result.clouds（选定雷达的完整 ROI 点云）和 stats，显示抽样不会改变这些数据。
所有变换使用 parent <- child，长度单位为米，时间戳为整数纳秒。
"""
from collections import deque
from contextlib import closing
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage

WORKSPACE = Path(os.environ.get('LIDAR_CUBE_WORKSPACE', Path.home() / 'calibration_ws')).expanduser()
DATA_DIR = WORKSPACE / 'data' if (WORKSPACE / 'data').is_dir() else Path.home()
FRONT = 'rslidar_airy_front'
KIT = 'sensor_kit_base_link'
BASE = 'base_link'
SIDES = ('front', 'left', 'right')
MAPPING_SIDES = ('front', 'left', 'right', 'top')
ALL_SIDES = MAPPING_SIDES
# Compatibility mapping for viewer configuration files written before the
# calibration set became an explicit list.  New code uses
# ``AccumulationRequest.calibration_sides`` instead.
LEGACY_CALIBRATION_TARGETS = {
    'front-left-right': ('front', 'left', 'right'),
    'front-top': ('front', 'top'),
}
COLORS = {'front': '#2b80df', 'left': '#25a05c', 'right': '#e66124',
          'top': '#8e44ad'}
POINT_TYPES = {1:'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4', 8:'f8'}


def lidar_topic(side):
    """Return the real raw point-cloud topic for a logical radar side."""
    return f'/sensing/lidar/{side}/pointcloud_raw_ex'


@dataclass(frozen=True)
class Transform:
    rotation: Rotation
    translation: np.ndarray

    @classmethod
    def identity(cls):
        return cls(Rotation.identity(), np.zeros(3))

    @classmethod
    def from_ros(cls, transform):
        q = transform.rotation
        p = transform.translation
        return cls(Rotation.from_quat([q.x, q.y, q.z, q.w]), np.array([p.x, p.y, p.z]))

    def __matmul__(self, child):
        return Transform(self.rotation * child.rotation, self.apply(child.translation))

    def apply(self, points):
        return self.rotation.apply(points) + self.translation

    def inverse(self):
        rotation = self.rotation.inv()
        return Transform(rotation, rotation.apply(-self.translation))


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def read_xyz(msg):
    fields = {field.name: field for field in msg.fields}
    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({
        'names': list('xyz'),
        'formats': [endian + POINT_TYPES[fields[name].datatype] for name in 'xyz'],
        'offsets': [fields[name].offset for name in 'xyz'],
        'itemsize': msg.point_step,
    })
    cloud = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                       strides=(msg.row_step, msg.point_step))
    xyz = np.stack([cloud[name].ravel() for name in 'xyz'], axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


class Bag:
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        if path.name == 'metadata.yaml':
            path = path.parent
        if path.is_file() and path.suffix == '.db3':
            self.files = [path]
            self.directory = path.parent
        elif path.is_dir():
            self.directory = path
            metadata = path / 'metadata.yaml'
            if metadata.is_file():
                info = yaml.safe_load(metadata.read_text())['rosbag2_bagfile_information']
                self.files = [path / name for name in info['relative_file_paths']]
                # Some bags copied between machines retain a stale filename
                # in metadata.yaml.  Fall back to the actual .db3 files so a
                # static capture can still be opened safely.
                if not self.files or any(not file.is_file() for file in self.files):
                    self.files = sorted(path.glob('*.db3'))
            else:
                self.files = sorted(path.glob('*.db3'))
        else:
            raise ValueError('Select a rosbag directory, metadata.yaml, or a .db3 file.')
        if not self.files or any(file.suffix != '.db3' or not file.is_file() for file in self.files):
            raise ValueError('This viewer requires a ROS 2 SQLite bag with accessible .db3 files.')
        self.topics = set()
        for file in self.files:
            with closing(self.connect(file)) as db:
                self.topics.update(row[0] for row in db.execute('SELECT name FROM topics'))

    @staticmethod
    def connect(file):
        return sqlite3.connect(file.as_uri() + '?mode=ro', uri=True)

    def first_or_last(self, topic, last=False):
        direction = 'DESC' if last else 'ASC'
        matches = []
        for file in self.files:
            with closing(self.connect(file)) as db:
                row = db.execute(
                    f'SELECT timestamp, data FROM messages WHERE topic_id IN '
                    f'(SELECT id FROM topics WHERE name=?) ORDER BY timestamp {direction}, id {direction} LIMIT 1',
                    (topic,),
                ).fetchone()
                if row is not None:
                    matches.append(row)
        return (max if last else min)(matches, key=lambda row: row[0]) if matches else None

    def records(self, topic):
        """顺序读取所有扫描，不以标定帧、关键帧或显示设置筛选输入。"""
        for file in self.files:
            with closing(self.connect(file)) as db:
                query = ('SELECT data FROM messages WHERE topic_id IN '
                         '(SELECT id FROM topics WHERE name=?) ORDER BY timestamp, id')
                for (data,) in db.execute(query, (topic,)):
                    yield data

    def lidar_topics(self):
        return {side: topic for side in ALL_SIDES
                for topic in (lidar_topic(side),)
                if topic in self.topics}


def static_edges(bag):
    edges = {}
    for data in bag.records('/tf_static'):
        for stamped in deserialize_message(data, TFMessage).transforms:
            edge = stamped.header.frame_id, stamped.child_frame_id
            # 使用首次出现的静态外参，避免包内后续更新覆盖初始安装参数。
            if not any(child == edge[1] for _, child in edges):
                edges[edge] = Transform.from_ros(stamped.transform)
    return edges


def find_chain(edges, parent, child):
    """Return a path whose transforms compose into parent <- child."""
    adjacency = {}
    for first, second in edges:
        adjacency.setdefault(first, []).append((second, (first, second), False))
        adjacency.setdefault(second, []).append((first, (first, second), True))
    visited = {parent}
    queue = deque([(parent, [])])
    while queue:
        frame, chain = queue.popleft()
        if frame == child:
            return chain
        for next_frame, edge, invert in adjacency.get(frame, []):
            if next_frame not in visited:
                visited.add(next_frame)
                queue.append((next_frame, chain + [(edge, invert)]))
    raise ValueError(f'TF chain missing: {parent} <- {child}')


def compose_chain(edges, chain, time_ns=None):
    result = Transform.identity()
    for edge, invert in chain:
        transform = edges[edge]
        if isinstance(transform, Timeline):
            transform = transform.at(time_ns)
            if transform is None:
                return None
        result = result @ (transform.inverse() if invert else transform)
    return result


def lookup(edges, parent, child):
    return compose_chain(edges, find_chain(edges, parent, child))


class Timeline:
    def __init__(self, rows, valid=None, static=False):
        if valid is None:
            valid = [True] * len(rows)
        if len(valid) != len(rows):
            raise ValueError('Trajectory validity flags must match pose samples.')
        samples = {}
        for row, flag in zip(rows, valid):
            if len(row) != 8 or not isinstance(row[0], int) or isinstance(row[0], bool):
                raise ValueError('Each pose needs integer timestamp_ns, x/y/z and quaternion xyzw.')
            if not isinstance(flag, bool) or not np.isfinite(row[1:]).all():
                raise ValueError('Invalid trajectory coordinates or validity flags.')
            if np.linalg.norm(row[4:8]) < 1e-10:
                raise ValueError('Trajectory quaternion must have nonzero norm.')
            if row[0] in samples and samples[row[0]] != (row, flag):
                raise ValueError('Conflicting trajectory samples share a timestamp.')
            samples[row[0]] = (row, flag)
        ordered = [samples[key] for key in sorted(samples)]
        rows = [sample[0] for sample in ordered]
        if len(rows) < 2:
            raise ValueError('A moving trajectory requires at least two pose samples.')
        self.valid = np.array([sample[1] for sample in ordered], dtype=bool)
        # 静止采集只需要两端时间戳来描述“整个 bag 内姿态不变”。
        # 动态轨迹仍然使用下面的时间间隔检查，防止在 NDT 轨迹缺口中
        # 插值出没有依据的位姿。
        self.static = bool(static)
        self.origin = int(rows[0][0])
        self.seconds = np.array([(int(row[0]) - self.origin) / 1e9 for row in rows])
        self.positions = np.array([row[1:4] for row in rows])
        self.rotation = Slerp(self.seconds, Rotation.from_quat([row[4:8] for row in rows]))

    def at(self, time_ns):
        time = (int(time_ns) - self.origin) / 1e9
        if self.static:
            # 静止采集时，前/左/右三个话题的首尾时间戳可能相差几百
            # 纳秒，不能用前雷达两端时间戳把其它话题裁掉。只要轨迹
            # 文件已经通过 bag 关联校验，就对整个 bag 返回同一个位姿。
            if not self.valid.all():
                return None
            return Transform(self.rotation(self.seconds[0]), self.positions[0])
        if time < self.seconds[0] or time > self.seconds[-1]:
            return None
        right = np.searchsorted(self.seconds, time)
        if self.seconds[right] == time:
            if not self.valid[right]:
                return None
        elif not (self.valid[right - 1] and self.valid[right]):
            return None
        if 0 < right < len(self.seconds) and self.seconds[right] != time:
            if self.seconds[right] - self.seconds[right - 1] > .35:
                return None
        translation = np.array([np.interp(time, self.seconds, axis) for axis in self.positions.T])
        return Transform(self.rotation(time), translation)


def normalize_pose_row(row):
    """把轨迹中的一行统一为 [timestamp_ns, x, y, z, qx, qy, qz, qw]。

    JSON 或 NumPy 往返后时间戳有时会变成整数值的 float；这种值仍然是
    同一个纳秒时间戳，可以安全转换回 int。也兼容带字段名的位姿对象。
    """
    if isinstance(row, dict):
        row = [row['timestamp_ns'], row['x'], row['y'], row['z'],
               row.get('qx', row.get('quaternion_x')),
               row.get('qy', row.get('quaternion_y')),
               row.get('qz', row.get('quaternion_z')),
               row.get('qw', row.get('quaternion_w'))]
    if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 8:
        raise ValueError('Each pose needs timestamp_ns, x/y/z and quaternion xyzw.')
    timestamp = row[0]
    if isinstance(timestamp, (bool, np.bool_)):
        raise ValueError('Each pose needs an integer timestamp_ns.')
    try:
        timestamp_float = float(timestamp)
    except (TypeError, ValueError):
        raise ValueError('Each pose needs an integer timestamp_ns.') from None
    if not np.isfinite(timestamp_float) or not timestamp_float.is_integer():
        raise ValueError('Each pose needs an integer timestamp_ns.')
    return [int(timestamp_float)] + [float(value) for value in row[1:]]


def normalize_pose_rows(rows):
    return [normalize_pose_row(row) for row in rows]


def poses_from_file(path, bag, static):
    data = json.loads(Path(path).expanduser().read_text())
    if not isinstance(data, dict) or not {'frame', 'poses'} <= data.keys():
        raise ValueError('Trajectory JSON requires frame and poses.')
    if data.get('timestamp_unit', 'ns') != 'ns' or data.get('schema_version', 1) != 1:
        raise ValueError('Unsupported trajectory schema or timestamp unit.')
    if data.get('bag') and data['bag'] != bag.directory.name:
        raise ValueError(f'Trajectory belongs to {data["bag"]}, selected bag is {bag.directory.name}.')
    poses = normalize_pose_rows(data['poses'])
    static_trajectory = (
        data.get('mapping_mode') == 'static'
        or data.get('source') == 'lidar_cube_viewer/static_mapping'
    )
    timeline = Timeline(poses, data.get('valid'), static=static_trajectory)
    # 新导出文件没有 bag 路径。用扫描头时间戳验证数据关联，防止误选其他录制。
    topic = bag.lidar_topics()['front']
    stamps = {row[0] for row in poses}
    # 旧版界面曾把纳秒时间戳经过 float64 保存，可能产生几十到几百纳秒的
    # 舍入误差。允许很小的时间戳误差，同时仍能拦截选错 bag 的轨迹文件。
    bag_stamps = [stamp_ns(deserialize_message(raw, PointCloud2).header.stamp)
                  for raw in bag.records(topic)]
    if not any(abs(scan_stamp - pose_stamp) <= 1000
               for scan_stamp in bag_stamps for pose_stamp in stamps):
        raise ValueError('Trajectory has no matching front scan timestamps in this bag.')
    if data['frame'] == 'initial_front':
        anchor = lookup(static, BASE, FRONT)
        return lambda time: timeline.at(time), anchor, 'Rebuilt front-LiDAR trajectory, aligned to initial vehicle axes'
    reference = data.get('reference_frame', FRONT)
    reference_from_front = lookup(static, reference, FRONT)

    def front_pose(time):
        pose = timeline.at(time)
        return None if pose is None else pose @ reference_from_front

    return front_pose, Transform.identity(), f'{data["frame"]} | {data.get("source", "trajectory file")}'


def poses_from_tf(bag, static):
    dynamic = {}
    for data in bag.records('/tf'):
        for stamped in deserialize_message(data, TFMessage).transforms:
            t = stamped.transform.translation
            q = stamped.transform.rotation
            key = stamped.header.frame_id, stamped.child_frame_id
            dynamic.setdefault(key, []).append([stamp_ns(stamped.header.stamp), t.x,t.y,t.z,q.x,q.y,q.z,q.w])
    edges = dict(static)
    edges.update({key: Timeline(rows) for key, rows in dynamic.items()
                  if len({row[0] for row in rows}) >= 2})
    for frame in ('calibration_map', 'map', 'odom', 'world'):
        try:
            chain = find_chain(edges, frame, FRONT)
        except ValueError:
            continue
        if any(isinstance(edges[key], Timeline) for key, _ in chain):
            return lambda time: compose_chain(edges, chain, time), Transform.identity(), f'Bag /tf world frame: {frame}'
    raise ValueError('No dynamic map/odom/world -> front TF in this bag. Select a matching trajectory JSON.')


def load_calibration(path):
    data = yaml.safe_load(Path(path).expanduser().read_text())
    if not isinstance(data, dict):
        raise ValueError('Extrinsics YAML must map parent frames to LiDAR transforms.')
    edges = {}
    for parent, children in data.items():
        if not isinstance(children, dict):
            continue
        for child, values in children.items():
            if not isinstance(values, dict) or not {'x','y','z','roll','pitch','yaw'} <= values.keys():
                continue
            xyz = np.array([values[key] for key in ('x','y','z')], dtype=float)
            angles = np.array([values[key] for key in ('roll','pitch','yaw')], dtype=float)
            if not np.isfinite(np.r_[xyz, angles]).all():
                raise ValueError(f'Non-finite extrinsics for {child}.')
            edges[parent, child] = Transform(Rotation.from_euler('xyz', angles), xyz)
    if not edges:
        raise ValueError('No x/y/z and roll/pitch/yaw extrinsics were found in the YAML.')
    children = [child for _, child in edges]
    if len(children) != len(set(children)):
        raise ValueError('An extrinsics frame cannot have multiple parents.')
    return edges


@dataclass
class AccumulationRequest:
    """GUI/命令行与累计业务之间的输入契约，bounds 始终属于公共地图坐标系。"""
    bag: str
    trajectory: str
    bounds: np.ndarray
    extrinsics_source: str = 'bag'
    extrinsics: str = ''
    # Kept for loading old settings.  It is no longer the source of truth.
    calibration_target: str = 'front-left-right'
    calibration_sides: tuple = ()
    # 0 means all scans; a positive value limits each selected radar to its
    # first N scans.  The limit is deliberately per radar because different
    # topics may contain different numbers of messages.
    max_frames: int = 0
    # Static capture is the only supported accumulation mode.  The field is
    # kept in the request so old saved configurations remain readable.
    static_mapping: bool = True

    def active_sides(self):
        if self.calibration_sides:
            return tuple(self.calibration_sides)
        return LEGACY_CALIBRATION_TARGETS.get(
            self.calibration_target, LEGACY_CALIBRATION_TARGETS['front-left-right'])

    def validate(self):
        self.bounds = np.asarray(self.bounds, dtype=float)
        if (self.bounds.shape != (3, 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError('XYZ bounds need finite min < max for each axis.')
        if self.extrinsics_source not in ('bag', 'yaml'):
            raise ValueError('Extrinsics source must be bag or yaml.')
        if self.extrinsics_source == 'yaml' and not self.extrinsics:
            raise ValueError('Select the calibrated extrinsics YAML.')
        sides = tuple(self.active_sides())
        if (len(sides) < 2 or len(sides) != len(set(sides))
                or any(side not in ALL_SIDES for side in sides)):
            raise ValueError('至少选择两个有效雷达参与累计和标定。')
        if 'front' not in sides:
            raise ValueError('参与累计和标定的雷达必须包含前雷达。')
        if (isinstance(self.max_frames, bool)
                or not isinstance(self.max_frames, (int, np.integer))):
            raise ValueError('累计帧数必须是大于等于 0 的整数（0 表示全部帧）。')
        if self.max_frames < 0:
            raise ValueError('累计帧数必须是大于等于 0 的整数（0 表示全部帧）。')
        self.max_frames = int(self.max_frames)
        self.calibration_sides = sides
        if not self.bag:
            raise ValueError('Select the original scan bag.')


@dataclass
class AccumulationResult:
    """后续指标评估的输入接口。

    clouds[name] 是该雷达全部有效帧落入 ROI 的有限 XYZ 点，形状为 (N,3)。
    此处保留原始累计点，不做体素、重叠筛选或 ICP，以免展示影响评估。
    stats 记录每层的数据覆盖情况，缺失轨迹或未扫到目标可据此区分。
    future evaluate(result) 可以独立于查看器运行，并另行生成指标文件。
    """
    request: AccumulationRequest
    clouds: dict
    stats: dict
    frame: str
    provenance: dict = field(default_factory=dict)
    # ROI points grouped by source scan.  This is used by the viewer's frame
    # inspection controls and does not replace the full accumulated clouds.
    frame_clouds: dict = field(default_factory=dict)


def in_bounds(points, bounds):
    return points[np.logical_and(points >= bounds[:, 0], points <= bounds[:, 1]).all(axis=1)]


def thin(points, size):
    """只供展示使用。每个体素保留一个实测点，size=0 表示全部显示。"""
    if not len(points) or size <= 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def sensor_mounts(static, sensor_frames, request):
    """适配外参文件，统一返回 front <- sensor；前雷达轨迹始终固定。"""
    edges = dict(static)
    if request.extrinsics_source == 'yaml':
        supplied = load_calibration(request.extrinsics)
        supplied_children = {child for _, child in supplied}
        # 清除旧的父边，避免一个雷达同时挂在两个父节点导致路径歧义。
        edges = {key: value for key, value in edges.items() if key[1] not in supplied_children}
        edges.update(supplied)
        for side, frame in sensor_frames.items():
            if side != 'front' and frame not in supplied_children:
                raise ValueError(f'Extrinsics YAML has no transform for {side} frame {frame}.')
    return {side: lookup(edges, FRONT, frame) for side, frame in sensor_frames.items()}


def accumulate(request, progress=None, cancelled=None):
    """遍历整段 bag，逐帧变换后立即裁剪，只累计 ROI 内的点。

    progress(text) 和 cancelled() 是可选接口，核心不依赖任何 GUI。
    静态采集模式下所有雷达使用固定的前雷达世界姿态；旧的动态轨迹
    分支仅保留用于读取历史配置。当前变换粒度为整帧；此函数没有逐点
    运动去畸变功能。
    """
    request.validate()
    bag = Bag(request.bag)
    available_topics = bag.lidar_topics()
    required_sides = request.active_sides()
    missing = [side for side in required_sides if side not in available_topics]
    if missing:
        topic_names = ', '.join(
            f'/sensing/lidar/{side}/pointcloud_raw_ex' for side in missing)
        raise ValueError(
            f'当前 bag 缺少标定所需话题：{topic_names}。'
            '请选择同时包含前雷达和目标雷达原始点云的 ROS bag。')
    topics = {side: available_topics[side] for side in required_sides}
    static = static_edges(bag)
    if progress:
        progress('Loading static capture pose and extrinsics...')
    if request.static_mapping:
        # A static capture has no motion compensation and therefore needs no
        # NDT trajectory.  The front LiDAR frame is the world frame; every
        # scan uses the same identity pose while sensor_mounts applies the
        # selected /tf_static or YAML extrinsics.
        world_pose = lambda _time_ns: Transform.identity()
        anchor = Transform.identity()
        frame = 'Static front LiDAR frame'
    elif request.trajectory:
        world_pose, anchor, frame = poses_from_file(request.trajectory, bag, static)
    else:
        world_pose, anchor, frame = poses_from_tf(bag, static)
    sensor_frames = {}
    for side, topic in topics.items():
        first = bag.first_or_last(topic)
        if first is None:
            raise ValueError(f'No scans recorded for {side}.')
        sensor_frames[side] = deserialize_message(first[1], PointCloud2).header.frame_id
    if sensor_frames['front'] != FRONT:
        raise ValueError(f'Front raw scans must be in {FRONT}, got {sensor_frames["front"]}.')
    mounts = sensor_mounts(static, sensor_frames, request)
    clouds, stats, frame_clouds = {}, {}, {}
    for side, topic in topics.items():
        count = dict(total_frames=0, used_frames=0, pose_skips=0, roi_frames=0,
                     finite_input_points=0, roi_points=0)
        batches = []
        per_frame = []
        for frame_index, raw in enumerate(bag.records(topic)):
            if request.max_frames > 0 and frame_index >= request.max_frames:
                break
            if cancelled and cancelled():
                raise InterruptedError('Accumulation cancelled.')
            msg = deserialize_message(raw, PointCloud2)
            if msg.header.frame_id != sensor_frames[side]:
                raise ValueError(f'{side} changes frame_id within this bag.')
            count['total_frames'] += 1
            pose = world_pose(stamp_ns(msg.header.stamp))
            if pose is None:
                count['pose_skips'] += 1
                continue
            count['used_frames'] += 1
            points = read_xyz(msg)
            count['finite_input_points'] += len(points)
            if len(points):
                world = (anchor @ pose @ mounts[side]).apply(points)
                cropped = in_bounds(world, request.bounds).astype(np.float32)
                if len(cropped):
                    batches.append(cropped)
                    count['roi_frames'] += 1
                    count['roi_points'] += len(cropped)
                    per_frame.append(cropped)
                else:
                    per_frame.append(np.empty((0, 3), dtype=np.float32))
            else:
                per_frame.append(np.empty((0, 3), dtype=np.float32))
            if progress and count['total_frames'] % 25 == 0:
                progress(f'{side}: {count["total_frames"]} frames, {count["roi_points"]:,} ROI points')
        clouds[side] = np.concatenate(batches) if batches else np.empty((0, 3), dtype=np.float32)
        stats[side] = count
        frame_clouds[side] = per_frame
    return AccumulationResult(request, clouds, stats, frame, {
        'bag_files': [str(path) for path in bag.files],
        'sensor_frames': sensor_frames,
        'front_from_sensor': {side: dict(translation=mount.translation.tolist(),
                                        quaternion_xyzw=mount.rotation.as_quat().tolist())
                              for side, mount in mounts.items()},
            'trajectory': ('static identity pose' if request.static_mapping else
                           (str(Path(request.trajectory).expanduser().resolve())
                            if request.trajectory else 'bag /tf')),
        'extrinsics_source': request.extrinsics_source,
        'extrinsics': request.extrinsics if request.extrinsics_source == 'yaml' else 'bag initial /tf_static',
        'max_frames_per_radar': request.max_frames,
            'motion_compensation': ('none (static capture)' if request.static_mapping
                                    else 'scan timestamp only'),
    }, frame_clouds)
