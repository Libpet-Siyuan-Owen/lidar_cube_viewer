"""Portable per-report ROI clouds and their measured plane footprints."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
from lidar_cube_core import (AccumulationRequest, Bag, Transform, PointCloud2, deserialize_message,
                             sensor_mounts, static_edges)
from v02_parameters import PARAMS
from lidar_cube_joint import evaluate_request
from lidar_cube_viewer import ViewerSettings


def hull(plane):
    n = np.asarray(plane.normal)
    points = np.asarray(plane.points)
    # Parameters are already represented in the common front frame.
    projected = points - (points @ n + plane.offset)[:, None] * n / (n @ n)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, axis)
    try:
        boundary = ConvexHull(np.column_stack((projected @ u, projected @ np.cross(n, u))))
        vertices = projected[boundary.vertices].tolist()
    except QhullError:
        vertices = []
    return dict(normal=n.tolist(), offset=float(plane.offset), count=len(points), vertices=vertices)


@dataclass
class Scene:
    points: dict
    metadata: dict

    @classmethod
    def capture(cls, clouds, planes):
        fields = clouds.provenance
        return cls(clouds.clouds, dict(frame=clouds.frame, frames=fields['sensor_frames'],
                   bounds=clouds.request.bounds.tolist(), mounts=fields['front_from_sensor'],
                   faces={s: [hull(p) for p in group] for s, group in planes.items()}))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            tmp = Path(file.name)
        try:
            with tmp.open('wb') as file:
                np.savez_compressed(file, metadata=json.dumps(self.metadata), **self.points)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def read(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            return cls({s: data[s] for s in meta['frames']}, meta)


def pack_edges(edges):
    return [[p, c, t.translation.tolist(), t.rotation.as_quat().tolist()] for (p, c), t in edges.items()]


def transform_key(mounts):
    values = []
    for side, mount in sorted(mounts.items()):
        vector = np.r_[mount.translation, mount.rotation.as_matrix().ravel()].round(11)
        values.append([side, np.where(vector == 0, 0., vector).tolist()])
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


class SceneReport:
    def __init__(self, directory):
        from v02_batch import SLOTS, sides_for, valid_bounds
        self.path = Path(directory).resolve()
        self.data = json.loads((self.path/'validation.json').read_text())
        self.final = self.path/'suggested_extrinsics.yaml'
        cases = self.data['cases']
        if (not self.final.is_file() or self.data['failures'] or [c['slot'] for c in cases] != list(SLOTS)
                or len(self.data['validation']) != 6):
            raise ValueError('请选择已成功导出四雷达外参的六包结果目录')
        self.cases = {}
        for case, row in zip(cases, self.data['validation']):
            slot = case['slot']
            if (not case['bag'] or Path(case['bag']).name != case['bag'] or case['bag'] in ('.','..')
                    or len(row['pairs']) != 3 or row['summary']['valid'] != 3 or any(
                    m.get('quality') not in ('PASS','WARNING') or m.get('matched_planes') != 3 for m in row['pairs'].values())):
                raise ValueError(f'{slot} 缺少完整的成功复验记录')
            self.cases[slot] = (case['bag'], valid_bounds(case['bounds']), sides_for(slot))

    def _source_info(self, slot, bag_path, sides):
        info = self.data.get('scene_sources', {}).get(slot)
        if info:
            return info
        # An old report has no transform snapshot. Restore it once from its bag.
        folder = self.path/'clouds_3d'/slot
        path = folder/'source.json'
        if path.is_file():
            return json.loads(path.read_text())
        if not bag_path.is_dir():
            raise FileNotFoundError(f'此旧报告需要原始包 {bag_path} 来恢复外参')
        bag = Bag(bag_path)
        edges = static_edges(bag)
        topics = bag.lidar_topics()
        frames = {s: deserialize_message(bag.first_or_last(topics[s])[1], PointCloud2).header.frame_id for s in sides}
        bounds = self.cases[slot][1]
        info = dict(frames=frames, edges=pack_edges(edges))
        for stage in ('initial', 'joint'):
            req = AccumulationRequest(str(bag_path), '', bounds, calibration_sides=sides,
                        **({'extrinsics_source':'yaml','extrinsics':str(self.final)} if stage == 'joint' else {}))
            info[stage] = transform_key(sensor_mounts(edges, frames, req))
        folder.mkdir(exist_ok=True, parents=True)
        with NamedTemporaryFile(mode='w', dir=folder, delete=False) as stream:
            json.dump(info, stream)
        Path(stream.name).replace(path)
        return info

    def load(self, slot, source, cancelled):
        name, bounds, sides = self.cases[slot]
        folder = self.path/'clouds_3d'/slot
        bag_path = self.path.parent/name
        standard = source in ('initial', 'joint')
        if standard:
            path = folder/(source+'.npz')
            if path.is_file():
                return Scene.read(path)
        yaml_path = self.final if source == 'joint' else None if source == 'initial' else Path(source)
        req = AccumulationRequest(str(bag_path), '', bounds.copy(), calibration_sides=sides,
                                  **({'extrinsics_source':'yaml', 'extrinsics':str(yaml_path)} if yaml_path else {}))
        info = self._source_info(slot, bag_path, sides)
        if not standard:
            edges = {(p,c): Transform(Rotation.from_quat(q), np.array(x)) for p,c,x,q in info['edges']}
            key = transform_key(sensor_mounts(edges, info['frames'], req))
            path = folder/(key+'.npz')
            for stage in ('initial','joint'):
                if key == info[stage]:
                    path = folder/(stage+'.npz')
                    break
            if path.is_file():
                return Scene.read(path)
        if not bag_path.is_dir():
            raise FileNotFoundError(f'此视图尚无缓存，需要原始包 {bag_path}')
        settings = ViewerSettings(req, **{**PARAMS, **self.data['params']})
        clouds, planes, _, _ = evaluate_request(req, settings, lambda _: None, cancelled)
        if cancelled():
            raise InterruptedError()
        scene = Scene.capture(clouds, planes)
        if not standard:
            actual = {s: Transform(Rotation.from_quat(t['quaternion_xyzw']), np.array(t['translation']))
                      for s,t in scene.metadata['mounts'].items()}
            if transform_key(actual) != key:
                raise ValueError('累计时使用的外参与所选快照不一致，请核对原包和 YAML')
        scene.save(path)
        return scene
