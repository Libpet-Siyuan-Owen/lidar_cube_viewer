"""FLT 与 FRT 共用同一套整车外参，按各包实际观测的三台雷达回代。"""
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from lidar_cube_core import KIT, MAPPING_SIDES
from lidar_cube_joint import (JOINT_CONFIG, aggregate, fit_joint, load_joint_config,
                              prepare, read_json, replay, xyz_record)

SIX_CONFIG = 'six_fit_config.yaml'
GROUPS = (('front', 'left', 'top'), ('front', 'right', 'top'))
LABELS = ('顶部取 FLT', '顶部取 FRT', '顶部取中间解', '六包联合外参 N')


def read_group(path, sides):
    config, report = load_joint_config(path)
    if set(report['mounts']) != set(sides):
        raise ValueError('所选三包配置中的雷达组合不符，请按 FLT、FRT 分别选择。')
    return config['cases'], report['mounts']


def midpoint(a, b):
    angles = ('roll', 'pitch', 'yaw')
    first, second = [Rotation.from_euler('xyz', [v[k] for k in angles]) for v in (a, b)]
    rotation = first * Rotation.from_rotvec((first.inv() * second).as_rotvec() / 2)
    return {**{k: (a[k]+b[k])/2 for k in 'xyz'}, **dict(zip(angles, rotation.as_euler('xyz').tolist()))}


def load_six(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('six_fit_version') != 1:
        raise ValueError('请选择六包联合配置 six_fit_config.yaml。')
    result = read_json(path.parent / 'validation.json')
    if len(config['group_configs']) != 2 or config['group_configs'] != result['group_configs']:
        raise ValueError('六包配置和报告不一致。')
    return config, result


def run_six(paths, load_settings, progress, cancelled, output_parent=None):
    paths = [Path(path).resolve() for path in paths]
    if len(paths) != 2 or paths[0] == paths[1]:
        raise ValueError('请选择一份 FLT 配置和一份 FRT 配置。')
    captures, group_mounts, inputs = [], [], []
    frames, baseline = {}, {}
    for path, sides in zip(paths, GROUPS):
        cases, mounts = read_group(path, sides)
        group_mounts.append(mounts)
        for case in cases:
            if cancelled():
                raise InterruptedError('六包联合拟合已停止')
            bag = Path(case['bag'])
            progress(f'检查 {bag.name} 的 ROI 和单包平面')
            cap = prepare(bag, Path(case['report']), load_settings, path.parent / case['config_snapshot'])
            if cap.sides != sides:
                raise ValueError(f'{bag.name} 的实际雷达组合与所选配置不符。')
            for side in sides:
                if cap.frames[side] != mounts[side]['frame']:
                    raise ValueError(f'{bag.name} 与三包结果的雷达 frame 不一致。')
                old = cap.baseline[side]
                if side in baseline:
                    diff = baseline[side].inverse() @ old
                    if (frames[side] != cap.frames[side] or np.linalg.norm(diff.translation) > 1e-6
                            or diff.rotation.magnitude() > 1e-6):
                        raise ValueError(f'{bag.name} 的 {side} 安装基准或 frame 不一致。')
                frames[side], baseline[side] = cap.frames[side], old
            captures.append(cap)
            inputs.append(dict(bag=str(cap.bag), report=str(cap.report), sides=sides,
                               config_snapshot=f'bag{len(captures)}_config.yaml'))
    bags = {c.bag for c in captures}
    if len(bags) != 6 or len({b.parent for b in bags}) != 1 or len(set(frames.values())) != 4:
        raise ValueError('需要同一台机器目录下六个不同的标定包和四个不同的雷达 frame。')
    parent = Path(output_parent or captures[0].bag.parent).expanduser().resolve()
    if not parent.is_dir():
        raise ValueError(f'结果保存目录不存在：{parent}')
    progress('六包联合求解中，锁定前雷达，共同求解左、右和顶部安装外参')
    poses = fit_joint(captures, cancelled, MAPPING_SIDES)
    front = baseline['front']
    mounts, joint = {}, {}
    for side, correction in zip(MAPPING_SIDES, poses):
        old = baseline[side]
        new = old if side == 'front' else front @ correction @ front.inverse() @ old
        delta = 100*(new.translation - old.translation)
        joint[frames[side]] = xyz_record(new)
        mounts[side] = dict(frame=frames[side], old=xyz_record(old), new=xyz_record(new),
                            delta_xyz_cm=delta.tolist(), displacement_cm=float(np.linalg.norm(delta)),
                            rotation_deg=float(np.rad2deg((old.rotation.inv()*new.rotation).magnitude())))
    left, right = group_mounts
    fixed = {frames['front']: joint[frames['front']], frames['left']: left['left']['new'],
             frames['right']: right['right']['new']}
    a, b = left['top']['new'], right['top']['new']
    candidates = [{**fixed, frames['top']: t} for t in (a, b, midpoint(a, b))] + [joint]
    group_paths = [str(p) for p in paths]
    result = dict(group_configs=group_paths, reference='front', baseline='/tf_static', inputs=inputs,
                  method='六摆位独立正交三面体，共享左、右、顶安装外参；逐面等权 soft-L1 15 mm',
                  mounts=mounts, corner_comparison=[], candidates=[], errors=[])
    out = parent / f'FLRT_result_report_{datetime.now():%Y%m%d_%H%M%S_%f}'
    with TemporaryDirectory(prefix='.six_fit_', dir=parent) as temp:
        stage = Path(temp)/'report'
        stage.mkdir()
        for cap, case in zip(captures, inputs):
            (stage/case['config_snapshot']).write_text(cap.config_text, encoding='utf-8')
        for i, (label, params) in enumerate(zip(LABELS, candidates), 1):
            file = stage/f'candidate_{i}.yaml'
            file.write_text(yaml.safe_dump({KIT: params}, sort_keys=False), encoding='utf-8')
            values, rows = [], []
            for k, cap in enumerate(captures, 1):
                if cancelled():
                    raise InterruptedError('六包联合拟合已停止')
                prefix = f'{label}（{i}/4）· 情况{k}/6 · '
                try:
                    row = replay(cap, file, lambda m: progress(prefix + m), cancelled)
                except InterruptedError:
                    raise
                except Exception as exc:
                    row = dict(pairs={}, summary=aggregate([], 3), error=str(exc))
                row['bag'] = str(cap.bag)
                rows.append(row)
                valid = row['summary']['valid'] == 3
                values.append(row['summary']['corner_cm'] if valid else None)
                if not valid:
                    result['errors'].append(prefix + row.get('error', '有效配对不足 3 组'))
            result['corner_comparison'].append(dict(label=label, cases_cm=values,
                                         worst_cm=max(values) if None not in values else None))
            result['candidates'].append(dict(label=label, extrinsics=file.name, bags=rows,
                                        overall=aggregate([p for r in rows for p in r['pairs'].values()],18)))
        passed = all(len(r['pairs']) == 3 and all(m.get('quality') in ('PASS','WARNING')
                     for m in r['pairs'].values()) and r['summary']['valid']==3 for r in rows)
        if passed:
            (stage/'suggested_extrinsics.yaml').write_bytes(file.read_bytes())
        else:
            result['errors'].append('六包联合解未通过最终复验，候选外参保存在 candidate_4.yaml。')
        result['exported'] = passed
        result['status'] = '验证完成' if not result['errors'] else '验证不完整'
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        (stage/'validation.json').write_text(text, encoding='utf-8')
        (stage/'metrics.log').write_text('六包四雷达联合解\n'+text, encoding='utf-8')
        (stage/SIX_CONFIG).write_text(yaml.safe_dump(dict(six_fit_version=1, group_configs=group_paths),
                                      sort_keys=False, allow_unicode=True), encoding='utf-8')
        stage.rename(out)
    return out, result
