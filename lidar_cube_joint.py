"""三摆位共享安装外参：每包一个正交靶标，前雷达固定。"""
from dataclasses import dataclass, replace
from pathlib import Path
from itertools import combinations
from datetime import datetime
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from lidar_cube_core import (AccumulationRequest, Bag, FRONT, KIT, MAPPING_SIDES,
                             Transform, accumulate, lookup, sensor_mounts, static_edges)
from lidar_cube_processing import (FilterConfig, compute_pair_metrics, extract_planes,
                                  filter_points, compute_plane_metrics, summarize_plane_fit)

CONFIG = 'Cube_Viewer_config.yaml'
JOINT_CONFIG = 'joint_fit_config.yaml'
LABEL = {'front': '前', 'left': '左', 'right': '右', 'top': '顶部'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_joint_config(out, result):
    singles = result['stability']['single']
    cases = []
    for i, item in enumerate(result['inputs']):
        case = {key: item[key] for key in ('bag', 'report', 'config_snapshot')}
        if i < len(singles) and singles[i].get('snapshot'):
            case['single_snapshot'] = singles[i]['snapshot']
        cases.append(case)
    config = dict(joint_fit_version=1, validation='validation.json', cases=cases,
                  compare_singles=result['stability']['compare_singles'])
    (out / JOINT_CONFIG).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                   encoding='utf-8')


def load_joint_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('joint_fit_version') != 1:
        raise ValueError('请选择联合拟合配置文件 joint_fit_config.yaml。')
    if config.get('validation') != 'validation.json':
        raise ValueError('配置缺少对应的验证报告 validation.json。')
    result = read_json(path.parent / 'validation.json')
    cases = config['cases']
    inputs = result['inputs']
    if len(cases) != 3 or len(inputs) != 3:
        raise ValueError('联合配置必须包含三个情况。')
    for i, (case, item) in enumerate(zip(cases, inputs), 1):
        if (any(case[key] != item[key] for key in ('bag', 'report', 'config_snapshot'))
                or Path(case['report']).parent != Path(case['bag'])
                or case['config_snapshot'] != f'bag{i}_config.yaml'
                or case.get('single_snapshot') not in (None, f'bag{i}_single_extrinsics.yaml')):
            raise ValueError(f'情况{i}的配置和验证报告不一致。')
    return config, result


def report_directories(bag):
    return sorted((p for p in Path(bag).glob('result_report*') if p.is_dir()), reverse=True)


def xyz_record(transform):
    values = np.r_[transform.translation, transform.rotation.as_euler('xyz')]
    return dict(zip(('x', 'y', 'z', 'roll', 'pitch', 'yaw'), values.tolist()))


def old_mount(metrics, side):
    values = metrics.get('extrinsics', {}).get('front-' + side, {}).get('old_pair')
    if values is None:
        raise ValueError(f'缺少累计时的前–{LABEL[side]}相对外参，请重新保存单包拟合结果')
    return Transform(Rotation.from_quat(values['quaternion_xyzw']),
                     np.array([values[k] for k in 'xyz']))


@dataclass
class Capture:
    bag: Path
    report: Path
    config_text: str
    settings: object
    sides: tuple
    frames: dict
    baseline: dict
    surfaces: list
    metadata: dict


def correspondence(metrics, a, b):
    pairs = metrics['pairs']
    reverse = f'{a}-{b}' not in pairs
    entry = pairs.get(f'{b}-{a}' if reverse else f'{a}-{b}', {})
    mapping = {}
    for match in entry.get('matches', []):
        i, j = match['plane_a_index'] - 1, match['plane_b_index'] - 1
        mapping[j if reverse else i] = i if reverse else j
    if set(mapping) != {0, 1, 2} or set(mapping.values()) != {0, 1, 2}:
        raise ValueError(f'{LABEL[a]}–{LABEL[b]}缺少三个面的一一对应')
    return [mapping[i] for i in range(3)]


def prepare(bag_path, report_path, load_settings, config_path=None):
    bag = bag_path.resolve()
    report = report_path.resolve()
    if report.parent != bag or report not in report_directories(bag):
        raise ValueError('请选择本包下保存的单包报告')
    config_path = config_path or bag / CONFIG
    config_text = config_path.read_text(encoding='utf-8')
    settings = load_settings(config_path)
    if config_text != config_path.read_text(encoding='utf-8'):
        raise ValueError('读取期间配置发生变化，请重试')
    manifest = read_json(report / 'validation.json')
    metrics = read_json(report / 'metrics_report.json')
    for file in (manifest['bag'], settings.request.bag):
        if Bag(file).directory.resolve() != bag:
            raise ValueError('报告或配置指向另一个 bag')
    if manifest['provenance'].get('motion_compensation') != 'none (static capture)':
        raise ValueError('请选择静态累计报告')
    sides = tuple(s for s in MAPPING_SIDES if s in manifest['calibration_sides'])
    if len(sides) != 3 or sides[0] != 'front' or set(settings.request.active_sides()) != set(sides):
        raise ValueError('报告与配置必须选择相同的三台雷达，并包含前雷达')
    frames = {s: manifest['provenance']['sensor_frames'][s] for s in sides}
    if frames['front'] != FRONT or len(set(frames.values())) != 3:
        raise ValueError('传感器 frame 不一致')
    edges = static_edges(Bag(bag))
    baseline = {s: lookup(edges, KIT, frames[s]) for s in sides}
    orders = [[0, 1, 2]] + [correspondence(metrics, 'front', s) for s in sides[1:]]
    between = correspondence(metrics, sides[1], sides[2])
    if [between[i] for i in orders[1]] != orders[2]:
        raise ValueError('三个雷达的逐面对应不一致')
    patches = []
    for side, order in zip(sides, orders):
        entries = manifest.get('planes', {}).get(side, [])
        if len(entries) != 3 or metrics.get('sensor_summary', {}).get(side, {}).get('quality') == 'FAIL':
            raise ValueError(f'{LABEL[side]}雷达拟合了 {len(entries)} 个面，需要三个合格的面')
        # 单包 ROI 中的点原本在 front 中。先撤销当时使用的外参，再统一放入
        # /tf_static 初始前雷达系。历史 YAML 即使被覆盖也无需重新读取。
        base_mount = baseline['front'].inverse() @ baseline[side]
        if side == 'front' or manifest['extrinsics_source'] == 'bag':
            change = Transform.identity()
        elif side in manifest['provenance'].get('front_from_sensor', {}):
            value = manifest['provenance']['front_from_sensor'][side]
            recorded = Transform(Rotation.from_quat(value['quaternion_xyzw']), np.array(value['translation']))
            change = base_mount @ recorded.inverse()
        else:
            change = base_mount @ old_mount(metrics, side).inverse()
        sensor = []
        for i in order:
            entry = entries[i]
            filename = report / entry['points_file']
            points = np.load(filename, allow_pickle=False)
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or
                    len(points) != entry['inlier_count'] or not np.isfinite(points).all()):
                raise ValueError(f'{LABEL[side]}雷达面 {i + 1} 的内点无效：{filename.name}')
            sensor.append(change.apply(points))
        patches.append(sensor)
    return Capture(bag, report, config_text, settings, sides, frames, baseline, patches, manifest)


def seed_board(front_faces):
    normals, offsets = [], []
    for pts in front_faces:
        center = pts.mean(axis=0)
        _, singular, axes = np.linalg.svd(pts - center, full_matrices=False)
        if singular[1] < 1e-9:
            raise ValueError('前雷达平面点近共线，不能确定靶面方向')
        normals.append(axes[-1])
        offsets.append(axes[-1] @ center)
    corner = np.linalg.solve(normals, offsets)
    u, _, vt = np.linalg.svd(normals)
    return u @ vt, corner


def fit_joint(captures, cancelled, sides=None):
    """每个摆位有九个点集，在对应靶面上最小化点面距离。每个面等权。

    前雷达固定，各个非前雷达共用其物理雷达对应的六个未知量，
    每个靶标另有六个未知量。各摆位共用同一组安装外参。
    """
    sides = tuple(sides or captures[0].sides)
    radar_cols = 6 * (len(sides) - 1)
    poses = [Transform.identity() for _ in sides]
    boards = [seed_board(c.surfaces[0]) for c in captures]

    n_parameters = radar_cols + 6 * len(captures)

    def linearize(poses, boards, derivative):
        cost = 0.
        gram = np.zeros((n_parameters, n_parameters))
        gradient = np.zeros(n_parameters)
        for k, cap in enumerate(captures):
            normals, center = boards[k]
            target_cols = list(range(radar_cols + 6*k, radar_cols + 6*(k+1)))
            for side, faces in zip(cap.sides, cap.surfaces):
                s = sides.index(side)
                for n, pts in zip(normals, faces):
                    q = poses[s].apply(pts)
                    distances = (q - center) @ n
                    robust = np.hypot(1, distances / .015)
                    cost += np.mean(2 * distances**2 / (robust + 1))
                    if not derivative:
                        continue
                    axis = np.broadcast_to(n, q.shape)
                    jac = np.column_stack((np.cross(n, q - center), -axis))
                    cols = target_cols
                    if s:
                        jac = np.column_stack((np.cross(q, n), axis, jac))
                        cols = list(range(6*(s-1), 6*s)) + target_cols
                    weighted = jac / (len(q) * robust[:, None])
                    gram[np.ix_(cols, cols)] += weighted.T @ jac
                    gradient[cols] += weighted.T @ distances
        return cost, gram, gradient

    for _ in range(80):
        if cancelled():
            raise InterruptedError('联合拟合已停止')
        cost, gram, gradient = linearize(poses, boards, True)
        if np.linalg.matrix_rank(gram) != len(gram):
            raise ValueError('所选摆位的几何约束不足，无法确定外参')
        step = np.linalg.solve(gram, -gradient)
        if abs(step).max() < 1e-8:
            return poses
        for scale in (1., .5, .25, .125, .0625, .03125):
            increments = (scale * step).reshape(-1, 6)
            trial = [poses[0]]
            for pose, dx in zip(poses[1:], increments[:len(sides)-1]):
                trial.append(Transform(Rotation.from_rotvec(dx[:3]), dx[3:]) @ pose)
            targets = [(Rotation.from_rotvec(dx[:3]).apply(axes), center + dx[3:])
                       for (axes, center), dx in zip(boards, increments[len(sides)-1:])]
            updated, _, _ = linearize(trial, targets, False)
            if updated <= cost:
                break
        else:
            if abs(gradient).max() < 1e-6:
                return poses
            raise ValueError('联合拟合未收敛，请检查靶面对应和覆盖范围')
        poses, boards = trial, targets
        if cost - updated < 1e-13:
            return poses
    raise ValueError('联合拟合达到迭代上限')


def aggregate(pairs, expected):
    valid = [p for p in pairs if p.get('matched_planes') == 3
             and p.get('corner_distance_m') is not None and p.get('normal_error_deg') is not None]
    return dict(valid=len(valid), expected=expected,
                corner_cm=float(np.mean([100*p['corner_distance_m'] for p in valid])) if valid else None,
                normal_deg=float(np.mean([p['normal_error_deg'] for p in valid])) if valid else None)


def replay(cap, extrinsics, progress, cancelled, on_geometry=None):
    settings = cap.settings
    req = AccumulationRequest(str(cap.bag), '', settings.request.bounds.copy(),
                              extrinsics_source='yaml', extrinsics=str(extrinsics),
                              calibration_sides=cap.sides, max_frames=settings.request.max_frames)
    clouds, planes, sensor_stats, pairs = evaluate_request(req, settings, progress, cancelled)
    if on_geometry is not None:
        on_geometry(clouds, planes)
    return dict(pairs=pairs, sensors=sensor_stats, frames=clouds.stats,
                input_extrinsics=Path(extrinsics).name, summary=aggregate(pairs.values(), 3))


def evaluate_request(req, settings, progress, cancelled):
    progress('累计 ROI 点云')
    clouds = accumulate(req, progress=progress, cancelled=cancelled)
    fc = FilterConfig(voxel_size=settings.analysis_voxel, neighbor_radius=settings.filter_radius,
                      min_neighbors=settings.filter_min_neighbors)
    planes, sensor_stats = {}, {}
    for side in req.active_sides():
        if cancelled():
            raise InterruptedError('回代验证已停止')
        progress(f'{LABEL[side]}雷达滤波与提面')
        points = filter_points(clouds.clouds[side], fc)
        fitted = []
        if len(points) >= max(settings.plane_min_inliers, 3):
            fitted, _ = extract_planes(points, distance_threshold=settings.plane_threshold,
                                       min_inliers=settings.plane_min_inliers,
                                       max_planes=settings.plane_max_planes,
                                       max_iterations=settings.plane_max_iterations,
                                       min_plane_angle_deg=settings.plane_min_angle,
                                       min_plane_span=settings.plane_min_span,
                                       max_plane_span=settings.plane_max_span, random_seed=0)
        planes[side] = fitted
        entries = compute_plane_metrics(points, fitted)
        sensor_stats[side] = dict(summary=summarize_plane_fit(side, entries), faces=entries,
                                  filtered_points=len(points))
    pairs = {}
    for a, b in combinations(req.active_sides(), 2):
        key = f'{a}-{b}'
        metric = compute_pair_metrics(planes[a], planes[b])
        invalid = [f'{LABEL[s]}雷达：{sensor_stats[s]["summary"]["reason"]}' for s in (a, b)
                   if sensor_stats[s]['summary']['quality'] == 'FAIL']
        if invalid:
            metric = dict(matched_planes=0, quality='FAIL', reason='；'.join(invalid),
                          measured=metric)
        pairs[key] = metric
    return clouds, planes, sensor_stats, pairs


def replay_spread(rows):
    """仅当三个摆位都有三组可测的雷达配对时，才比较三个包均值的极差。"""
    if len(rows) != 3 or any(row['summary']['valid'] != 3 for row in rows):
        return {}
    return {key: float(np.ptp([row['summary'][key] for row in rows]))
            for key in ('corner_cm', 'normal_deg')}


def corner_comparison(inputs, single_results, joint_bags):
    """按选择顺序对齐三种情况。任一情况缺少完整测量时，保留空值。"""
    def row(label, bags):
        summaries = {item.get('bag_path', item['bag']): item['summary'] for item in bags}
        distances = []
        for case in inputs:
            summary = summaries.get(case['bag'], summaries.get(Path(case['bag']).name, {}))
            value = summary.get('corner_cm') if summary.get('valid') == 3 else None
            distances.append(value)
        worst = max(distances) if len(distances) == 3 and None not in distances else None
        return dict(label=label, cases_cm=distances, worst_cm=worst)

    rows = [row(f'情况{i} 单包外参 {letter}', single['bags'])
            for i, (letter, single) in enumerate(zip('ABC', single_results), 1)]
    rows.append(row('三包联合外参 N', joint_bags))
    return rows


def placement_text(comparison):
    if not comparison.get('finished'):
        return '跨摆位回代尚未完成。'
    if comparison['errors']:
        return '跨摆位稳定性对比不完整。' + '；'.join(comparison['errors'])
    joint = comparison['joint']['spread']
    if not comparison['compare_singles']:
        return f"固定 N 的跨包极差：交点距离 {joint['corner_cm']:.2f} cm，法向夹角 {joint['normal_deg']:.2f}°。"
    single = comparison['single_mean_spread']
    # 与界面保留两位小数的显示精度一致。
    lower = [round(joint[k], 2) < round(single[k], 2) for k in ('corner_cm', 'normal_deg')]
    verdict = {(True, True): '联合解在这三个摆位上的两项误差波动都更小。',
               (True, False): '联合解的交点距离波动更小，法向波动未减小。',
               (False, True): '联合解的法向波动更小，交点距离波动未减小。',
               (False, False): '这三个摆位上，两项指标均未显示联合解的波动减小。'}[tuple(lower)]
    return (f"交点距离波动：单包平均 {single['corner_cm']:.2f} cm → 联合 {joint['corner_cm']:.2f} cm。\n"
            f"法向夹角波动：单包平均 {single['normal_deg']:.2f}° → 联合 {joint['normal_deg']:.2f}°。\n" + verdict)


def run_joint(selections, load_settings, progress, cancelled=lambda: False, *, output_parent=None, compare_singles=False, snapshots=None):
    """Worker entry point. No Tk calls. Return the output directory and summary."""
    if len(selections) != 3 or len({Path(b).resolve() for b, _ in selections}) != 3:
        raise ValueError('请选择三个不同的数据包')
    parent = Path(output_parent or selections[0][0]).expanduser().resolve()
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f'结果保存目录不存在或不可写，请重新选择：{parent}')
    captures = []
    snapshots = snapshots or [(None, None)] * len(selections)
    for (bag, report), (config_snapshot, _) in zip(selections, snapshots):
        bag = Path(bag)
        progress(f'检查 {bag.name} 的配置和单包报告')
        try:
            cap = prepare(bag, Path(report), load_settings, config_snapshot)
            if captures:
                first = captures[0]
                if cap.frames != first.frames or cap.sides != first.sides:
                    raise ValueError('三包选择的雷达或 frame 不一致')
                for s in cap.sides:
                    diff = first.baseline[s].inverse() @ cap.baseline[s]
                    if np.linalg.norm(diff.translation) > 1e-6 or diff.rotation.magnitude() > 1e-6:
                        raise ValueError(f'{LABEL[s]}雷达的原始 /tf_static 与第一包不同，无法共用安装基准')
            captures.append(cap)
        except Exception as exc:
            raise ValueError(f'{bag.name} / {Path(report).name}：{exc}') from exc
    first = captures[0]
    progress('正在联合拟合三个标定包的三面体')
    poses = fit_joint(captures, cancelled)
    sensor_code = ''.join(side[0].upper() for side in first.sides)
    out = parent / f'{sensor_code}_result_report_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir()
    extrinsics, mounts = {}, {}
    anchor = first.baseline['front']
    for side, correction in zip(first.sides, poses):
        old = first.baseline[side]
        new = anchor @ correction @ anchor.inverse() @ old if side != 'front' else old
        extrinsics[first.frames[side]] = xyz_record(new)
        delta = (new.translation - old.translation) * 100
        mounts[side] = dict(frame=first.frames[side], old=xyz_record(old), new=xyz_record(new),
                            delta_xyz_cm=delta.tolist(), displacement_cm=float(np.linalg.norm(delta)),
                            rotation_deg=float(np.rad2deg((old.rotation.inv()*new.rotation).magnitude())))
    yaml_path = out / 'suggested_extrinsics.yaml'
    yaml_path.write_text(yaml.safe_dump({KIT: extrinsics}, sort_keys=False), encoding='utf-8')
    result = dict(reference='front', baseline='/tf_static', validation='固定联合外参 N，在三个原始包重新累计、拟合平面并匹配',
                  validation_extrinsics=yaml_path.name,
                  method='正交三面体等权点面拟合，soft-L1 15 mm', mounts=mounts, inputs=[], bags=[])
    for i, cap in enumerate(captures, 1):
        name = f'bag{i}_config.yaml'
        (out / name).write_text(cap.config_text, encoding='utf-8')
        result['inputs'].append(dict(bag=str(cap.bag), report=str(cap.report), config_snapshot=name,
                                     fit_roi=cap.metadata['bounds_xyz'], replay_roi=cap.settings.request.bounds.tolist(),
                                     fit_extrinsics_source=cap.metadata['extrinsics_source'],
                                     fit_extrinsics=cap.metadata.get('extrinsics', '')))
    comparison = dict(protocol='raw_bag_replay_v1',
                      method='每套外参分别对三个原始 bag 重新累计 ROI、滤波、提面和匹配，取三个包平均误差的极差。',
                      joint={}, single=[], errors=[], finished=False, compare_singles=compare_singles)
    result['stability'] = comparison
    if compare_singles:
        progress('读取三份单包外参，准备重新累计')
        for i, cap in enumerate(captures, 1):
            source = snapshots[i-1][1] or cap.report / 'suggested_extrinsics.yaml'
            snapshot = out / f'bag{i}_single_extrinsics.yaml'
            single = dict(source=str(source), bags=[], spread={})
            comparison['single'].append(single)
            try:
                snapshot.write_bytes(source.read_bytes())
                request = replace(cap.settings.request, extrinsics_source='yaml', extrinsics=str(snapshot))
                sensor_mounts(static_edges(Bag(cap.bag)), cap.frames, request)
                single['snapshot'] = snapshot.name
            except (OSError, ValueError, TypeError, OverflowError, yaml.YAMLError) as exc:
                comparison['errors'].append(f'包 {i} 单包外参不可用：{exc}')

    def persist():
        all_pairs = [p for row in result['bags'] for p in row['pairs'].values()]
        result['overall'] = aggregate(all_pairs, 9)
        result['corner_comparison'] = corner_comparison(result['inputs'], comparison['single'], result['bags'])
        result['status'] = ('验证进行中' if not comparison['finished'] else
                            '验证不完整' if comparison['errors'] else '验证完成')
        comparison['summary'] = placement_text(comparison)
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        pending = out / 'validation.json.tmp'
        pending.write_text(text, encoding='utf-8')
        pending.replace(out / 'validation.json')
        with (out / 'metrics.log').open('w', encoding='utf-8') as log:
            print('三标定包联合拟合 · 前雷达锁定 · 安装修正相对 /tf_static', file=log)
            print(result['stability']['method'], result['stability']['summary'], sep='\n', file=log)
            for cap in captures:
                print(f'输入 {cap.bag}  单包报告 {cap.report.name}', file=log)
            for row in result['bags']:
                print(f"\n{row['bag']}  {row.get('error', '')}", file=log)
                for pair, m in row['pairs'].items():
                    print(pair, m['quality'], m.get('reason', ''), file=log)
                    for face in m.get('matches', []):
                        print('  P%d ↔ P%d  法向 %.3f°  点面 RMS %.3f cm  P95 %.3f cm' %
                              (face['plane_a_index'],face['plane_b_index'],face['normal_error_deg'],
                               100*face['point_to_plane_rms_m'],100*face['point_to_plane_p95_m']), file=log)
            print('\n完整指标与外参\n' + text, file=log)
    persist()
    write_joint_config(out, result)

    def replay_three(path, label, rows):
        for i, cap in enumerate(captures, 1):
            if cancelled():
                raise InterruptedError('回代验证已停止')
            prefix = f'{label} · 包 {i}/3 · {cap.bag.name} · '
            try:
                data = replay(cap, path, lambda m: progress(prefix + m), cancelled)
            except InterruptedError:
                raise
            except Exception as exc:
                data = dict(pairs={}, summary=aggregate([], 3), error=str(exc))
            rows.append(dict(bag=cap.bag.name, bag_path=str(cap.bag), **data))
            if data['summary']['valid'] != 3:
                comparison['errors'].append(f"{label}在包 {i} 有效配对 {data['summary']['valid']}/3")

    try:
        replay_three(yaml_path, '联合解', result['bags'])
        comparison['joint']['spread'] = replay_spread(result['bags'])
        persist()
        for i, single in enumerate(comparison['single'], 1):
            if 'snapshot' not in single:
                continue
            replay_three(out / single['snapshot'], f'单包解 {i}/3', single['bags'])
            single['spread'] = replay_spread(single['bags'])
            persist()
        if compare_singles and all(single['spread'] for single in comparison['single']):
            comparison['single_mean_spread'] = {
                key: float(np.mean([single['spread'][key] for single in comparison['single']]))
                for key in ('corner_cm', 'normal_deg')}
    except InterruptedError:
        comparison['errors'].append('回代验证已停止')
    comparison['finished'] = True
    persist()
    return out, result
