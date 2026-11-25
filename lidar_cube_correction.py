"""单包外参修正与三雷达闭环复验，供查看器和三包回代共用。"""
import numpy as np
from scipy.spatial.transform import Rotation
from lidar_cube_core import Bag, FRONT, KIT, Transform, lookup, sensor_mounts, static_edges
from lidar_cube_processing import optimize_three_radar_extrinsics, validate_transformed_pair


def _transform_report(transform, parent_frame, child_frame):
    """把内部 Transform 转成界面和 YAML 报告都能使用的标量。"""
    rpy = transform.rotation.as_euler('xyz')
    return {
        'parent_frame': parent_frame,
        'child_frame': child_frame,
        'x': float(transform.translation[0]),
        'y': float(transform.translation[1]),
        'z': float(transform.translation[2]),
        'roll': float(rpy[0]),
        'pitch': float(rpy[1]),
        'yaw': float(rpy[2]),
        'roll_deg': float(np.degrees(rpy[0])),
        'pitch_deg': float(np.degrees(rpy[1])),
        'yaw_deg': float(np.degrees(rpy[2])),
        'quaternion_xyzw': transform.rotation.as_quat().tolist(),
    }

def attach_extrinsic_report(result, planes, metrics, pair_weights=None):
    """Attach pairwise corrections and refine a selected three-radar loop.

    Pairwise plane fitting remains the source of the three observations.
    When exactly three radars are selected, the front-left and front-top
    transforms initialize the solution and the third pair is used as a
    soft loop-closure constraint.  The optimized reference-relative poses
    are then converted back to sensor_kit_base_link using the unchanged
    front-radar anchor.
    """
    metrics.pop('extrinsics', None)
    metrics.pop('three_radar_optimization', None)
    if result is None or not metrics:
        return
    corrections = metrics.get('extrinsic_corrections', {})
    if not corrections:
        return
    try:
        provenance = result.provenance or {}
        sensor_frames = provenance['sensor_frames']
        bag = Bag(result.request.bag)
        static = static_edges(bag)
        recorded = provenance.get('front_from_sensor', {})
        mounts = ({side: Transform(Rotation.from_quat(value['quaternion_xyzw']),
                                   np.asarray(value['translation'])) for side, value in recorded.items()}
                  if recorded else sensor_mounts(static, sensor_frames, result.request))
        front_frame = sensor_frames.get('front', FRONT)
        kit_from_front = lookup(static, KIT, FRONT)
        front_kit = _transform_report(
            kit_from_front, KIT, front_frame)
    except Exception as error:
        metrics['extrinsics'] = {
            'error': f'无法读取旧外参：{error}',
        }
        return

    def transform_matrix(transform):
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = transform.rotation.as_matrix()
        matrix[:3, 3] = np.asarray(transform.translation, dtype=float)
        return matrix

    def transform_from_matrix(matrix):
        matrix = np.asarray(matrix, dtype=float)
        return Transform(
            Rotation.from_matrix(matrix[:3, :3]),
            matrix[:3, 3].copy())

    report = {}
    observed_pairs = {}
    for pair, correction in corrections.items():
        first, second = pair.split('-', 1)
        entry = {
            'reference_frame': sensor_frames.get(first, first),
            'source_frame': sensor_frames.get(second, second),
            'correction': correction,
        }
        if not correction.get('available'):
            entry.update({
                'available': False,
                'eligible': correction.get('eligible', False),
                'reason': correction.get('reason', '未知原因'),
            })
            report[pair] = entry
            continue
        try:
            rotation = Rotation.from_matrix(np.asarray(
                correction['rotation_matrix'], dtype=float))
            delta = Transform(rotation, np.asarray(
                correction['translation_xyz_m'], dtype=float))
            # mounts[side] is FRONT <- sensor. The correction delta is
            # estimated in the common front/world frame, so it cannot be
            # applied as delta @ old_pair for every pair. That shortcut
            # is valid only when reference=front. For left-top, convert
            # the world correction back to the relative sensor frame.
            old_pair = mounts[first].inverse() @ mounts[second]
            new_pair = mounts[first].inverse() @ delta @ mounts[second]
            observed_pairs[pair] = transform_matrix(new_pair)
            first_frame = sensor_frames[first]
            second_frame = sensor_frames[second]
            entry.update({
                'available': True,
                'eligible': correction.get('eligible', True),
                'old_pair': _transform_report(
                    old_pair, first_frame, second_frame),
                'new_pair': _transform_report(
                    new_pair, first_frame, second_frame),
            })
            if first == 'front':
                # 前雷达参考时，pair 变换就是前 <- source；同时给出
                # sensor_kit_base_link 下可直接写回 YAML 的建议值。
                old_mount = mounts[second]
                new_mount = new_pair
                old_report = _transform_report(old_mount, FRONT, second_frame)
                new_report = _transform_report(new_mount, FRONT, second_frame)
                old_kit = _transform_report(
                    kit_from_front @ old_mount, KIT, second_frame)
                new_kit = _transform_report(
                    kit_from_front @ new_mount, KIT, second_frame)
                entry.update({
                    'child_frame': second_frame,
                    'old': old_report,
                    'new': new_report,
                    'old_sensor_kit': old_kit,
                    'new_sensor_kit': new_kit,
                })
            report[pair] = entry
        except Exception as error:
            entry.update({
                'available': False,
                'eligible': correction.get('eligible', True),
                'reason': f'外参修正转换失败：{error}',
            })
            report[pair] = entry

    selected = tuple(result.request.active_sides())
    if len(selected) == 3 and 'front' in selected:
        optimization = optimize_three_radar_extrinsics(observed_pairs, selected, pair_weights=pair_weights)
        # relative_final 是 first <- second。还原成初始公共坐标中的修正，
        # 才能用原来累计的内点复验同一组对应面，尤其是非前雷达配对。
        final_scores = {}
        if optimization.get('available'):
            for key in corrections:
                first, second = key.split('-')
                matrices = optimization['optimized_pairs']
                relative = (np.asarray(matrices[key]) if key in matrices else
                            np.linalg.inv(matrices[f'{second}-{first}']))
                delta = mounts[first] @ transform_from_matrix(relative) @ mounts[second].inverse()
                final_scores[key] = validate_transformed_pair(
                    planes[first], planes[second],
                    metrics['pairs'][key]['matches'],
                    delta.rotation.as_matrix(), delta.translation,
                    metrics['thresholds']['overlap_distance_m'])
            failed = [f"{key} P{m['plane_a_index']}↔P{m['plane_b_index']}"
                      for key, score in final_scores.items() for m in score['matches']
                      if m['quality'] == 'FAIL']
            if failed:
                optimization.update(available=False, reason='联合复验超限：' + '、'.join(failed))
        optimization['solver_converged'] = optimization.get('success', False)
        optimization['success'] = optimization['available']
        optimization['final_validation'] = final_scores
        optimization['quality'] = ('FAIL' if not optimization.get('available') else
                                   'WARNING' if any(s['quality'] == 'WARNING' for s in final_scores.values()) else 'PASS')
        metrics['three_radar_optimization'] = optimization
        if optimization.get('available'):
            # 将优化后的三组相对外参加回报告。前-非前配对还会
            # 重新生成 KIT 外参；非前配对只作为闭环后的相对结果。
            for optimized_pair, matrix_list in optimization.get(
                    'optimized_pairs', {}).items():
                matrix = np.asarray(matrix_list, dtype=float)
                report_key = optimized_pair
                oriented = matrix
                if report_key not in report:
                    first_name, second_name = optimized_pair.split('-', 1)
                    reverse_key = f'{second_name}-{first_name}'
                    if reverse_key in report:
                        report_key = reverse_key
                        oriented = np.linalg.inv(matrix)
                entry = report.get(report_key)
                if entry is None or not entry.get('available'):
                    continue
                first_name, second_name = report_key.split('-', 1)
                entry['observed_new_pair'] = entry.get('new_pair')
                entry['new_pair'] = _transform_report(
                    transform_from_matrix(oriented),
                    sensor_frames[first_name], sensor_frames[second_name])
                entry['optimization_applied'] = True
                # optimized_pair is always emitted in FRONT <- sensor
                # direction.  Use that direction for KIT conversion even if
                # the UI stored the same pair in reverse order.
                if optimized_pair.startswith('front-'):
                    optimized_sensor = optimized_pair.split('-', 1)[1]
                    optimized_mount = transform_from_matrix(matrix)
                    entry['child_frame'] = sensor_frames[optimized_sensor]
                    entry['new'] = _transform_report(
                        optimized_mount, FRONT, sensor_frames[optimized_sensor])
                    entry['new_sensor_kit'] = _transform_report(
                        kit_from_front @ optimized_mount,
                        KIT, sensor_frames[optimized_sensor])
        else:
            # 单对候选留作诊断，不发布未完成整组验收的 KIT 外参。
            for entry in report.values():
                if 'new_sensor_kit' in entry:
                    entry['candidate_sensor_kit'] = entry.pop('new_sensor_kit')

    metrics['extrinsics'] = report
    # 只展示实际导出的安装外参对应的分数。四雷达下的非前向配对也须使用同一份 YAML 的相对关系。
    joint = metrics.get('three_radar_optimization', {})
    if len(selected) == 3:
        if joint.get('available'):
            metrics['saved_pair_validation'] = joint['final_validation']
    else:
        deltas = {'front': Transform.identity()}
        for key, entry in report.items():
            if entry.get('new_sensor_kit'):
                side = key.split('-')[1]
                new = entry['new']
                mount = Transform(Rotation.from_quat(new['quaternion_xyzw']), np.array([new[k] for k in 'xyz']))
                deltas[side] = mount @ mounts[side].inverse()
        for key, pair in metrics['pairs'].items():
            first, second = key.split('-')
            if first not in deltas or second not in deltas:
                continue
            if first == 'front':
                metrics['saved_pair_validation'][key] = corrections[key]['candidate_validation']
            elif len(pair['matches']) == 3:
                delta = deltas[first].inverse() @ deltas[second]
                metrics['saved_pair_validation'][key] = validate_transformed_pair(
                    planes[first], planes[second], pair['matches'], delta.rotation.as_matrix(),
                    delta.translation, metrics['thresholds']['overlap_distance_m'])
    return front_kit


def validate_single_capture(result, planes, metrics, settings, cancelled):
    """Use the weighted loop only if the usual three-lidar solution fails raw replay."""
    from dataclasses import replace
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import yaml
    from lidar_cube_joint import evaluate_request, face_weight

    attempts = []
    with TemporaryDirectory(prefix='cube_single_') as tmp:
        path = Path(tmp) / 'candidate.yaml'
        for weighted in (False, True):
            weights = None
            if weighted:
                weights = {key: min((face_weight(p.points) for s in key.split('-') for p in planes[s]), default=1.)
                           for key in metrics['pairs']}
            front = attach_extrinsic_report(result, planes, metrics, weights)
            joint = metrics.get('three_radar_optimization', {})
            method = '弱面闭环降权' if weighted else '原三雷达闭环'
            if not joint.get('available'):
                attempts.append(dict(method=method, reason=joint.get('reason', '无法形成候选外参')))
                if not front or not all(c.get('available') for c in metrics['extrinsic_corrections'].values()):
                    break
                continue
            mounts = [front] + [entry['new_sensor_kit'] for entry in metrics['extrinsics'].values() if 'new_sensor_kit' in entry]
            if len(mounts) != 3:
                raise ValueError('三雷达候选外参不完整')
            path.write_text(yaml.safe_dump({KIT: {m['child_frame']: {k:m[k] for k in ('x','y','z','roll','pitch','yaw')} for m in mounts}},sort_keys=False))
            req = replace(result.request, extrinsics_source='yaml', extrinsics=str(path))
            _, _, _, pairs = evaluate_request(req, settings, lambda _: None, cancelled)
            failed = [key for key,m in pairs.items() if m['quality'] not in ('PASS','WARNING') or m.get('matched_planes') != 3]
            attempts.append(dict(method=method, failed=failed, weights=weights))
            if not failed:
                metrics['saved_pair_validation'] = pairs
                metrics['single_solver'] = dict(selected=method, attempts=attempts)
                return front
        metrics['single_solver'] = dict(selected=None, attempts=attempts)
        joint = metrics.get('three_radar_optimization')
        if joint:
            joint.update(available=False, success=False, quality='FAIL', reason='单包候选未通过验证：' + str(attempts[-1]))
        metrics['saved_pair_validation'] = {}
        for entry in metrics.get('extrinsics', {}).values():
            if isinstance(entry,dict) and 'new_sensor_kit' in entry:
                entry['candidate_sensor_kit'] = entry.pop('new_sensor_kit')
        return front
