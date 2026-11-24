"""ROI 点云后处理。

输入是 lidar_cube_core.AccumulationResult.clouds 中的世界坐标点。
本模块不读取 ROS bag，也不依赖 Tk，便于后续指标模块复用。
"""
from dataclasses import dataclass, replace
from itertools import combinations, permutations

import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    """第一版滤波参数，先保持参数少而且容易解释。"""

    voxel_size: float = 0.01
    min_neighbors: int = 3
    neighbor_radius: float = 0.08


@dataclass
class PlaneResult:
    normal: np.ndarray
    offset: float
    inlier_mask: np.ndarray
    residuals: np.ndarray
    # The points belonging to this plane in the input array.  This is filled
    # by extract_planes so a caller can draw the fitted plane without having
    # to reconstruct the intermediate remaining-point arrays.
    points: np.ndarray | None = None

    @property
    def inlier_ratio(self):
        return float(self.inlier_mask.mean()) if len(self.inlier_mask) else 0.0


def voxel_downsample(points, voxel_size):
    """每个体素保留一个点，只用于处理和显示，不改变原始累计结果。"""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0 or voxel_size <= 0:
        return points.copy()
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def radius_filter(points, radius=0.08, min_neighbors=3):
    """保留指定半径内邻居数足够的点。

    使用 scipy 的 cKDTree。半径和邻居数需要根据雷达距离、材质和点密度调节。
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points.copy()
    if radius <= 0 or min_neighbors <= 0:
        return points.copy()
    tree = cKDTree(points)
    counts = np.asarray(tree.query_ball_point(points, radius, return_length=True))
    return points[counts >= min_neighbors]


def filter_points(points, config=FilterConfig()):
    """执行体素降采样和邻域密度滤波，返回处理后的点云。"""
    sampled = voxel_downsample(points, config.voxel_size)
    return radius_filter(sampled, config.neighbor_radius, config.min_neighbors)


def fit_plane(points, distance_threshold=0.015):
    """用 SVD 拟合平面，并按点到平面的距离返回内点。

    平面写成 n dot x + d = 0，normal 会归一化。
    该函数适合一个 ROI 中已经基本对应同一物理面的点云。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError('At least three points are required to fit a plane.')
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    residuals = np.abs(points @ normal + offset)
    return PlaneResult(normal, offset, residuals <= distance_threshold, residuals)


def plane_from_three_points(a, b, c):
    """由三个不共线点计算平面 n dot x + d = 0。"""
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    length = np.linalg.norm(normal)
    if length < 1e-10:
        return None
    normal = normal / length
    offset = -float(normal @ a)
    return normal, offset


def point_plane_distances(points, normal, offset):
    """计算每个点到单位法向量平面的绝对距离。"""
    return np.abs(points @ normal + offset)


def plane_spans(points):
    """返回点集在自身平面内两个主方向上的跨度。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0, 0.0
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return float(np.ptp(centered @ vh[0])), float(np.ptp(centered @ vh[1]))


def _plane_quality(rms_m, p95_m, inlier_ratio, spans_m):
    """根据实测点云给出单雷达平面质量标签。

    40 cm 标定物在不同雷达上的回波密度和边缘噪声差异较大，因此把
    PASS 门槛放宽到 5 cm RMS / 10 cm P95。仍保留跨度、内点比例和更宽的
    WARNING 区间，用来识别明显的背景面或严重稀疏结果。
    """
    span = max(spans_m)
    if rms_m <= .05 and p95_m <= .10 and inlier_ratio >= .01 and span <= 1.20:
        return 'PASS'
    if rms_m <= .08 and p95_m <= .20 and inlier_ratio >= .005 and span <= 1.50:
        return 'WARNING'
    return 'FAIL'


def compute_plane_metrics(cloud, planes):
    """计算单个雷达的平面质量指标。

    ``inlier_ratio`` 使用该平面内点数除以本雷达输入点数，便于跨雷达比较。
    返回值只包含 Python 标量和列表，可以直接写入 JSON/YAML。
    """
    cloud = np.asarray(cloud, dtype=np.float64)
    total = len(cloud)
    report = []
    for index, plane in enumerate(planes or [], start=1):
        points = np.asarray(plane.points if plane.points is not None else np.empty((0, 3)),
                            dtype=np.float64)
        if len(points):
            distances = point_plane_distances(points, plane.normal, plane.offset)
            rms_m = float(np.sqrt(np.mean(distances ** 2)))
            p95_m = float(np.percentile(distances, 95))
        else:
            rms_m = p95_m = float('inf')
        spans = plane_spans(points)
        inlier_ratio = float(len(points) / total) if total else 0.0
        report.append({
            'plane_index': index,
            'normal': np.asarray(plane.normal, dtype=float).tolist(),
            'offset_m': float(plane.offset),
            'rms_m': rms_m,
            'p95_point_plane_distance_m': p95_m,
            'inlier_count': int(len(points)),
            'inlier_ratio': inlier_ratio,
            'span_m': [float(spans[0]), float(spans[1])],
            'quality': _plane_quality(rms_m, p95_m, inlier_ratio, spans),
        })
    return report


def summarize_plane_fit(side, entries):
    """直接使用本雷达独立拟合的面；跨雷达匹配和外参修正均在此之后。"""
    count = len(entries)
    summary = dict(plane_count=count, quality='FAIL', reason='', plane_angles=[],
                   max_right_angle_deviation_deg=None, max_plane_rms_m=None)
    if count != 3:
        summary['reason'] = f'{side} 拟合到 {count} 个平面，标定要求恰好 3 个。'
        return summary

    normals = np.asarray([entry['normal'] for entry in entries], dtype=float)
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        summary['reason'] = f'{side} 拟合面法向量无效。'
        return summary
    normals = normals / lengths[:, None]
    for i, j in combinations(range(3), 2):
        # 法向量可能翻转，取无向夹角，偏离直角的幅度不受符号影响。
        cosine = np.clip(abs(normals[i] @ normals[j]), 0., 1.)
        angle = float(np.degrees(np.arccos(cosine)))
        summary['plane_angles'].append(dict(plane_a_index=entries[i]['plane_index'],
                                            plane_b_index=entries[j]['plane_index'],
                                            angle_deg=angle, deviation_deg=90. - angle))
    summary['max_right_angle_deviation_deg'] = max(a['deviation_deg'] for a in summary['plane_angles'])
    rms = [entry['rms_m'] for entry in entries]
    if all(np.isfinite(rms)):
        summary['max_plane_rms_m'] = float(max(rms))
    if any(entry['quality'] == 'FAIL' for entry in entries):
        summary['reason'] = f'{side} 存在质量为 FAIL 的拟合面。'
    else:
        summary['quality'] = 'WARNING' if any(e['quality'] == 'WARNING' for e in entries) else 'PASS'
    return summary


def _orient_plane(normal, offset, reference):
    normal = np.asarray(normal, dtype=float)
    offset = float(offset)
    if float(normal @ reference) < 0:
        return -normal, -offset
    return normal, offset


def _pair_plane_metrics(plane_a, plane_b, overlap_distance=.03):
    """计算一对对应平面的法向量、点面距离和点云邻近误差。

    平面方程常数项指标已移除。它相对于全局原点，
    会把法向量的小角度误差放大成虚假的平移误差。这里使用两组平面
    内点到对方拟合平面的双向距离。
    """
    from scipy.spatial import cKDTree

    def invalid(reason):
        return {
            'normal_error_deg': None,
            'point_to_plane_rms_m': float('inf'),
            'point_to_plane_p95_m': float('inf'),
            'reference_to_source_point_to_plane_rms_m': float('inf'),
            'reference_to_source_point_to_plane_p95_m': float('inf'),
            'source_to_reference_point_to_plane_rms_m': float('inf'),
            'source_to_reference_point_to_plane_p95_m': float('inf'),
            'chamfer_distance_m': float('inf'),
            'overlap_ratio': 0.0,
            'quality': 'FAIL',
            'reason': reason,
        }

    normal_a = np.asarray(plane_a.normal, dtype=float)
    norm_a = float(np.linalg.norm(normal_a))
    if norm_a <= 1e-12:
        return invalid('参考平面法向量无效。')
    normal_a = normal_a / norm_a
    offset_a = float(plane_a.offset) / norm_a

    normal_b_raw = np.asarray(plane_b.normal, dtype=float)
    norm_b = float(np.linalg.norm(normal_b_raw))
    if norm_b <= 1e-12:
        return invalid('源平面法向量无效。')
    normal_b, offset_b = _orient_plane(
        normal_b_raw / norm_b,
        float(plane_b.offset) / norm_b,
        normal_a,
    )
    angle_deg = float(np.degrees(np.arccos(
        np.clip(abs(float(normal_a @ normal_b)), -1.0, 1.0))))

    points_a = np.asarray(
        plane_a.points if plane_a.points is not None else np.empty((0, 3)),
        dtype=float)
    points_b = np.asarray(
        plane_b.points if plane_b.points is not None else np.empty((0, 3)),
        dtype=float)
    points_a = points_a[np.isfinite(points_a).all(axis=1)] if len(points_a) else points_a
    points_b = points_b[np.isfinite(points_b).all(axis=1)] if len(points_b) else points_b

    def point_plane_stats(points, normal, offset):
        if not len(points):
            return float('inf'), float('inf'), np.empty((0,), dtype=float)
        distances = np.abs(points @ normal + offset)
        return (float(np.sqrt(np.mean(distances ** 2))),
                float(np.percentile(distances, 95)),
                distances)

    # 双向点面距离：A 面内点到 B 平面，B 面内点到 A 平面。
    a_to_b_rms, a_to_b_p95, a_to_b = point_plane_stats(
        points_a, normal_b, offset_b)
    b_to_a_rms, b_to_a_p95, b_to_a = point_plane_stats(
        points_b, normal_a, offset_a)
    all_point_distances = np.concatenate((a_to_b, b_to_a))
    if len(all_point_distances):
        point_rms = float(np.sqrt(np.mean(all_point_distances ** 2)))
        point_p95 = float(np.percentile(all_point_distances, 95))
    else:
        point_rms = point_p95 = float('inf')

    if len(points_a) and len(points_b):
        distances_a = cKDTree(points_b).query(points_a, k=1)[0]
        distances_b = cKDTree(points_a).query(points_b, k=1)[0]
        chamfer = float((distances_a.mean() + distances_b.mean()) / 2.0)
        overlap = float(((distances_a <= overlap_distance).mean()
                         + (distances_b <= overlap_distance).mean()) / 2.0)
    else:
        chamfer = float('inf')
        overlap = 0.0

    point_metric_finite = np.isfinite(point_rms) and np.isfinite(point_p95)
    # Chamfer 仅保留为兼容旧报告的内部字段，不再显示，也不参与质量判定。
    # 放宽跨雷达面面匹配门槛：允许实测中的小角度和边缘噪声，
    # 但保留 25° 以上的明显姿态失配为 FAIL。
    if (angle_deg <= 15 and point_metric_finite
            and point_rms <= .05 and point_p95 <= .10):
        quality = 'PASS'
    elif (angle_deg <= 25 and point_metric_finite
          and point_rms <= .08 and point_p95 <= .20):
        quality = 'WARNING'
    else:
        quality = 'FAIL'
    return {
        'normal_error_deg': angle_deg,
        'point_to_plane_rms_m': point_rms,
        'point_to_plane_p95_m': point_p95,
        'reference_to_source_point_to_plane_rms_m': a_to_b_rms,
        'reference_to_source_point_to_plane_p95_m': a_to_b_p95,
        'source_to_reference_point_to_plane_rms_m': b_to_a_rms,
        'source_to_reference_point_to_plane_p95_m': b_to_a_p95,
        'chamfer_distance_m': chamfer,
        'overlap_ratio': overlap,
        'quality': quality,
    }


def _match_trihedra(planes_a, planes_b):
    """在初始姿态下枚举 3! 种对应；平移不影响几何准入。"""
    groups = []
    for planes in (planes_a, planes_b):
        if len(planes) != 3:
            return [], '两台雷达都必须恰好拟合出 3 个面。'
        normals = np.asarray([p.normal for p in planes], dtype=float)
        lengths = np.linalg.norm(normals, axis=1)
        if not np.isfinite(lengths).all() or min(lengths) < 1e-12:
            return [], '存在无效法向量。'
        normals = normals / lengths[:, None]
        if np.linalg.matrix_rank(normals) != 3:
            return [], '三个法向量不独立。'
        groups.append(normals)
    dots = np.abs(groups[0] @ groups[1].T)
    allowed = dots >= np.cos(np.radians(25.0)) - 1e-12
    feasible = [order for order in permutations(range(3))
                if all(allowed[i, j] for i, j in enumerate(order))]
    if len(feasible) != 1:
        return [], ('25° 内没有完整三面对应。' if not feasible else '25° 内有多套三面对应，无法唯一匹配。')
    return list(enumerate(feasible[0])), ''


def compute_pair_metrics(planes_a, planes_b, overlap_distance=.03):
    indices, reason = _match_trihedra(planes_a, planes_b)
    if reason:
        return dict(matched_planes=0, matches=[], quality='FAIL', reason=reason,
                    corner_distance_m=None, mean_normal_error_deg=None,
                    normal_error_deg=None, point_to_plane_rms_m=None,
                    point_to_plane_p95_m=None, chamfer_distance_m=None, overlap_ratio=0.)
    return _score_matches(planes_a, planes_b, indices, overlap_distance)


def validate_transformed_pair(reference, source, matches, rotation, translation, overlap_distance=.03):
    """同一个公共坐标系刚体变换作用到源内点及源平面，沿用求解前的对应。"""
    transformed = []
    for plane in source:
        normal = rotation @ plane.normal
        transformed.append(replace(plane, normal=normal, offset=plane.offset - normal @ translation,
                                   points=np.asarray(plane.points) @ rotation.T + translation))
    indices = [(m['plane_a_index'] - 1, m['plane_b_index'] - 1) for m in matches]
    return _score_matches(reference, transformed, indices, overlap_distance)


def trihedral_corner(planes):
    """三个拟合平面 n·x+d=0 的公共交点，单位与输入点云一致（米）。"""
    if len(planes) != 3:
        return None
    normals = np.asarray([plane.normal for plane in planes], dtype=float)
    offsets = np.asarray([plane.offset for plane in planes], dtype=float)
    if not np.isfinite(normals).all() or not np.isfinite(offsets).all():
        return None
    if np.linalg.matrix_rank(normals) < 3:
        return None
    try:
        return np.linalg.solve(normals, -offsets)
    except np.linalg.LinAlgError:
        return None


def _score_matches(planes_a, planes_b, indices, overlap_distance):
    matches = []
    for ia, ib in indices:
        metrics = _pair_plane_metrics(planes_a[ia], planes_b[ib], overlap_distance)
        metrics.update(plane_a_index=ia + 1, plane_b_index=ib + 1)
        matches.append(metrics)
    normal_errors = [m['normal_error_deg'] for m in matches]
    point_rms = [m['point_to_plane_rms_m'] for m in matches]
    point_p95 = [m['point_to_plane_p95_m'] for m in matches]
    chamfers = [m['chamfer_distance_m'] for m in matches]
    overlaps = [m['overlap_ratio'] for m in matches]
    if len(planes_a) != 3 or len(planes_b) != 3 or len(matches) != 3 or any(
            m['quality'] == 'FAIL' for m in matches):
        quality = 'FAIL'
    elif all(m['quality'] == 'PASS' for m in matches):
        quality = 'PASS'
    else:
        quality = 'WARNING'
    corner_a = trihedral_corner(planes_a)
    corner_b = trihedral_corner(planes_b)
    corner_distance = (float(np.linalg.norm(corner_a - corner_b))
                       if corner_a is not None and corner_b is not None else None)
    return {
        'corner_distance_m': corner_distance,
        'matched_planes': len(matches),
        'normal_error_deg': float(np.mean(normal_errors)),
        'mean_normal_error_deg': float(np.mean(normal_errors)),
        'max_normal_error_deg': float(max(normal_errors)),
        'point_to_plane_rms_m': float(max(point_rms)),
        'mean_point_to_plane_rms_m': float(np.mean(point_rms)),
        'point_to_plane_p95_m': float(max(point_p95)),
        'mean_point_to_plane_p95_m': float(np.mean(point_p95)),
        'chamfer_distance_m': float(max(chamfers)),
        'mean_chamfer_distance_m': float(np.mean(chamfers)),
        'overlap_ratio': float(min(overlaps)),
        'quality': quality,
        'matches': matches,
    }


def estimate_extrinsic_correction(planes_reference, planes_source, pair_metrics=None, overlap_distance=.03):
    """由两个三面体估计 ``source 点云 -> reference 点云`` 的刚体修正。

    输入平面已经位于同一个公共坐标系中，因此返回的修正变换可以直接
    左乘到当前 ``reference <- source`` 外参上。三面体的三个法向量提供
    旋转约束，三个平面偏移提供平移约束。只有三个平面都存在时才计算。
    """
    reference = list(planes_reference or [])
    source = list(planes_source or [])
    if len(reference) != 3 or len(source) != 3:
        return {
            'available': False,
            'reason': '两个雷达都必须恰好拟合出三个面。',
        }
    pair_metrics = pair_metrics or compute_pair_metrics(reference, source)
    if pair_metrics.get('reason'):
        return {'available': False, 'reason': pair_metrics.get('reason') or '对应面匹配质量为 FAIL。'}
    matches = list(pair_metrics.get('matches', []))
    if len(matches) != 3:
        return {
            'available': False,
            'reason': '三个面没有形成一一对应关系。',
        }

    source_normals = []
    reference_normals = []
    offset_rhs = []
    for match in matches:
        ref_plane = reference[int(match['plane_a_index']) - 1]
        src_plane = source[int(match['plane_b_index']) - 1]
        ref_normal = np.asarray(ref_plane.normal, dtype=float)
        ref_length = np.linalg.norm(ref_normal)
        ref_normal = ref_normal / ref_length
        src_length = np.linalg.norm(src_plane.normal)
        src_normal, src_offset = _orient_plane(
            np.asarray(src_plane.normal) / src_length, src_plane.offset / src_length, ref_normal)
        source_normals.append(src_normal)
        reference_normals.append(ref_normal)
        # 对 source 平面施加 x' = R x + t 后：
        # d_reference = d_source - n_reference dot t。
        offset_rhs.append(float(src_offset - ref_plane.offset / ref_length))

    source_normals = np.asarray(source_normals, dtype=float)
    reference_normals = np.asarray(reference_normals, dtype=float)
    if np.linalg.matrix_rank(reference_normals) < 3:
        return {
            'available': False,
            'reason': '三个平面法向量不独立，无法稳定求解平移。',
        }

    # Kabsch：求 R，使 R @ source_normal ~= reference_normal。
    covariance = source_normals.T @ reference_normals
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    # 旋转以后，源平面的法向量已经是 R @ n_source。平移方程必须使用
    # 这个旋转后的法向量；直接使用 reference_normals 会在法向量存在噪声
    # 时把旋转误差错误地转成较大的平移误差。
    rotated_source_normals = (rotation @ source_normals.T).T
    translation, _, _, _ = np.linalg.lstsq(
        rotated_source_normals, np.asarray(offset_rhs, dtype=float), rcond=None)

    # 仅使用平面方程时，法向量的几度拟合误差会在目标离坐标原点较远时
    # 放大成厘米级边缘偏移。使用各对应平面中的实际内点做一次鲁棒点到
    # 平面优化，可以直接最小化“变换后的源点到参考平面”的距离。
    # 每个平面等量采样，避免点数更多的单个面支配优化结果。
    from scipy.spatial.transform import Rotation

    refinement_used = False
    point_fit_rms = None
    point_fit_p95 = None
    try:
        from scipy.optimize import least_squares

        source_points = []
        target_normals = []
        target_offsets = []
        max_points_per_plane = 800
        for match in matches:
            ref_plane = reference[int(match['plane_a_index']) - 1]
            src_plane = source[int(match['plane_b_index']) - 1]
            points = np.asarray(
                src_plane.points if src_plane.points is not None else np.empty((0, 3)),
                dtype=float,
            )
            if len(points) < 3:
                continue
            if len(points) > max_points_per_plane:
                sample = np.linspace(
                    0, len(points) - 1, max_points_per_plane, dtype=int)
                points = points[sample]
            normal = np.asarray(ref_plane.normal, dtype=float)
            normal_length = np.linalg.norm(normal)
            normal = normal / normal_length
            source_points.append(points)
            target_normals.append(np.repeat(normal[None, :], len(points), axis=0))
            target_offsets.append(np.full(len(points), float(ref_plane.offset) / normal_length))

        if len(source_points) == 3:
            source_points = np.concatenate(source_points, axis=0)
            target_normals = np.concatenate(target_normals, axis=0)
            target_offsets = np.concatenate(target_offsets, axis=0)
            initial = np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation]

            def residuals(state):
                candidate_rotation = Rotation.from_rotvec(state[:3])
                transformed = candidate_rotation.apply(source_points) + state[3:]
                return np.einsum(
                    'ij,ij->i', target_normals, transformed) + target_offsets

            # soft_l1 抑制边缘点、背景残留和少量错误内点的影响。
            refined = least_squares(
                residuals,
                initial,
                loss='soft_l1',
                f_scale=0.02,
                max_nfev=300,
            )
            if refined.success and np.isfinite(refined.x).all():
                rotation = Rotation.from_rotvec(refined.x[:3]).as_matrix()
                translation = np.asarray(refined.x[3:], dtype=float)
                final_residuals = residuals(refined.x)
                point_fit_rms = float(np.sqrt(np.mean(final_residuals ** 2)))
                point_fit_p95 = float(np.percentile(np.abs(final_residuals), 95))
                refinement_used = True
    except Exception:
        # scipy 优化不可用或输入点不足时，仍保留上面的平面方程解。
        refinement_used = False

    rotation_obj = Rotation.from_matrix(rotation)
    residual_normals = np.linalg.norm(
        (rotation @ source_normals.T).T - reference_normals, axis=1)
    residual_offsets = (rotation @ source_normals.T).T @ translation \
        - np.asarray(offset_rhs)
    plane_equation_offset_fit_rms = float(
        np.sqrt(np.mean(residual_offsets ** 2)))
    # 优化后，真正有意义的偏移误差是变换后源内点到参考平面的距离；
    # 旋转后的三个法向量不完全一致时，单独比较三个平面方程常数项会
    # 把正常的法向量噪声误报成很大的“偏移误差”。
    reported_offset_fit_rms = (
        point_fit_rms if point_fit_rms is not None
        else plane_equation_offset_fit_rms)
    validation = validate_transformed_pair(reference, source, matches, rotation, translation, overlap_distance)
    failed = [f"P{m['plane_a_index']}↔P{m['plane_b_index']}" for m in validation['matches']
              if m['quality'] == 'FAIL']
    return {
        'available': not failed,
        'reason': '候选验收超限：' + '、'.join(failed) if failed else '',
        'candidate_validation': validation,
        'source_frame': 'source_cloud',
        'reference_frame': 'reference_cloud',
        'rotation_matrix': rotation.tolist(),
        'translation_xyz_m': np.asarray(translation, dtype=float).tolist(),
        'rotation_rpy_deg': np.degrees(rotation_obj.as_euler('xyz')).tolist(),
        # 四元数和旋转向量用于稳定比较；欧拉角只为兼容现有 YAML 输出。
        'rotation_quaternion_xyzw': rotation_obj.as_quat().tolist(),
        'rotation_vector_deg': np.degrees(rotation_obj.as_rotvec()).tolist(),
        'rotation_error_deg': float(np.degrees(rotation_obj.magnitude())),
        'translation_error_m': float(np.linalg.norm(translation)),
        'normal_fit_rms': float(np.sqrt(np.mean(residual_normals ** 2))),
        'offset_fit_rms_m': reported_offset_fit_rms,
        'plane_equation_offset_fit_rms_m': plane_equation_offset_fit_rms,
        'refinement_method': 'robust_point_to_plane' if refinement_used else 'plane_equation_fallback',
        'point_to_plane_fit_rms_m': point_fit_rms,
        'point_to_plane_fit_p95_m': point_fit_p95,
        'matched_planes': matches,
    }



def _transform_matrix_to_vector(transform):
    """Convert a 4x4 SE(3) matrix to ``[tx, ty, tz, rx, ry, rz]``.

    The rotation is represented by a rotation vector instead of Euler angles,
    so the three-radar refinement does not introduce an artificial gimbal-lock
    problem near +/-90 degree pitch.
    """
    from scipy.spatial.transform import Rotation

    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError('A transform matrix must have shape (4, 4).')
    return np.concatenate([
        transform[:3, 3],
        Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
    ])


def _transform_vector_to_matrix(vector):
    """Convert ``[tx, ty, tz, rx, ry, rz]`` back to a 4x4 SE(3) matrix."""
    from scipy.spatial.transform import Rotation

    vector = np.asarray(vector, dtype=float)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError('A transform vector must have six finite values.')
    result = np.eye(4, dtype=float)
    result[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    result[:3, 3] = vector[:3]
    return result


def _se3_residual(measured, candidate, translation_scale=.01,
                  rotation_scale=np.deg2rad(1.0)):
    """Return a dimensionless 6D residual for two transforms.

    ``measured`` and ``candidate`` have the same direction.  The error is
    ``inverse(measured) @ candidate``; translation and rotation are scaled so
    one centimetre and one degree are comparable during least-squares fitting.
    """
    from scipy.spatial.transform import Rotation

    measured = np.asarray(measured, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    error = np.linalg.inv(measured) @ candidate
    translation = error[:3, 3] / float(translation_scale)
    rotation = Rotation.from_matrix(error[:3, :3]).as_rotvec()
    return np.concatenate([translation, rotation / float(rotation_scale)])


def _se3_error_summary(measured, candidate):
    """Return human-readable translation/rotation error for two SE(3) poses."""
    from scipy.spatial.transform import Rotation

    measured = np.asarray(measured, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    error = np.linalg.inv(measured) @ candidate
    rotation = Rotation.from_matrix(error[:3, :3])
    return {
        'translation_error_m': float(np.linalg.norm(error[:3, 3])),
        'rotation_error_deg': float(np.degrees(rotation.magnitude())),
        'translation_xyz_m': error[:3, 3].astype(float).tolist(),
        'rotation_vector_deg': np.degrees(rotation.as_rotvec()).astype(float).tolist(),
    }


def optimize_three_radar_extrinsics(observed_pairs, calibration_sides,
                                    pair_weights=None):
    """Jointly refine a three-radar pose graph with one reference radar.

    ``observed_pairs`` contains the pairwise transforms already estimated from
    the fitted trihedral planes.  For ``('front', 'left', 'top')`` the expected
    keys are ``front-left``, ``front-top`` and ``left-top``.  The first two
    measurements initialize the two unknown transforms to the reference radar;
    the third pair supplies the loop-closure constraint.

    The function returns optimized pair matrices and before/after closure
    errors.  It does not use KIT-frame transforms, so the result is independent
    of the absolute sensor-kit anchor until the caller converts the optimized
    reference-relative poses back to KIT.
    """
    sides = tuple(dict.fromkeys(calibration_sides or ()))
    if len(sides) != 3 or 'front' not in sides:
        return {'available': False,
                'reason': '三雷达联合优化需要恰好三个雷达且必须包含前雷达。'}

    reference = 'front'
    others = [side for side in sides if side != reference]
    first, second = others

    def pair_key(a, b):
        direct = f'{a}-{b}'
        reverse = f'{b}-{a}'
        if direct in observed_pairs:
            return direct, False
        if reverse in observed_pairs:
            return reverse, True
        return None, False

    def get_pair(a, b):
        key, reverse = pair_key(a, b)
        if key is None:
            raise KeyError(f'缺少 {a}-{b} 两两外参观测。')
        matrix = np.asarray(observed_pairs[key], dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f'{key} 不是 4x4 变换矩阵。')
        return np.linalg.inv(matrix) if reverse else matrix

    try:
        measured_first = get_pair(reference, first)
        measured_second = get_pair(reference, second)
        measured_between = get_pair(first, second)
    except (KeyError, ValueError, np.linalg.LinAlgError) as error:
        return {'available': False, 'reason': str(error)}

    weights = dict(pair_weights or {})

    def weight_for(a, b):
        return float(weights.get(f'{a}-{b}', weights.get(f'{b}-{a}', 1.0)))

    weight_first = max(weight_for(reference, first), 1e-3)
    weight_second = max(weight_for(reference, second), 1e-3)
    weight_between = max(weight_for(first, second), 1e-3)
    initial_first = measured_first.copy()
    initial_second = measured_second.copy()

    def residual(vector):
        candidate_first = _transform_vector_to_matrix(vector[:6])
        candidate_second = _transform_vector_to_matrix(vector[6:])
        predicted_between = np.linalg.inv(candidate_first) @ candidate_second
        return np.concatenate([
            np.sqrt(weight_first) * _se3_residual(measured_first, candidate_first),
            np.sqrt(weight_second) * _se3_residual(measured_second, candidate_second),
            np.sqrt(weight_between) * _se3_residual(measured_between, predicted_between),
        ])

    try:
        from scipy.optimize import least_squares
        initial = np.concatenate([
            _transform_matrix_to_vector(initial_first),
            _transform_matrix_to_vector(initial_second),
        ])
        result = least_squares(
            residual,
            initial,
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=200,
        )
        optimized_first = _transform_vector_to_matrix(result.x[:6])
        optimized_second = _transform_vector_to_matrix(result.x[6:])
        success = bool(result.success and np.isfinite(result.cost))
        message = str(result.message)
        iterations = int(result.nfev)
    except Exception as error:
        return {'available': False, 'reason': f'三雷达联合优化失败：{error}'}

    optimized_between = np.linalg.inv(optimized_first) @ optimized_second
    initial_between = np.linalg.inv(initial_first) @ initial_second
    initial_residuals = {
        f'{reference}-{first}': _se3_error_summary(measured_first, initial_first),
        f'{reference}-{second}': _se3_error_summary(measured_second, initial_second),
        f'{first}-{second}': _se3_error_summary(measured_between, initial_between),
    }
    final_residuals = {
        f'{reference}-{first}': _se3_error_summary(measured_first, optimized_first),
        f'{reference}-{second}': _se3_error_summary(measured_second, optimized_second),
        f'{first}-{second}': _se3_error_summary(measured_between, optimized_between),
    }
    return {
        'available': success,
        'success': success,
        'message': message,
        'iterations': iterations,
        'reference': reference,
        'calibration_sides': list(sides),
        'weights': {
            f'{reference}-{first}': weight_first,
            f'{reference}-{second}': weight_second,
            f'{first}-{second}': weight_between,
        },
        'initial_residuals': initial_residuals,
        'final_residuals': final_residuals,
        'optimized_pairs': {
            f'{reference}-{first}': optimized_first.tolist(),
            f'{reference}-{second}': optimized_second.tolist(),
            f'{first}-{second}': optimized_between.tolist(),
        },
    }


def evaluate_metrics(clouds, plane_results, overlap_distance=.03,
                     calibration_sides=None,
                     calibration_target='front-left-right', *, include_corrections=True):
    """生成单雷达指标，并对选定雷达做两两评估。

    新接口由 ``calibration_sides`` 决定参与评估的雷达，例如
    ``('front', 'left', 'top')`` 会生成 front-left、front-top 和
    left-top 三组。前雷达仍然是公共世界坐标和运动轨迹参考，但指标
    与平面修正会对每一对已选雷达计算。
    ``calibration_target`` 只为旧调用方保留，新的查看器不再使用它。
    """
    per_sensor = {
        side: compute_plane_metrics(clouds.get(side, np.empty((0, 3))),
                                    (plane_results or {}).get(side, []))
        for side in clouds
    }
    if calibration_sides is None:
        legacy = {
            'front-left-right': ('front', 'left', 'right'),
            'front-top': ('front', 'top'),
        }
        calibration_sides = legacy.get(calibration_target)
    if calibration_sides is None:
        raise ValueError('Calibration sides are required.')
    calibration_sides = tuple(dict.fromkeys(calibration_sides))
    if len(calibration_sides) < 2 or 'front' not in calibration_sides:
        raise ValueError('Calibration sides must contain front and at least one other radar.')
    if any(side not in ('front', 'left', 'right', 'top') for side in calibration_sides):
        raise ValueError(f'Unsupported calibration side: {calibration_sides}')
    # 所有已选雷达两两配对。组合顺序保持用户勾选列表中的顺序，因而
    # 对三路输入会得到 front-left、front-top、left-top 三组结果。
    pair_sets = tuple(
        (calibration_sides[index], calibration_sides[other])
        for index in range(len(calibration_sides))
        for other in range(index + 1, len(calibration_sides))
    )
    sensor_summary = {side: summarize_plane_fit(side, per_sensor.get(side, []))
                      for side in calibration_sides}
    pairs = {}
    for first, second in pair_sets:
        pair_name = f'{first}-{second}'
        pairs[pair_name] = compute_pair_metrics(
            (plane_results or {}).get(first, []),
            (plane_results or {}).get(second, []),
            overlap_distance)
        pair = pairs[pair_name]
        failures = [sensor_summary[side]['reason'] for side in (first, second)
                    if sensor_summary[side]['quality'] == 'FAIL']
        if failures:
            pair.update(quality='FAIL', reason=' '.join(failures))
        reason = pair.get('reason', '')
        pair['geometry_quality'] = 'FAIL' if reason else 'PASS'
    metrics = {
        'version': 2,
        'calibration_sides': list(calibration_sides),
        'pairing_rule': 'all_selected_pairs',
        'thresholds': {
            'overlap_distance_m': float(overlap_distance),
            'plane_rms_pass_m': .05,
            'plane_p95_pass_m': .10,
            'plane_inlier_ratio_pass': .01,
            'plane_span_pass_m': 1.20,
            'plane_rms_warning_m': .08,
            'plane_p95_warning_m': .20,
            'normal_pass_deg': 15.0,
            'point_to_plane_rms_pass_m': .05,
            'point_to_plane_p95_pass_m': .10,
            'normal_warning_deg': 25.0,
            'point_to_plane_rms_warning_m': .08,
            'point_to_plane_p95_warning_m': .20,
        },
        'per_sensor': per_sensor,
        'sensor_summary': sensor_summary,
        'pairs': pairs,
    }
    if include_corrections:
        metrics['extrinsic_corrections'] = compute_extrinsic_corrections(plane_results, metrics)
    return metrics


def compute_extrinsic_corrections(plane_results, metrics):
    """在已完成的配对上求解，沿用同一组内点、对应关系和验收门槛。"""
    corrections = {}
    for key, pair in metrics['pairs'].items():
        first, second = key.split('-')
        reason = pair.get('reason', '')
        if reason:
            correction = dict(available=False, eligible=False, reason=reason)
        else:
            correction = estimate_extrinsic_correction(
                plane_results[first], plane_results[second], pair,
                metrics['thresholds']['overlap_distance_m'])
            correction['eligible'] = True
        correction['initial_pair_quality'] = pair['quality']
        corrections[key] = correction
    return corrections


def extract_plane(points, distance_threshold=0.015, max_iterations=500,
                  min_inliers=100, random_seed=0, forbidden_normals=None,
                  min_plane_angle_deg=30.0):
    """RANSAC 多平面提取的单平面框架。

    当前函数先提取一个平面。调用者可以删除返回的内点后再次调用，
    从而逐个提取正方体的三个面。RANSAC 的采样和评分步骤保留 TODO，
    方便继续学习和实现。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')
    if len(points) < 3:
        raise ValueError('At least three points are required for RANSAC.')
    if (distance_threshold <= 0 or max_iterations <= 0 or min_inliers <= 0
            or not 0 < min_plane_angle_deg < 90):
        raise ValueError('RANSAC parameters must be positive.')

    forbidden_normals = tuple(np.asarray(n, dtype=float) for n in (forbidden_normals or ()))
    normal_dot_limit = np.cos(np.deg2rad(min_plane_angle_deg))

    rng = np.random.default_rng(random_seed)
    best_result = None
    best_count = 0

    for _ in range(max_iterations):
        # TODO 1: 随机选择三个不同的点。
        sample_indices = rng.choice(len(points), size=3, replace=False)
        a, b, c = points[sample_indices]

        # 三点接近共线时，无法定义稳定平面，跳过本次采样。
        model = plane_from_three_points(a, b, c)
        if model is None:
            continue
        normal, offset = model

        # 多平面提取时，跳过与已接受平面近似平行的候选，避免把同一个
        # 大平面切成几个局部区域后误报成多个平面。
        if any(abs(float(normal @ previous)) >= normal_dot_limit
               for previous in forbidden_normals):
            continue

        # TODO 2: 计算所有点到当前候选平面的距离。
        residuals = point_plane_distances(points, normal, offset)

        # TODO 3: 根据距离阈值生成内点掩码。
        inlier_mask = residuals <= distance_threshold
        count = int(inlier_mask.sum())

        # TODO 4: 保存内点数量最多的候选平面。
        if count > best_count:
            best_count = count
            best_result = PlaneResult(normal, offset, inlier_mask, residuals)

    if best_result is None or best_count < min_inliers:
        return None

    # TODO 5: 可选的精修步骤。
    # 使用 best_result.inlier_mask 重新调用 fit_plane，
    # 让最终平面由全部内点共同拟合，而不是只由随机三点决定。
    return best_result


def extract_planes(points, distance_threshold=0.015, max_iterations=500,
                   min_inliers=100, max_planes=3, random_seed=0,
                   min_plane_angle_deg=30.0, max_plane_span=None,
                   min_plane_span=0.0):
    """迭代提取多个平面，返回平面结果列表和剩余点云。

    每轮只在 remaining_points 中寻找一个平面，提取成功后删除该平面的
    内点，再开始下一轮。对于三面直角目标，通常设置 max_planes=3。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')
    if (max_planes <= 0 or not 0 < min_plane_angle_deg < 90
            or (max_plane_span is not None and max_plane_span <= 0)
            or min_plane_span < 0):
        raise ValueError(
            'max_planes must be positive, min_plane_angle_deg must be in (0, 90), '
            'and min_plane_span must be nonnegative.')

    remaining_points = points.copy()
    planes = []
    accepted_normals = []

    # 允许跳过若干个超出跨度限制的背景平面，再提取目标平面。
    for plane_index in range(max_planes * 3):
        if len(planes) >= max_planes:
            break
        if len(remaining_points) < max(3, min_inliers):
            break

        # 使用不同的随机种子，避免每一轮重复同一组随机序列。
        plane_result = extract_plane(
            remaining_points,
            distance_threshold=distance_threshold,
            max_iterations=max_iterations,
            min_inliers=min_inliers,
            random_seed=random_seed + plane_index,
            forbidden_normals=accepted_normals,
            min_plane_angle_deg=min_plane_angle_deg,
        )
        if plane_result is None:
            break

        # 用当前平面的全部内点重新拟合，减少随机三点带来的误差。
        refined = fit_plane(
            remaining_points[plane_result.inlier_mask],
            distance_threshold=distance_threshold,
        )

        normal_dot_limit = np.cos(np.deg2rad(min_plane_angle_deg))
        if any(abs(float(refined.normal @ previous)) >= normal_dot_limit
               for previous in accepted_normals):
            # 精修可能让单个候选越过夹角门槛。保留剩余点，按循环上限换种子再试。
            continue

        # 重新在本轮剩余点中计算精修平面的内点，供下一轮删除。
        residuals = point_plane_distances(
            remaining_points,
            refined.normal,
            refined.offset,
        )
        refined_mask = residuals <= distance_threshold
        refined = PlaneResult(
            refined.normal,
            refined.offset,
            refined_mask,
            residuals,
            remaining_points[refined_mask].copy(),
        )

        if (max_plane_span is not None
                and max(plane_spans(refined.points)) > max_plane_span):
            # 当前候选面明显大于目标面，通常是 ROI 中的背景面。
            # 删除该背景面的内点后继续搜索，避免第一个背景面导致结果为空。
            remaining_points = remaining_points[~refined_mask]
            continue
        if min_plane_span > 0 and min(plane_spans(refined.points)) < min_plane_span:
            # The candidate may be a narrow background strip or a partial
            # return.  Remove it and continue searching for a wider target
            # plane in the remaining ROI points.
            remaining_points = remaining_points[~refined_mask]
            continue
        planes.append(refined)
        accepted_normals.append(refined.normal)
        remaining_points = remaining_points[~refined_mask]

    return planes, remaining_points


def process_clouds(clouds, config=FilterConfig(), plane_threshold=0.015):
    """对当前累计结果中的各路点云执行同一套初始处理。"""
    filtered = {side: filter_points(points, config) for side, points in clouds.items()}
    planes = {side: fit_plane(points, plane_threshold)
              for side, points in filtered.items() if len(points) >= 3}
    return filtered, planes
