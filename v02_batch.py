"""V02 six-bag placement recognition and batch calibration."""
from dataclasses import dataclass
from datetime import datetime
from itertools import product, islice
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
import json
import re
import numpy as np
import yaml
from scipy.spatial import cKDTree
from lidar_cube_core import (Bag, BASE, FRONT, KIT, MAPPING_SIDES, AccumulationRequest, Transform,
    PointCloud2, TFMessage, deserialize_message, static_edges, lookup, Timeline, stamp_ns, read_xyz, thin,
    find_chain, compose_chain)
from lidar_cube_joint import Capture, fit_joint, evaluate_request, replay, correspondence, xyz_record
from lidar_cube_viewer import ViewerSettings
from v02_scene import Scene, pack_edges, transform_key

SLOTS = tuple(f'{group}_{pose}' for group in ('FLT','FRT') for pose in ('F3d','0.3L','0.3R'))
PARAMS = dict(plane_threshold=.02, plane_min_inliers=20, plane_max_planes=3, plane_max_iterations=100,
              plane_min_angle=80., plane_min_span=0., plane_max_span=2.)
ROI_FILE = 'V02_ROI.yaml'
MACHINE_FILE = 'V02_machine.yaml'


def sides_for(slot):
    return ('front', 'left' if slot.startswith('FLT_') else 'right', 'top')


RADAR_KEYWORD = re.compile(r'front|left|right|top', re.IGNORECASE)
PLACEMENT = {'F': 'F3d', 'L': '0.3L', 'R': '0.3R'}


def slot_from_name(name):
    radars = {radar.lower() for radar in RADAR_KEYWORD.findall(name)}
    if radars == {'front', 'left', 'top'}:
        group = 'FLT'
    elif radars == {'front', 'right', 'top'}:
        group = 'FRT'
    else:
        return None

    # Exclude radar names themselves (e.g. FRONT) and uppercase words like FLT.
    remaining = RADAR_KEYWORD.sub(' ', name)
    directions = set(re.findall(r'[A-Z]+', remaining)).intersection(PLACEMENT)
    if len(directions) != 1:
        return None
    return f'{group}_{PLACEMENT[directions.pop()]}'


def discover(root):
    found = {slot: [] for slot in SLOTS}
    for bag in sorted(root.iterdir()):
        if bag.is_dir() and next(bag.glob('*.db3'), None):
            slot = slot_from_name(bag.name)
            if slot:
                found[slot].append(bag)
    return found


def box(points):
    return np.stack((points.min(0), points.max(0)), axis=1)


def corners(bounds):
    return np.array(list(product(*bounds)))


def valid_bounds(value):
    value = np.asarray(value, float)
    if value.shape != (3,2) or not np.isfinite(value).all() or (value[:,0]>=value[:,1]).any():
        raise ValueError('ROI 每个方向需要有效的最小值和最大值')
    return value


def write_yaml(path, data):
    with NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as stream:
        yaml.safe_dump(data,stream,allow_unicode=True,sort_keys=False)
    Path(stream.name).replace(path)


def preview_pose(bag, edges, timestamps):
    dynamic = {}
    if '/tf' in bag.topics:
        for raw in bag.records('/tf'):
            for tf in deserialize_message(raw,TFMessage).transforms:
                t = Transform.from_ros(tf.transform)
                dynamic.setdefault((tf.header.frame_id,tf.child_frame_id),{})[stamp_ns(tf.header.stamp)] = t
    edges = dict(edges)
    for key,data in dynamic.items():
        if len(data)>1:
            edges[key]=Timeline([[ts,*v.translation,*v.rotation.as_quat()] for ts,v in data.items()])
    chain=find_chain(edges,'map',FRONT)
    for ts in timestamps:
        pose=compose_chain(edges,chain,ts)
        if pose is not None:
            return pose
    raise ValueError('预览帧没有对应的 map 变换')


def locate(points, faces):
    # Translation-only search: the support fixes the orientation. Score all three faces separately.
    if len(points)<24:
        return None
    tree=cKDTree(points); ref=faces.reshape(-1,3); origin=ref.mean(0)
    seeds=np.unique(np.round((points-origin)/.09).astype(int),axis=0)*.09
    seeds=np.vstack(([0.,0.,0.],seeds)); rough=ref[::8]; scores=[]
    for start in range(0,len(seeds),180):
        batch=seeds[start:start+180]; dist=tree.query(rough[None,:,:]+batch[:,None,:])[0]
        scores.extend(np.mean(np.minimum(dist,.14),axis=1))
    candidates=[]
    for idx in np.argsort(scores):
        delta=seeds[idx].copy()
        if any(np.linalg.norm(delta-prev)<.09 for _,prev in candidates):continue
        for _ in range(16):
            dist,near=tree.query(ref+delta);mask=dist<.12
            if mask.sum()<20:break
            step=np.median(points[near[mask]]-(ref[mask]+delta),axis=0);delta+=step
            if np.linalg.norm(step)<.001:break
        dist=tree.query(ref+delta)[0].reshape(3,-1)
        score=float(np.max(np.mean(np.minimum(dist,.15),axis=1)))
        candidates.append((score,delta))
        if len(candidates)==12:break
    candidates.sort(key=lambda row:row[0]);best,shift=candidates[0]
    if best>.045 or any(q<best+.008 and np.linalg.norm(v-shift)>.16 for q,v in candidates[1:]):return None
    return shift


@dataclass
class Preview:
    bag: Path
    slot: str
    pose: Transform  # display <- front
    points: np.ndarray
    search: np.ndarray
    bounds: np.ndarray  # evaluation uses front-aligned ROI
    manual: bool
    note: str


def preview(path,slot,use_map=True):
    bag=Bag(path);edges=static_edges(bag); topics=bag.lidar_topics()
    missing=set(sides_for(slot))-topics.keys()
    if missing:raise ValueError('缺少话题 '+', '.join(sorted(missing)))
    base=lookup(edges,BASE,FRONT); msgs=[deserialize_message(raw,PointCloud2) for raw in islice(bag.records(topics['front']),8)]
    if not msgs or any(m.header.frame_id!=FRONT for m in msgs):raise ValueError('前雷达消息或 frame 不正确')
    world=True
    try:pose=preview_pose(bag,edges,[stamp_ns(m.header.stamp) for m in msgs])
    except ValueError:pose=base;world=False
    maps=sorted(bag.directory.glob('map[0-9]*/front_map.npy'),reverse=True) if use_map else []
    pts=np.load(maps[0],allow_pickle=False,mmap_mode='r') if maps else thin(np.concatenate([read_xyz(m) for m in msgs]),.015)
    with np.load(Path(__file__).with_name('aps_templates.npz')) as template:
        bound=box(base.inverse().apply(template[slot+'_box']));faces=base.inverse().apply(template[slot+'_faces'].reshape(-1,3)).reshape(3,-1,3)
    search=box(pose.apply(corners(bound)))+np.array([.8,.8,.5])[:,None]*[-1,1]
    pts=pose.apply(pts); keep=np.all((pts>=search[:,0]-.3)&(pts<=search[:,1]+.3),axis=1);pts=pts[keep]
    inside=pts[np.all((pts>=search[:,0])&(pts<=search[:,1]),axis=1)]
    delta=locate(pose.inverse().apply(inside),faces) if world else None
    if delta is not None:
        # Include returns from the side and top sensors near the template boundaries.
        bound = bound + delta[:,None] + np.array([-.025,.025])
    manual=delta is None;note=('地图坐标缺失，按设备坐标人工确认' if not world else '请人工确认模板框' if manual else '已定位')
    saved=bag.directory/ROI_FILE
    if saved.is_file():
        try:
            state = yaml.safe_load(saved.read_text())
            if not isinstance(state, dict):
                raise ValueError('ROI 配置为空或格式错误')
            if state['slot'] == slot and state['bag'] == bag.directory.name:
                bound = valid_bounds(state['bounds'])
                manual = False
                note = '已恢复 ROI' if world else '已恢复 ROI · 设备坐标'
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
            manual = True
            note = '已保存的 ROI 无法读取，请重新确认范围'
    return Preview(bag.directory,slot,pose,pts,search,bound,manual,note)


RADAR_NAMES = {'front': '前', 'left': '左', 'right': '右', 'top': '顶'}


def log_bag_result(log, slot, stage, stats, pairs):
    for side, sensor in stats.items():
        summary = sensor['summary']
        angle = summary['max_right_angle_deviation_deg']
        rms = summary['max_plane_rms_m']
        details = []
        if angle is not None:
            details.append(f'最大直角偏差 {angle:.2f}°')
        if rms is not None:
            details.append(f'最大单面 RMS {1000*rms:.1f} mm')
        if summary.get('reason'):
            details.append(summary['reason'])
        log(f"{slot} · {stage} · {RADAR_NAMES[side]}雷达 · 滤波后 {sensor['filtered_points']} 点，"
            f"拟合 {summary['plane_count']} 面，{summary['quality']}　" + '，'.join(details))
    for pair, metric in pairs.items():
        name = '–'.join(RADAR_NAMES[k] for k in pair.split('-'))
        corner, normal = metric.get('corner_distance_m'), metric.get('normal_error_deg')
        details = (f'交点距离 {corner*100:.2f} cm，平均法向夹角 {normal:.2f}°'
                   if corner is not None and normal is not None else metric.get('reason') or '配对不完整')
        log(f"{slot} · {stage} · {name} {metric['quality']}　{details}")


def calibrate(previews,params,parent,progress,cancelled,cache=None,log=None,on_result=None):
    if log is None:
        log = progress
    scenes = {}
    sources = {}
    changes = {}
    caps=[]; frames={};baseline={};initial=[]
    if [p.slot for p in previews]!=list(SLOTS) or len({p.bag for p in previews})!=6:raise ValueError('需要六个不同摆位的数据包')
    for p in previews:
        progress(p.slot+' · 初始提面'); sides=sides_for(p.slot)
        try:
            signature = (str(p.bag), p.bounds.tobytes(), tuple(sorted(params.items())),
                         tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in sorted(p.bag.glob('*.db3'))))
            saved = cache.get(p.slot) if cache is not None else None
            if saved and saved[0] == signature:
                cap, record, scenes[p.slot], sources[p.slot] = saved[1:]
                for side in sides:
                    if side in baseline:
                        diff = baseline[side].inverse() @ cap.baseline[side]
                        if frames[side]!=cap.frames[side] or np.linalg.norm(diff.translation)>1e-6 or diff.rotation.magnitude()>1e-6:
                            raise ValueError(f'{side} 安装基准不一致')
                    baseline[side],frames[side]=cap.baseline[side],cap.frames[side]
                log(f"{p.slot} · 已复用上次初始提面和配对结果")
                log_bag_result(log, p.slot, "初始匹配", record["sensors"], record["pairs"])
                if on_result:
                    on_result('bag', (p.slot, '初始', record['sensors'], record['pairs'], record['frames']))
                caps.append(cap)
                initial.append(record)
                continue
            req=AccumulationRequest(str(p.bag),'',valid_bounds(p.bounds),calibration_sides=sides)
            settings=ViewerSettings(req,analysis_voxel=.02,filter_radius=.06,filter_min_neighbors=5,**params);settings.validate()
            clouds,planes,stats,pairs=evaluate_request(req,settings,lambda s:progress(p.slot+' · '+s),cancelled)
            counts = '，'.join(f"{RADAR_NAMES[side]} {x['used_frames']} 帧/{x['roi_points']} 点"
                              for side, x in clouds.stats.items())
            log(f'{p.slot} · ROI 累计完成　{counts}')
            log_bag_result(log, p.slot, '初始匹配', stats, pairs)
            if on_result:
                on_result('bag', (p.slot, '初始', stats, pairs, clouds.stats))
            for side in sides:
                if stats[side]['summary']['quality']=='FAIL' or len(planes[side])!=3:raise ValueError(f'{side} 雷达三面拟合失败，请检查 ROI 或参数')
            order=[[0,1,2]]+[correspondence({'pairs':pairs},'front',s) for s in sides[1:]]
            cross=correspondence({'pairs':pairs},sides[1],sides[2]);
            if [cross[i] for i in order[1]]!=order[2]:raise ValueError('三组对应面不一致')
            fr=clouds.provenance['sensor_frames'];edges=static_edges(Bag(p.bag));old={s:lookup(edges,KIT,fr[s]) for s in sides}
            for s in sides:
                if s in baseline:
                    diff=baseline[s].inverse()@old[s]
                    if frames[s]!=fr[s] or np.linalg.norm(diff.translation)>1e-6 or diff.rotation.magnitude()>1e-6:raise ValueError(f'{s} 安装基准不一致')
                frames[s],baseline[s]=fr[s],old[s]
            scenes[p.slot] = Scene.capture(clouds, planes)
            sources[p.slot] = dict(frames=fr, edges=pack_edges(edges),
                initial=transform_key({s: old['front'].inverse() @ old[s] for s in sides}))
            patches=[[planes[s][i].points for i in idx] for s,idx in zip(sides,order)]
            caps.append(Capture(p.bag,None,'',settings,sides,fr,old,patches,{}));initial.append(dict(slot=p.slot,sensors=stats,pairs=pairs,frames=clouds.stats));del clouds
            if cache is not None:cache[p.slot]=(signature,caps[-1],initial[-1],scenes[p.slot],sources[p.slot])
        except Exception as exc:raise ValueError(f'{p.slot} / {p.bag.name}：{exc}') from exc
    progress('六包四雷达联合拟合');poses=fit_joint(caps,cancelled,MAPPING_SIDES);front=baseline['front']; mounts={}
    log('六包联合求解完成，前雷达锁定')
    for side,correction in zip(MAPPING_SIDES,poses):
        new=baseline[side] if side=='front' else front@correction@front.inverse()@baseline[side]
        mounts[frames[side]]=xyz_record(new)
        old = baseline[side]
        displacement_cm = 100 * np.linalg.norm(new.translation-old.translation)
        angle_deg = np.rad2deg((old.rotation.inv()*new.rotation).magnitude())
        changes[side] = [(100*(new.translation-old.translation)).tolist(), float(angle_deg)]
        if on_result:
            on_result('mount', (side, 100*(new.translation-old.translation), angle_deg))
        log(f'{RADAR_NAMES[side]}雷达 · 相对 /tf_static 安装修正 {displacement_cm:.2f} cm，{angle_deg:.2f}°')
    with TemporaryDirectory(dir=parent,prefix='.v02_') as tmp:
        stage=Path(tmp)/'result';stage.mkdir();candidate=stage/'candidate.yaml';write_yaml(candidate,{KIT:mounts});rows=[];failures=[]
        for p,cap in zip(previews,caps):
            try:
                folder = stage/'clouds_3d'/p.slot
                scenes[p.slot].save(folder/'initial.npz')
                def save_joint(clouds, planes):
                    scene = Scene.capture(clouds, planes)
                    scene.save(folder/'joint.npz')
                    from scipy.spatial.transform import Rotation
                    mounts_used = {s: Transform(Rotation.from_quat(t['quaternion_xyzw']), np.array(t['translation']))
                                   for s, t in scene.metadata['mounts'].items()}
                    sources[p.slot]['joint'] = transform_key(mounts_used)
                row=replay(cap,candidate,lambda msg:progress(p.slot+' · 复验 · '+msg),cancelled,on_geometry=save_joint)
            except Exception as exc:
                raise RuntimeError(f'{p.slot} / {p.bag.name} 复验失败：{exc}') from exc
            rows.append(row)
            log_bag_result(log, p.slot, '最终复验', row['sensors'], row['pairs'])
            if on_result:
                on_result('bag', (p.slot, '复验', row['sensors'], row['pairs'], row['frames']))
            for key,m in row['pairs'].items():
                if m.get('quality') not in ('PASS','WARNING') or m.get('matched_planes')!=3:failures.append(p.slot+' '+key)
            if row['summary']['valid']!=3 and not any(f.startswith(p.slot+' ') for f in failures):failures.append(p.slot+' 配对不完整')
        inputs=[dict(slot=p.slot,bag=p.bag.name,bounds=p.bounds.tolist()) for p in previews];write_yaml(stage/'batch_config.yaml',dict(params=params,cases=inputs));
        (stage/'validation.json').write_text(json.dumps(dict(cases=inputs,params=params,initial=initial,validation=rows,failures=failures,scene_sources=sources,mount_changes=changes),ensure_ascii=False,indent=2,allow_nan=False));
        if not failures:candidate.rename(stage/'suggested_extrinsics.yaml')
        out=parent/('FLRT_V02_result_report_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));stage.rename(out)
    return out,failures
