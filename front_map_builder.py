#!/usr/bin/env python3
"""Build one selected LiDAR's static map from a ROS 2 bag."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from nav_msgs.msg import Path as PathMsg
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty

from lidar_cube_core import (Bag, Timeline, lidar_topic, read_xyz, stamp_ns,
                             thin, WORKSPACE)
from rclpy.serialization import deserialize_message

# 独立地图保存分辨率。viewer 的显示体素只能减少显示点，不能恢复建图阶段
# 已经被丢弃的点，因此这里必须使用与期望显示/拟合分辨率一致的值。
MAP_VOXEL_SIZE = 0.01


def build_static_map(bag_path, output, sensor_side, mapping_lidar_frame):
    """静止采集模式：不依赖 NDT，直接在雷达自身坐标系累计点云。"""
    bag = Bag(bag_path)
    topics = bag.lidar_topics()
    if sensor_side not in topics:
        raise RuntimeError(f'bag 中没有 {sensor_side} 雷达点云话题。')

    batches = []
    total = 0
    first_stamp = None
    last_stamp = None
    for raw in bag.records(topics[sensor_side]):
        message = deserialize_message(raw, PointCloud2)
        timestamp = stamp_ns(message.header.stamp)
        points = read_xyz(message)
        if not len(points):
            continue
        first_stamp = timestamp if first_stamp is None else first_stamp
        last_stamp = timestamp
        batch = thin(points.astype(np.float32), MAP_VOXEL_SIZE)
        batches.append(batch)
        total += len(batch)
        if total > 400_000:
            batches = [thin(np.concatenate(batches), MAP_VOXEL_SIZE)]
            total = len(batches[0])

    if not batches:
        raise RuntimeError(f'{sensor_side} 雷达没有读取到有效点云。')
    map_points = thin(np.concatenate(batches), MAP_VOXEL_SIZE)
    if first_stamp is None or last_stamp is None:
        raise RuntimeError(f'{sensor_side} 雷达没有有效点云时间戳。')
    if last_stamp <= first_stamp:
        last_stamp = first_stamp + 1
    poses = [
        [int(first_stamp), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [int(last_stamp), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    bag_path_obj = Path(bag_path).expanduser().resolve()
    bag_name = (bag_path_obj.parent.name
                if bag_path_obj.is_file() and bag_path_obj.suffix == '.db3'
                else bag_path_obj.name)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f'{sensor_side}_map.npy', map_points)
    (output / f'{sensor_side}_map_meta.json').write_text(json.dumps({
        'bag': bag_name,
        'sensor_side': sensor_side,
        'mapping_lidar_frame': mapping_lidar_frame,
        'map_points': int(len(map_points)),
        'poses': 0,
        'pose_mode': 'identity_static',
        'trajectory_distance_m': 0.0,
        'max_pose_jump_m': 0.0,
        'mapping_mode': 'static',
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    return len(map_points), poses


def infer_lidar_frame(bag_path, sensor_side):
    """Read the first valid scan frame from the selected real radar topic."""
    bag = Bag(bag_path)
    topic = bag.lidar_topics().get(sensor_side)
    if topic is None:
        raise RuntimeError(f'bag 中没有 {sensor_side} 雷达点云话题。')
    first = bag.first_or_last(topic)
    if first is None:
        raise RuntimeError(f'{sensor_side} 雷达没有有效点云。')
    return deserialize_message(first[1], PointCloud2).header.frame_id


class Collector:
    def __init__(self, node, output, sensor_side, mapping_lidar_frame):
        self.output = Path(output)
        self.sensor_side = sensor_side
        self.mapping_lidar_frame = mapping_lidar_frame
        self.sensor_topic = lidar_topic(sensor_side)
        self.poses = []
        self.map_points = np.empty((0, 3), dtype=np.float32)
        node.create_subscription(PathMsg, '/frame_path', self.path, 10)
        node.create_subscription(PointCloud2, '/output_map', self.map, 10)

    def path(self, message):
        self.poses = [[stamp_ns(p.header.stamp), p.pose.position.x, p.pose.position.y,
                       p.pose.position.z, p.pose.orientation.x, p.pose.orientation.y,
                       p.pose.orientation.z, p.pose.orientation.w] for p in message.poses]

    def map(self, message):
        self.map_points = read_xyz(message).astype(np.float32)

    def rebuild_map_from_bag(self, bag_path):
        bag = Bag(bag_path)
        topic = bag.lidar_topics()[self.sensor_side]
        timeline = Timeline(self.poses)
        batches = []
        total = 0
        for raw in bag.records(topic):
            message = deserialize_message(raw, PointCloud2)
            pose = timeline.at(stamp_ns(message.header.stamp))
            if pose is None:
                continue
            points = read_xyz(message)
            if len(points):
                frame_points = thin(pose.apply(points).astype(np.float32), MAP_VOXEL_SIZE)
                batches.append(frame_points)
                total += len(frame_points)
                if total > 400_000:
                    batches = [thin(np.concatenate(batches), MAP_VOXEL_SIZE)]
                    total = len(batches[0])
        if batches:
            return thin(np.concatenate(batches), MAP_VOXEL_SIZE)
        return np.empty((0, 3), dtype=np.float32)

    def save(self, bag_path):
        bag_path_obj = Path(bag_path).expanduser().resolve()
        bag_name = (bag_path_obj.parent.name
                    if bag_path_obj.is_file() and bag_path_obj.suffix == '.db3'
                    else bag_path_obj.name)
        if not len(self.map_points):
            self.map_points = self.rebuild_map_from_bag(bag_path)
        if len(self.poses) < 10 or not len(self.map_points):
            raise RuntimeError(f'轨迹质量不足。{self.sensor_side} 雷达 NDT 没有产生足够的轨迹或地图点。')
        positions = np.asarray([row[1:4] for row in self.poses], dtype=float)
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if (not np.isfinite(positions).all() or not np.isfinite(jumps).all()
                or len(jumps) == 0 or jumps.max() > 2.0 or jumps.sum() < .5):
            raise RuntimeError(f'轨迹质量不足。{self.sensor_side} 雷达 NDT 轨迹存在异常跳变或有效运动范围过小。')
        self.output.mkdir(parents=True, exist_ok=True)
        np.save(self.output / f'{self.sensor_side}_map.npy', self.map_points)
        data = {
            'schema_version': 1, 'source': 'lidar_cube_viewer/independent_mapping',
            'sensor_side': self.sensor_side,
            'mapping_lidar_frame': self.mapping_lidar_frame,
            'bag': bag_name, 'frame': f'{self.sensor_side}_map',
            'reference_frame': self.mapping_lidar_frame, 'timestamp_unit': 'ns',
            'poses': self.poses, 'valid': [True] * len(self.poses),
        }
        (self.output / f'{self.sensor_side}_trajectory.json').write_text(json.dumps(data), encoding='utf-8')
        (self.output / f'{self.sensor_side}_map_meta.json').write_text(json.dumps({
            'bag': bag_name, 'sensor_side': self.sensor_side,
            'mapping_lidar_frame': self.mapping_lidar_frame,
            'map_points': len(self.map_points), 'poses': len(self.poses),
            'trajectory_distance_m': float(jumps.sum()), 'max_pose_jump_m': float(jumps.max()),
        }, indent=2), encoding='utf-8')


def run(args):
    mapping_lidar_frame = args.mapping_lidar_frame or infer_lidar_frame(
        args.bag, args.sensor_side)
    if args.static:
        output = Path(args.output).expanduser().resolve()
        map_points, _ = build_static_map(
            args.bag, output, args.sensor_side, mapping_lidar_frame)
        print(json.dumps({'output': str(output), 'sensor_side': args.sensor_side,
                          'poses': 0, 'map_points': map_points,
                          'mapping_mode': 'static'}), flush=True)
        return
    # Only the optional motion-mapping CLI needs the Autoware overlay.
    from tier4_sensor_calibration_msgs.srv import ExtrinsicCalibrator
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init()
    node = rclpy.create_node(f'lidar_cube_{args.sensor_side}_map_builder')
    processes = []
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    def start(command, log):
        stream = open(log, 'w', encoding='utf-8')
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append((process, stream))
        return process

    try:
        launch = WORKSPACE / 'install/sensor_calibration_manager/share/sensor_calibration_manager/launch/rdv/mapping_based_lidar_lidar_calibrator.launch.xml'
        launch_process = start([
            'ros2', 'launch', str(launch), 'rviz:=false',
            f'mapping_pointcloud:={lidar_topic(args.sensor_side)}',
            f'mapping_lidar_frame:={mapping_lidar_frame}',
            'mapping_min_range:=0.5', f'mapping_viz_leaf_size:={MAP_VOXEL_SIZE}',
            f'leaf_size_input:={MAP_VOXEL_SIZE}', 'lidar_calibration_max_frames:=1',
        ], output / f'{args.sensor_side}_mapping.launch.log')
        calibrate = node.create_client(ExtrinsicCalibrator, '/calibrate_lidar_lidar')
        stop = node.create_client(Empty, '/stop_mapping')
        deadline = time.monotonic() + args.start_timeout
        while not calibrate.wait_for_service(timeout_sec=.5):
            if time.monotonic() > deadline:
                raise RuntimeError(f'{args.sensor_side} mapping service did not start.')
        collector = Collector(node, output, args.sensor_side, mapping_lidar_frame)
        mapping_future = calibrate.call_async(ExtrinsicCalibrator.Request())
        time.sleep(2)
        bag_process = start(['ros2', 'bag', 'play', str(Path(args.bag).expanduser()),
                             '--clock', '-r', str(args.rate), '--disable-keyboard-controls'],
                             output / f'{args.sensor_side}_mapping.bag.log')
        while bag_process.poll() is None:
            rclpy.spin_once(node, timeout_sec=.2)
        deadline = time.monotonic() + args.stop_timeout
        while not stop.wait_for_service(timeout_sec=.2):
            if time.monotonic() > deadline:
                raise RuntimeError('Stop-mapping service disappeared.')
        stop_future = stop.call_async(Empty.Request())
        while not stop_future.done() or not mapping_future.done():
            rclpy.spin_once(node, timeout_sec=.2)
            if time.monotonic() > deadline:
                raise RuntimeError(f'{args.sensor_side} mapping did not finish after stop_mapping.')
        # ExtrinsicCalibrator has no success field. Completion means the mapper
        # reached FINISHED after /stop_mapping returned.
        mapping_future.result()
        try:
            collector.save(args.bag)
        except Exception as error:
            (output / f'{args.sensor_side}_mapping.error.txt').write_text(str(error), encoding='utf-8')
            raise
        print(json.dumps({'output': str(output), 'sensor_side': args.sensor_side,
                          'poses': len(collector.poses),
                          'map_points': len(collector.map_points)}), flush=True)
    finally:
        for process, stream in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            stream.close()
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sensor-side', choices=('front', 'left', 'right', 'top'), default='front')
    parser.add_argument('--mapping-lidar-frame', default='',
                        help='Use the selected LiDAR frame for independent mapping.')
    parser.add_argument('--static', action='store_true',
                        help='Build a fixed map without requiring NDT motion tracking.')
    parser.add_argument('--domain', type=int, default=101)
    # 1 倍速更适合 NDT 建图；高速回放可能导致关键帧角度跳变和跟踪丢失。
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--start-timeout', type=float, default=40)
    parser.add_argument('--stop-timeout', type=float, default=180)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
