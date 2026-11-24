# LiDAR Cube Viewer V01

用静止采集的三面体点云标定一台机器的前、左、右和顶部雷达。工具在 ROS 2 bag 上离线运行，前雷达作为固定参考。完成六包联合拟合后，从 `FLRT_result_report_…/suggested_extrinsics.yaml` 取得四台雷达相对 `sensor_kit_base_link` 的安装外参。

## 安装和启动

已验证环境为 Ubuntu 22.04、ROS 2 Humble 和系统 Python 3.10。GUI 需要图形桌面，WSL2 可使用 WSLg。在已安装 ROS 2 Humble 并配置好 ROS 软件源的机器上执行：

```bash
sudo apt update
sudo apt install git python3-numpy python3-scipy python3-yaml python3-tk python3-matplotlib fonts-noto-cjk ros-humble-rclpy ros-humble-sensor-msgs ros-humble-tf2-msgs ros-humble-nav-msgs ros-humble-std-srvs

git clone https://github.com/Libpet-Siyuan-Owen/lidar_cube_viewer.git
cd lidar_cube_viewer
source /opt/ros/humble/setup.bash
/usr/bin/python3 lidar_cube_viewer.py
```

每次打开新终端都要先 `source /opt/ros/humble/setup.bash`。第一次启动后选择自己的 bag，后续会记住上次使用的配置。需要指定默认浏览目录时，可在启动前设置 `LIDAR_CUBE_WORKSPACE` 为包含 `data/` 的工作区路径。静态流程可直接从源码启动，无需 `colcon build`。

## 准备数据

一台机器一个目录，每次采集期间保持机器和标定物静止。每个 bag 选一个摆位，让本组的三台雷达都能看到标定物的三个相互垂直的面。完整四雷达流程需要两组各三个不同的包，同一台机器、同一次安装，共用前雷达和顶部雷达。

| 组合 | 每个 bag 需要的点云 | 数量 |
|---|---|---|
| FLT | 前、左、顶 | 三个摆位，各一包 |
| FRT | 前、右、顶 | 三个摆位，各一包 |

输入为 ROS 2 SQLite bag，包含 `metadata.yaml` 和 `.db3`。分片 bag 请在选择窗口选 `metadata.yaml`。每个包还需录制 `/tf_static`，其中的变换链要能连接 `sensor_kit_base_link`、`rslidar_airy_front` 和各参与雷达的实际点云 frame。点云消息类型为 `sensor_msgs/msg/PointCloud2`，话题为 `/sensing/lidar/{front,left,right,top}/pointcloud_raw_ex`，按本组雷达录制即可。

## 从一个包到整车外参

1. **单包建图与累计。** 打开一个 bag，建立或加载各雷达的静态地图。检查标定物位置并确认 XYZ ROI，勾选前雷达和本组另外两台雷达。选择初始 `/tf_static` 或已有外参 YAML，执行累计。帧数设为 `0` 表示使用全部帧。ROI 和参数随包保存在 `Cube_Viewer_config.yaml`。
2. **单包提面与修正。** 执行滤波，再点“执行平面拟合和匹配”。先查看三面体和匹配结果，再点“修正并保存外参指标”。每个包生成自己的 `result_report…/`。六个包各做一次，确认三台雷达都提取到三个有效面。
3. **三个摆位联合拟合。** 在“三标定包联合拟合”中，先选择三个 FLT 包及各自的单包报告，把结果保存位置选为机器目录，然后开始。对 FRT 重复一次。机器目录中会得到 `FLT_result_report_…/` 和 `FRT_result_report_…/`，各含一份 `joint_fit_config.yaml`。
4. **六包四雷达联合拟合。** 在“六包四雷达联合解”里选择上一步的两份 `joint_fit_config.yaml`，点击开始。程序固定前雷达，在六个摆位上共同求解左、右、顶，再用四套候选外参分别从六个原始包重新累计验证。
5. **查看并导出。** 第一张表显示各包三组雷达配对的平均角点距离（cm），最后一列取最差摆位。第二张表显示联合解相对 `/tf_static` 的安装修正。最终复验满足导出条件时，取 `FLRT_result_report_…/suggested_extrinsics.yaml`。其中 XYZ 单位为米，roll/pitch/yaw 单位为弧度，前雷达沿用原安装值。详细法向夹角、逐面和配对指标可在 `validation.json` 或 `metrics.log` 查看。

## 常见问题

| 现象 | 检查方法 |
|---|---|
| 提示找不到 `rclpy`，或 Tk 无法创建窗口 | 在有图形桌面的终端加载 Humble 环境，用 `/usr/bin/python3` 启动。 |
| 缺少某个雷达话题或无法找到坐标变换 | 用 `ros2 bag info <包目录>` 核对点云和 `/tf_static`。前点云 frame 应为 `rslidar_airy_front`，其余通过消息头读取。 |
| 某台雷达不能拟合三个面 | 在累计后的点云中逐台检查 ROI 是否截掉靶面或包含大块背景，再核对平面跨度、夹角、内点阈值。 |
| 希望复查上次结果 | 三包面板加载 `joint_fit_config.yaml`，六包面板加载 `six_fit_config.yaml`。保留同目录的 `validation.json`。重新拟合还需要原始 bag、单包报告和快照所引用的路径。 |

更多实现细节见 [代码流程与模块](docs/流程与模块.md)。
