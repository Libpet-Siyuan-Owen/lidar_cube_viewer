"""Processing defaults shared by batch calibration and report replay."""
PARAMS = dict(
    plane_threshold=.02, plane_min_inliers=20, plane_max_planes=3,
    plane_max_iterations=100, plane_min_angle=80., plane_min_span=0., plane_max_span=2.,
    analysis_voxel=.02, filter_radius=.06, filter_min_neighbors=5,
)
