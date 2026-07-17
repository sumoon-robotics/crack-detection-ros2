#!/usr/bin/env python3
"""
crack_fusion_node.py

Stage 2: Multi-frame fusion of crack masks with the FAST-LIVO2 colored point cloud.
Uses LIO-estimated camera poses (not VIO) for projection — robust at low HIK frame rates.

Key reference: §3.3 "Multi-frame Multi-modal Fusion" in Deng et al. 2025.

Subscribes:
  /crack/mask                      (sensor_msgs/Image, mono8)
  /fast_livo2/cloud_registered     (sensor_msgs/PointCloud2, XYZRGB)
  /fast_livo2/odometry             (nav_msgs/Odometry)  — LIO pose

Publishes:
  /crack/semantic_cloud            (sensor_msgs/PointCloud2, XYZRGBA — A=crack label)
  /crack/crack_cloud               (sensor_msgs/PointCloud2, crack points only)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import message_filters

import numpy as np
import struct
import threading
from collections import deque
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# PointCloud2 helpers
# ---------------------------------------------------------------------------

def pc2_to_xyz_rgb(msg: PointCloud2):
    """Unpack PointCloud2 → (N,3) float32 XYZ, (N,3) uint8 RGB."""
    if msg.width == 0 or msg.point_step == 0 or len(msg.data) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    # Determine field offsets
    fields = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)

    def _get(name, dtype):
        f = fields[name]
        raw = np.ascontiguousarray(data[:, f.offset: f.offset + np.dtype(dtype).itemsize])
        return raw.view(dtype).reshape(-1)

    xyz = np.stack([_get('x', np.float32),
                    _get('y', np.float32),
                    _get('z', np.float32)], axis=1)

    if 'rgb' in fields:
        rgb_packed = _get('rgb', np.float32).view(np.uint32)
        r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_packed >> 8)  & 0xFF).astype(np.uint8)
        b = ((rgb_packed)       & 0xFF).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=1)
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)

    # Filter NaN/inf
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], rgb[valid]


def make_pc2(header, xyz: np.ndarray, rgb: np.ndarray,
             crack_label: np.ndarray) -> PointCloud2:
    """Pack (N,3) XYZ + (N,3) RGB + (N,) crack label into PointCloud2."""
    fields = [
        PointField(name='x',     offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',     offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',     offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb',   offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='crack', offset=16, datatype=PointField.UINT8,   count=1),
    ]
    point_step = 20
    N = len(xyz)
    buf = np.zeros((N, point_step), dtype=np.uint8)

    xyz32 = np.ascontiguousarray(xyz, dtype=np.float32)
    buf[:, 0:4]  = np.ascontiguousarray(xyz32[:, 0]).view(np.uint8).reshape(-1, 4)
    buf[:, 4:8]  = np.ascontiguousarray(xyz32[:, 1]).view(np.uint8).reshape(-1, 4)
    buf[:, 8:12] = np.ascontiguousarray(xyz32[:, 2]).view(np.uint8).reshape(-1, 4)

    rgb_packed = np.ascontiguousarray(
        (rgb[:, 0].astype(np.uint32) << 16 |
         rgb[:, 1].astype(np.uint32) << 8  |
         rgb[:, 2].astype(np.uint32))
    )
    buf[:, 12:16] = rgb_packed.view(np.uint8).reshape(-1, 4)
    buf[:, 16]    = crack_label.astype(np.uint8)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width  = N
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step   = point_step
    msg.row_step     = point_step * N
    msg.data         = buf.tobytes()
    msg.is_dense     = True
    return msg


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def odom_to_T(odom: Odometry) -> np.ndarray:
    """nav_msgs/Odometry → 4×4 homogeneous transform (world←body)."""
    p = odom.pose.pose.position
    q = odom.pose.pose.orientation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [p.x, p.y, p.z]
    return T


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class CrackFusionNode(Node):

    def __init__(self):
        super().__init__('crack_fusion_node')

        # Camera intrinsics — override via params or calibration launch arg
        self.declare_parameter('fx', 1000.0)
        self.declare_parameter('fy', 1000.0)
        self.declare_parameter('cx', 640.0)
        self.declare_parameter('cy', 400.0)
        self.declare_parameter('img_w', 1280)
        self.declare_parameter('img_h', 800)

        # Extrinsic: the param is named T_lidar_cam for historical reasons,
        # but per FAST-LIVO2 source (src/vio.cpp: "Pci = Rcl * Pli + Pcl"),
        # Rcl/Pcl transform LiDAR-frame points directly INTO the camera
        # frame: p_cam = Rcl @ p_lidar + Pcl. That is, this matrix already
        # IS T_cam_lidar — no inversion needed. (Earlier code incorrectly
        # treated it as T_lidar_cam and inverted it, causing wildly wrong
        # projected pixel coordinates — see debug session 2026-06-17.)
        self.declare_parameter('T_lidar_cam', [
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ])

        self.declare_parameter('mask_buffer_size', 10)
        self.declare_parameter('min_crack_points', 5)
        self.declare_parameter('max_range_m', 1.0)

        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        self._img_w = self.get_parameter('img_w').value
        self._img_h = self.get_parameter('img_h').value
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        T_flat = self.get_parameter('T_lidar_cam').value
        T_param = np.array(T_flat, dtype=np.float64).reshape(4, 4)
        # T_param IS T_cam_lidar directly (see comment above) — use as-is.
        self._T_cam_lidar = T_param
        self._T_lidar_cam = np.linalg.inv(self._T_cam_lidar)

        buf = self.get_parameter('mask_buffer_size').value
        self._min_pts = self.get_parameter('min_crack_points').value
        self._max_range_m = self.get_parameter('max_range_m').value

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        # Ring buffer: (stamp_sec, mask_ndarray)
        self._mask_buf: deque = deque(maxlen=buf)
        # Latest LIO pose
        self._T_world_lidar: np.ndarray = np.eye(4)

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub_mask  = self.create_subscription(
            Image, '/crack/mask', self.mask_cb, 10)
        self.sub_odom  = self.create_subscription(
            Odometry, '/aft_mapped_to_init', self.odom_cb, qos_be)
        self.sub_cloud = self.create_subscription(
            PointCloud2, '/cloud_registered', self.cloud_cb, qos_be)

        self.pub_semantic = self.create_publisher(
            PointCloud2, '/crack/semantic_cloud', 5)
        self.pub_crack = self.create_publisher(
            PointCloud2, '/crack/crack_cloud', 5)

        self.get_logger().info('CrackFusionNode ready.')

    # ------------------------------------------------------------------
    def mask_cb(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'mask cv_bridge: {e}')
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._mask_buf.append((stamp, (mask > 127).astype(bool),
                                   self._T_world_lidar.copy()))

    def odom_cb(self, msg: Odometry):
        with self._lock:
            self._T_world_lidar = odom_to_T(msg)

    def cloud_cb(self, msg: PointCloud2):
        """
        On each new point cloud, project all buffered crack masks.
        Uses the LIO pose stored when each mask was received (multi-frame fusion).
        """
        with self._lock:
            mask_list = list(self._mask_buf)
            T_world_lidar_now = self._T_world_lidar.copy()

        if not mask_list:
            return

        xyz_w, rgb = pc2_to_xyz_rgb(msg)
        if len(xyz_w) == 0:
            return

        crack_label = np.zeros(len(xyz_w), dtype=np.uint8)

        debug_info = []
        for (_, mask, T_world_lidar_at_mask) in mask_list:
            # Transform point cloud into camera frame at mask acquisition time
            # T_cam_world = T_cam_lidar @ T_lidar_world
            T_lidar_world = np.linalg.inv(T_world_lidar_at_mask)
            T_cam_world   = self._T_cam_lidar @ T_lidar_world

            # Bring current cloud (world frame) to camera frame
            xyz_h   = np.hstack([xyz_w, np.ones((len(xyz_w), 1))])  # (N,4)
            xyz_cam = (T_cam_world @ xyz_h.T).T[:, :3]              # (N,3)

            # Keep points in front of camera AND within max detection range.
            # Range-gated to reduce noise: distant points have lower LiDAR
            # point density, more projection uncertainty (small pose/extrinsic
            # errors translate to larger pixel offset at range), and are more
            # likely to be background clutter rather than the inspected
            # surface. Default ceiling matches close-range inspection use case
            # (sensor held within ~1m of the structure being inspected).
            depth = xyz_cam[:, 2]
            in_front = (depth > 0.1) & (depth <= self._max_range_m)
            n_in_front = int(in_front.sum())
            if not in_front.any():
                debug_info.append(f"0 in front/range (of {len(xyz_w)})")
                continue

            # Project to pixel
            uvw  = (self._K @ xyz_cam[in_front].T).T   # (M,3)
            u    = (uvw[:, 0] / uvw[:, 2]).astype(int)
            v    = (uvw[:, 1] / uvw[:, 2]).astype(int)

            valid = (u >= 0) & (u < self._img_w) & \
                    (v >= 0) & (v < self._img_h)
            n_valid = int(valid.sum())
            uv_range = f"u[{u.min()},{u.max()}] v[{v.min()},{v.max()}]" if len(u) else "n/a"

            in_front_idx = np.where(in_front)[0]
            crack_pts_local = mask[v[valid], u[valid]]
            n_on_mask = int(crack_pts_local.sum())
            crack_global_idx = in_front_idx[valid][crack_pts_local]
            crack_label[crack_global_idx] = 1

            debug_info.append(
                f"in_front={n_in_front} in_image={n_valid} on_crack_mask={n_on_mask} "
                f"mask_total_px={int(mask.sum())} {uv_range}"
            )

        if debug_info:
            self.get_logger().info(' | '.join(debug_info))

        # Publish semantic cloud (all points, crack channel set)
        sem_msg = make_pc2(msg.header, xyz_w, rgb, crack_label)
        self.pub_semantic.publish(sem_msg)

        # Publish crack-only cloud
        crack_mask = crack_label > 0
        if crack_mask.sum() >= self._min_pts:
            crack_msg = make_pc2(msg.header,
                                 xyz_w[crack_mask],
                                 rgb[crack_mask],
                                 crack_label[crack_mask])
            self.pub_crack.publish(crack_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CrackFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
