#!/usr/bin/env python3
"""
range_limited_cloud_node.py

Maintains a PERSISTENT local map: accumulates points from /cloud_registered
over time (voxel-deduplicated to bound memory), but on every publish cycle
filters the accumulated store down to only points within max_range_m of the
sensor's CURRENT position. This gives a "sliding window" map that follows
the sensor — points stay visible as long as you remain near them, and drop
out once you've moved more than max_range_m away, rather than either (a)
showing the entire building forever (no filter) or (b) showing only the
single current frame with no persistence (naive per-message filter).

Subscribes:
  /cloud_registered     (sensor_msgs/PointCloud2, XYZRGB, world frame)
  /aft_mapped_to_init   (nav_msgs/Odometry, current LIO pose)

Publishes:
  /cloud_registered_range_limited   (sensor_msgs/PointCloud2, filtered)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry


def pc2_to_xyz_rgb(msg: PointCloud2):
    if msg.width == 0 or msg.point_step == 0 or len(msg.data) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    fields = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)

    def _f32(name):
        off = fields[name].offset
        raw = np.ascontiguousarray(data[:, off:off + 4])
        return raw.view(np.float32).reshape(-1)

    xyz = np.stack([_f32('x'), _f32('y'), _f32('z')], axis=1)

    if 'rgb' in fields:
        off = fields['rgb'].offset
        raw = np.ascontiguousarray(data[:, off:off + 4])
        rgb_packed = raw.view(np.uint32).reshape(-1)
        r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_packed >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_packed & 0xFF).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=1)
    else:
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)

    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], rgb[valid]


def make_pc2_xyzrgb(header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 16
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

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = N
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * N
    msg.data = buf.tobytes()
    msg.is_dense = True
    return msg


class RangeLimitedCloudNode(Node):

    def __init__(self):
        super().__init__('range_limited_cloud_node')

        self.declare_parameter('max_range_m', 1.5)
        self.declare_parameter('voxel_size_m', 0.02)
        self.declare_parameter('publish_rate_hz', 5.0)

        self._max_range = self.get_parameter('max_range_m').value
        self._voxel_size = self.get_parameter('voxel_size_m').value
        publish_rate = self.get_parameter('publish_rate_hz').value

        self._sensor_pos = None  # [x, y, z] in world frame, from LIO pose
        self._last_header = None

        # Persistent voxel map: key=(vx,vy,vz) int tuple -> (xyz, rgb)
        # This is the full accumulated history; we filter by current
        # distance only at publish time, not at accumulation time, so
        # points are never lost from the store — just not shown when
        # far from the sensor's current position.
        self._voxel_map = {}

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub_odom = self.create_subscription(
            Odometry, '/aft_mapped_to_init', self.odom_cb, qos_be)
        self.sub_cloud = self.create_subscription(
            PointCloud2, '/cloud_registered', self.cloud_cb, qos_be)

        self.pub_cloud = self.create_publisher(
            PointCloud2, '/cloud_registered_range_limited', 5)

        self.create_timer(1.0 / publish_rate, self._publish_filtered)

        self.get_logger().info(
            f'RangeLimitedCloudNode ready. max_range_m={self._max_range}, '
            f'voxel_size_m={self._voxel_size}, publish_rate={publish_rate}Hz '
            f'(persistent local map — accumulates, then filters by CURRENT '
            f'distance at publish time)'
        )

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._sensor_pos = np.array([p.x, p.y, p.z])

    def cloud_cb(self, msg: PointCloud2):
        xyz, rgb = pc2_to_xyz_rgb(msg)
        if len(xyz) == 0:
            return

        self._last_header = msg.header

        voxel_idx = np.floor(xyz / self._voxel_size).astype(np.int64)
        for i in range(len(xyz)):
            key = (int(voxel_idx[i, 0]), int(voxel_idx[i, 1]), int(voxel_idx[i, 2]))
            if key not in self._voxel_map:
                self._voxel_map[key] = (xyz[i].copy(), rgb[i].copy())

    def _publish_filtered(self):
        if self._sensor_pos is None or self._last_header is None:
            return
        if len(self._voxel_map) == 0:
            return

        all_xyz = np.array([v[0] for v in self._voxel_map.values()], dtype=np.float32)
        all_rgb = np.array([v[1] for v in self._voxel_map.values()], dtype=np.uint8)

        dist = np.linalg.norm(all_xyz - self._sensor_pos, axis=1)
        within_range = dist <= self._max_range

        if not within_range.any():
            return

        header = self._last_header
        header.stamp = self.get_clock().now().to_msg()
        filtered_msg = make_pc2_xyzrgb(header, all_xyz[within_range], all_rgb[within_range])
        self.pub_cloud.publish(filtered_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RangeLimitedCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

