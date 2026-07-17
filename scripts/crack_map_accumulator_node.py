#!/usr/bin/env python3
"""
crack_map_accumulator_node.py

Builds and publishes a REAL persistent, growing 3D map (not a rendering
trick) by accumulating incoming semantic point clouds over time, with
voxel-grid deduplication to keep memory/bandwidth bounded as the same area
is revisited during a scan.

This complements (does not replace) FAST-LIVO2's own internal map: FAST-
LIVO2 builds its own SLAM map for pose estimation, but only exposes the
*current frame's* points on /cloud_registered each tick — there is no live
topic for its full accumulated map (confirmed by source inspection; the
declared /Laser_map publisher is never actually called, and the full map is
only written to disk via savePCD() at shutdown). This node fills that gap
for crack-aware live visualization during a scan.

Subscribes:  /crack/semantic_cloud   (PointCloud2, XYZRGB + crack uint8 field)
Publishes:   /crack/accumulated_map  (PointCloud2, XYZRGB + crack uint8 field)
             — full deduplicated map so far, republished at publish_rate Hz

Strategy:
  - Maintain an in-memory voxel dictionary: key = (vx, vy, vz) integer voxel
    coordinate at voxel_size resolution, value = (xyz, rgb, crack_label).
  - On crack/non-crack conflict within the same voxel (rare, only at the
    crack boundary), crack label wins — better to over-flag at the boundary
    than lose a true detection to overwrite by a later non-crack observation
    of the same physical point from a slightly different angle.
  - Publish the full map periodically rather than on every input message,
    since publishing potentially hundreds of thousands of points every
    ~0.1s would saturate bandwidth for no practical visualization benefit.
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField


# ---------------------------------------------------------------------------
# PointCloud2 helpers (matching the encoding used by crack_fusion_node.py)
# ---------------------------------------------------------------------------

def pc2_to_xyz_rgb_crack(msg: PointCloud2):
    """Unpack PointCloud2 (XYZ + rgb float32 + crack uint8) → arrays."""
    if msg.width == 0 or msg.point_step == 0 or len(msg.data) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.uint8))

    fields = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)

    def _f32(name):
        off = fields[name].offset
        raw = np.ascontiguousarray(data[:, off:off + 4])
        return raw.view(np.float32).reshape(-1)

    x = _f32('x')
    y = _f32('y')
    z = _f32('z')
    xyz = np.stack([x, y, z], axis=1)

    if 'rgb' in fields:
        off = fields['rgb'].offset
        raw = np.ascontiguousarray(data[:, off:off + 4])
        rgb_packed = raw.view(np.uint32).reshape(-1)
        r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_packed >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_packed & 0xFF).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=1)
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)

    if 'crack' in fields:
        off = fields['crack'].offset
        crack = np.ascontiguousarray(data[:, off:off + 1]).reshape(-1)
    else:
        crack = np.zeros(len(xyz), dtype=np.uint8)

    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], rgb[valid], crack[valid]


def make_pc2_xyzrgb_crack(header, xyz: np.ndarray, rgb: np.ndarray,
                            crack: np.ndarray) -> PointCloud2:
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
    buf[:, 16]    = crack.astype(np.uint8)

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


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class CrackMapAccumulatorNode(Node):

    def __init__(self):
        super().__init__('crack_map_accumulator_node')

        self.declare_parameter('voxel_size_m', 0.02)   # 2cm dedup resolution
        self.declare_parameter('publish_rate_hz', 2.0)
        self.declare_parameter('max_voxels', 3_000_000)  # memory safety cap

        self._voxel_size = self.get_parameter('voxel_size_m').value
        publish_rate = self.get_parameter('publish_rate_hz').value
        self._max_voxels = self.get_parameter('max_voxels').value

        # Map storage: dict keyed by (vx,vy,vz) int32 tuple -> (rgb, crack)
        # Kept as plain dict for O(1) insert/update; fine up to a few million
        # entries on Orin's RAM. Could switch to an Open3D voxel structure
        # if this becomes a bottleneck on very large structures.
        self._voxel_map = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(
            PointCloud2, '/crack/semantic_cloud', self.cloud_cb, qos)

        self.pub_map = self.create_publisher(
            PointCloud2, '/crack/accumulated_map', 5)

        self._last_header = None
        self._dirty = False  # whether new points arrived since last publish

        self.create_timer(1.0 / publish_rate, self._publish_map)

        self.get_logger().info(
            f'CrackMapAccumulatorNode ready. voxel_size={self._voxel_size}m, '
            f'publish_rate={publish_rate}Hz'
        )

    def cloud_cb(self, msg: PointCloud2):
        xyz, rgb, crack = pc2_to_xyz_rgb_crack(msg)
        if len(xyz) == 0:
            return

        self._last_header = msg.header

        # Quantize to voxel grid
        voxel_idx = np.floor(xyz / self._voxel_size).astype(np.int64)

        if len(self._voxel_map) >= self._max_voxels:
            self.get_logger().warn(
                f'Map hit max_voxels cap ({self._max_voxels}); '
                f'dropping new points until restart. Consider increasing '
                f'voxel_size_m or max_voxels.'
            )
            return

        for i in range(len(xyz)):
            key = (int(voxel_idx[i, 0]), int(voxel_idx[i, 1]), int(voxel_idx[i, 2]))
            new_crack = int(crack[i])
            existing = self._voxel_map.get(key)
            if existing is None:
                self._voxel_map[key] = (xyz[i].copy(), rgb[i].copy(), new_crack)
            else:
                _, _, old_crack = existing
                # Crack label wins on conflict (see module docstring rationale).
                # Keep first-seen position/color to avoid jitter from re-obs.
                if new_crack and not old_crack:
                    old_xyz, old_rgb, _ = existing
                    self._voxel_map[key] = (old_xyz, old_rgb, 1)

        self._dirty = True

    def _publish_map(self):
        if not self._dirty or self._last_header is None:
            return
        if len(self._voxel_map) == 0:
            return

        n = len(self._voxel_map)
        xyz_out = np.empty((n, 3), dtype=np.float32)
        rgb_out = np.empty((n, 3), dtype=np.uint8)
        crack_out = np.empty((n,), dtype=np.uint8)

        for i, (xyz_v, rgb_v, crack_v) in enumerate(self._voxel_map.values()):
            xyz_out[i] = xyz_v
            rgb_out[i] = rgb_v
            crack_out[i] = crack_v

        header = self._last_header
        header.stamp = self.get_clock().now().to_msg()
        msg = make_pc2_xyzrgb_crack(header, xyz_out, rgb_out, crack_out)
        self.pub_map.publish(msg)
        self._dirty = False

        n_crack = int(crack_out.sum())
        self.get_logger().info(
            f'Accumulated map: {n} total points, {n_crack} crack-labeled'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CrackMapAccumulatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
