#!/usr/bin/env python3
"""
crack_measurement_node.py

Stage 3: Automated geometric measurement of cracks from the 3D semantic point cloud.
Implements §3.4 of Deng et al. 2025:
  - PCA skeleton extraction per connected crack segment
  - EDT-based width estimation in 3D
  - Crack length along skeleton
  - Publishes per-crack geometry as JSON string on /crack/measurements

Subscribes:  /crack/crack_cloud  (PointCloud2, crack points only)
Publishes:   /crack/measurements (std_msgs/String, JSON)
             /crack/markers      (visualization_msgs/MarkerArray)  — RViz
"""

import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

try:
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pc2_xyz(msg: PointCloud2) -> np.ndarray:
    """Unpack PointCloud2 → (N,3) float32 XYZ (crack cloud only)."""
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)
    fields = {f.name: (f.offset, f.datatype) for f in msg.fields}

    def _f(name):
        off, _ = fields[name]
        return np.ascontiguousarray(data[:, off: off+4]).view(np.float32).reshape(-1)

    xyz = np.stack([_f('x'), _f('y'), _f('z')], axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


def measure_crack_segment(pts: np.ndarray) -> dict:
    """
    Compute crack geometry for a connected 3D point cluster.

    Returns dict with:
      centroid        : [x, y, z]
      length_m        : float  — arc length along PCA skeleton
      width_mean_m    : float  — mean EDT-derived width
      width_max_m     : float  — maximum width
      orientation_vec : [dx, dy, dz]  — principal axis
      n_points        : int
    """
    if len(pts) < 5:
        return {}

    centroid = pts.mean(axis=0)
    pts_c    = pts - centroid

    # PCA → principal axis = crack direction
    pca = PCA(n_components=3)
    pca.fit(pts_c)
    axis = pca.components_[0]          # crack elongation direction
    perp = pca.components_[1]          # width direction (2nd principal)

    # Project onto principal axis → sorted skeleton
    proj_main = pts_c @ axis
    order     = np.argsort(proj_main)
    skel_proj = proj_main[order]

    # Length = span along main axis (approx arc length for near-linear cracks)
    length_m = float(skel_proj[-1] - skel_proj[0])

    # Width estimation: for each point, find nearest neighbors in perp plane
    # and compute half-span (analogous to EDT max in 2D paper)
    proj_perp = pts_c @ perp
    # Sliding-window width along crack (divide into 10 bins)
    n_bins  = min(10, len(pts) // 3)
    widths  = []
    if n_bins > 0:
        bins = np.array_split(np.argsort(proj_main), n_bins)
        for b in bins:
            if len(b) < 2:
                continue
            w = proj_perp[b].max() - proj_perp[b].min()
            widths.append(w)

    width_mean = float(np.mean(widths)) if widths else 0.0
    width_max  = float(np.max(widths))  if widths else 0.0

    # Sorted skeleton points along principal axis — used for LINE_STRIP marker
    proj_main = pts_c @ axis
    order = np.argsort(proj_main)
    skeleton_pts = (pts_c[order] + centroid).tolist()  # back to world coords

    return {
        'centroid':        centroid.tolist(),
        'length_m':        round(length_m, 4),
        'width_mean_m':    round(width_mean, 4),
        'width_max_m':     round(width_max, 4),
        'orientation_vec': axis.tolist(),
        'n_points':        int(len(pts)),
        'skeleton_pts':    skeleton_pts,
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class CrackMeasurementNode(Node):

    def __init__(self):
        super().__init__('crack_measurement_node')

        self.declare_parameter('dbscan_eps', 0.02)         # 2 cm cluster radius
        self.declare_parameter('dbscan_min_samples', 5)
        self.declare_parameter('min_crack_length_m', 0.005) # 5 mm min crack

        self._eps     = self.get_parameter('dbscan_eps').value
        self._min_s   = self.get_parameter('dbscan_min_samples').value
        self._min_len = self.get_parameter('min_crack_length_m').value

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(
            PointCloud2, '/crack/crack_cloud', self.cloud_cb, qos_be)

        self.pub_meas   = self.create_publisher(String, '/crack/measurements', 10)
        if HAS_VIZ:
            self.pub_markers = self.create_publisher(
                MarkerArray, '/crack/markers', 10)

        self.get_logger().info('CrackMeasurementNode ready.')

    def cloud_cb(self, msg: PointCloud2):
        xyz = pc2_xyz(msg)
        if len(xyz) < self._min_s:
            return

        # Cluster crack points into individual cracks
        db = DBSCAN(eps=self._eps, min_samples=self._min_s).fit(xyz)

        cracks = []
        for label in set(db.labels_):
            if label == -1:
                continue
            pts   = xyz[db.labels_ == label]
            stats = measure_crack_segment(pts)
            if not stats:
                continue
            if stats.get('length_m', 0) < self._min_len:
                continue
            stats['id'] = int(label)
            cracks.append(stats)

        if not cracks:
            return

        payload = {
            'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'frame': msg.header.frame_id,
            'n_cracks': len(cracks),
            'cracks': cracks,
        }
        out = String()
        out.data = json.dumps(payload)
        self.pub_meas.publish(out)

        # Log summary
        for c in cracks:
            self.get_logger().info(
                f"Crack {c['id']}: L={c['length_m']*1000:.1f} mm  "
                f"W_mean={c['width_mean_m']*1000:.2f} mm  "
                f"W_max={c['width_max_m']*1000:.2f} mm  "
                f"pts={c['n_points']}"
            )

        if HAS_VIZ:
            self._publish_markers(cracks, msg.header)

    def _publish_markers(self, cracks, header):
        ma = MarkerArray()

        # First, delete all old markers to avoid stale geometry from
        # previous frames with different crack counts/IDs
        del_all = Marker()
        del_all.header = header
        del_all.ns = 'crack_line'
        del_all.id = 0
        del_all.action = Marker.DELETEALL
        ma.markers.append(del_all)

        for c in cracks:
            cen = c['centroid']
            skeleton = c.get('skeleton_pts', [])

            if len(skeleton) >= 2:
                # LINE_STRIP through actual sorted crack skeleton points —
                # connects all points in sequence, giving a continuous crack
                # trace that matches the real crack geometry rather than
                # a PCA arrow approximation or disconnected point cloud dots
                line = Marker()
                line.header = header
                line.ns = 'crack_line'
                line.id = c['id']
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.004  # line width in meters
                line.color.r = 1.0
                line.color.g = 0.1
                line.color.b = 0.1
                line.color.a = 1.0
                line.points = [
                    Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for p in skeleton
                ]
                ma.markers.append(line)

            # Text label with dimensions
            t = Marker()
            t.header = header
            t.ns = 'crack_label'
            t.id = c['id'] + 10000
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(cen[0])
            t.pose.position.y = float(cen[1])
            t.pose.position.z = float(cen[2]) + 0.05
            t.scale.z = 0.02
            t.color.r = 1.0; t.color.g = 1.0; t.color.b = 0.0; t.color.a = 1.0
            t.text = (f"L:{c['length_m']*1000:.0f}mm "
                      f"W:{c['width_mean_m']*1000:.1f}mm")
            ma.markers.append(t)

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = CrackMeasurementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
