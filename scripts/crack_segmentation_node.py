#!/usr/bin/env python3
"""
crack_segmentation_node.py

Stage 1: Per-frame crack segmentation using a REAL pretrained crack model.

Model: UNet16 (VGG16 encoder) from khanhha/crack_segmentation, trained on
~11,200 images merged from 12 public crack datasets (CRACK500, DeepCrack,
GAPs384, CFD, AEL, CrackTree200, etc.), explicitly designed to be robust
to "crack in large context" scenes (people, equipment, clutter in frame) —
exactly our deployment scenario.

This REPLACES the earlier untrained-DeepLab + SAM pipeline, which either
hallucinated masks across the whole image (random-init DeepLab) or, in the
classical-heuristic fallback, locked onto the wrong dark linear feature
(block seams instead of the actual crack). A real trained model is the
correct fix, not further heuristic tuning.

Subscribes:  /left_camera/image  (sensor_msgs/Image)
Publishes:   /crack/mask           (sensor_msgs/Image, mono8)
             /crack/mask_viz       (sensor_msgs/Image, bgr8, debug overlay)

Requires the khanhha/crack_segmentation repo cloned locally so we can import
its UNet16 architecture definition directly (matches the checkpoint exactly):
    git clone https://github.com/khanhha/crack_segmentation.git /mnt/ssd/crack_segmentation
and the pretrained checkpoint downloaded to:
    /mnt/ssd/models/unet_vgg16_crack.pt
"""

import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model state — loaded once on node startup
# ---------------------------------------------------------------------------
_model = None
_device = None
_input_size = (448, 448)   # matches unet_transfer.input_size in upstream repo


def _load_model(repo_path: str, ckpt_path: str, device: str):
    global _model, _device
    if _model is not None:
        return

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from unet.unet_transfer import UNet16   # noqa: import from cloned repo

    _device = torch.device(device)
    _model = UNet16(pretrained=True)

    checkpoint = torch.load(ckpt_path, map_location=_device)
    if 'model' in checkpoint:
        _model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        _model.load_state_dict(checkpoint['state_dict'])
    else:
        # Some checkpoints are a bare state_dict with no wrapper key
        _model.load_state_dict(checkpoint)

    _model.to(_device).eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_CHANNEL_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_CHANNEL_STDS  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """Resize to model input size, normalize with ImageNet stats (matches
    upstream train_tfms = ToTensor() + Normalize(channel_means, channel_stds))."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, _input_size, interpolation=cv2.INTER_AREA)
    img_f = img_resized.astype(np.float32) / 255.0
    img_f = (img_f - _CHANNEL_MEANS) / _CHANNEL_STDS
    tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(_device)


def crack_infer(img_bgr: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Run the trained UNet16 crack model.
    Returns binary mask (H, W) uint8 at the ORIGINAL image resolution.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    x = _preprocess(img_bgr)

    with torch.no_grad():
        logits = _model(x)                      # (1, 1, 448, 448)
        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()

    prob_full = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = (prob_full > threshold).astype(np.uint8)
    return mask, prob_full


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class CrackSegmentationNode(Node):

    def __init__(self):
        super().__init__('crack_segmentation_node')

        self.declare_parameter('repo_path', '/mnt/ssd/crack_segmentation')
        self.declare_parameter('ckpt_path', '/mnt/ssd/models/unet_vgg16_crack.pt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('threshold', 0.3)
        self.declare_parameter('publish_viz', True)
        self.declare_parameter('inference_skip', 1)   # process every N-th frame

        repo_path = self.get_parameter('repo_path').value
        ckpt_path = self.get_parameter('ckpt_path').value
        device    = self.get_parameter('device').value
        self._threshold = self.get_parameter('threshold').value

        self.get_logger().info(f'Loading UNet16 crack model on {device}...')
        _load_model(repo_path, ckpt_path, device)
        self.get_logger().info('Model loaded.')

        self.bridge = CvBridge()
        self._frame_count = 0
        self._skip = self.get_parameter('inference_skip').value
        self._pub_viz = self.get_parameter('publish_viz').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.sub_img = self.create_subscription(
            Image, '/left_camera/image', self.image_cb, qos)

        self.pub_mask = self.create_publisher(Image, '/crack/mask', 10)
        if self._pub_viz:
            self.pub_viz = self.create_publisher(Image, '/crack/mask_viz', 10)

    def image_cb(self, msg: Image):
        self._frame_count += 1
        if self._frame_count % self._skip != 0:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        mask, prob = crack_infer(bgr, threshold=self._threshold)

        # Publish binary mask (mono8, values 0/255)
        mask_out = (mask * 255).astype(np.uint8)
        mask_msg = self.bridge.cv2_to_imgmsg(mask_out, encoding='mono8')
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        if mask.sum() > 0:
            self.get_logger().info(
                f'Crack pixels detected: {int(mask.sum())} '
                f'(max prob: {prob.max():.3f})'
            )

        # Publish visualization overlay
        if self._pub_viz:
            viz = bgr.copy()
            viz[mask > 0] = viz[mask > 0] * 0.4 + np.array([0, 0, 220]) * 0.6
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz, contours, -1, (0, 255, 0), 1)
            viz_msg = self.bridge.cv2_to_imgmsg(viz.astype(np.uint8), encoding='bgr8')
            viz_msg.header = msg.header
            self.pub_viz.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CrackSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
