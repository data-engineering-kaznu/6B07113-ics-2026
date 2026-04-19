import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    radius: int
    score: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic ball detection for Lab 12"
    )
    parser.add_argument(
        "--video",
        default="../5275695-uhd_2160_4096_25fps.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        default="output_auto_detected.mp4",
        help="Path to output annotated video",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize frames to this width for faster processing",
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=10,
        help="Minimum ball radius after resize",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show the combined detection mask in a separate window",
    )
    return parser.parse_args()


def resize_frame(frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / float(width)
    new_size = (max_width, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def clamp_bbox(bbox, shape):
    x, y, w, h = [int(v) for v in bbox]
    frame_h, frame_w = shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def expand_bbox(bbox, shape, factor=2.2):
    x, y, w, h = bbox
    cx, cy = bbox_center(bbox)
    half_w = int((w * factor) / 2)
    half_h = int((h * factor) / 2)
    return clamp_bbox((cx - half_w, cy - half_h, 2 * half_w, 2 * half_h), shape)


def orange_mask(hsv):
    lower_1 = np.array([4, 80, 50], dtype=np.uint8)
    upper_1 = np.array([24, 255, 255], dtype=np.uint8)

    lower_2 = np.array([0, 60, 35], dtype=np.uint8)
    upper_2 = np.array([8, 255, 220], dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_1, upper_1)
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)
    return cv2.bitwise_or(mask_1, mask_2)


def prepare_mask(color_mask, motion_mask=None):
    mask = color_mask.copy()
    if motion_mask is not None:
        mask = cv2.bitwise_and(mask, motion_mask)

    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_big = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
    mask = cv2.dilate(mask, kernel_small, iterations=1)
    return mask


def score_contour(contour, offset_x, offset_y, target_center=None):
    area = cv2.contourArea(contour)
    if area < 40:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.35:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    radius = int(radius)
    if radius < 10 or radius > 80:
        return None

    center_x = int(cx) + offset_x
    center_y = int(cy) + offset_y

    distance_penalty = 0.0
    if target_center is not None:
        distance_penalty = np.hypot(center_x - target_center[0], center_y - target_center[1]) * 1.8

    score = area * circularity * 2.0 + radius * 8.0 - distance_penalty
    bbox = (center_x - radius, center_y - radius, radius * 2, radius * 2)

    return Detection(
        bbox=bbox,
        center=(center_x, center_y),
        radius=radius,
        score=score,
    )


def detect_ball(frame, bg_subtractor, previous_bbox=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color = orange_mask(hsv)

    fg = bg_subtractor.apply(frame)
    _, motion = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    target_center = bbox_center(previous_bbox) if previous_bbox is not None else None

    masks_to_try = []
    masks_to_try.append(("motion+color", prepare_mask(color, motion), None))

    if previous_bbox is not None:
        local_box = expand_bbox(previous_bbox, frame.shape)
        masks_to_try.append(("local color", prepare_mask(color), local_box))

    masks_to_try.append(("global color", prepare_mask(color), None))

    for mode, mask, region in masks_to_try:
        if region is not None:
            x, y, w, h = region
            mask_view = mask[y : y + h, x : x + w]
            offset_x, offset_y = x, y
        else:
            mask_view = mask
            offset_x, offset_y = 0, 0

        contours, _ = cv2.findContours(mask_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            detection = score_contour(contour, offset_x, offset_y, target_center=target_center)
            if detection is None:
                continue
            if best is None or detection.score > best.score:
                best = detection

        if best is not None:
            return best, mask, mode

    return None, prepare_mask(color, motion), "not found"


def smooth_bbox(new_bbox, old_bbox, alpha=0.65):
    if old_bbox is None:
        return new_bbox

    smoothed = []
    for new_value, old_value in zip(new_bbox, old_bbox):
        value = int(alpha * new_value + (1.0 - alpha) * old_value)
        smoothed.append(value)
    return tuple(smoothed)


def draw_overlay(frame, bbox, label, status):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        frame,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    video_path = (base_dir / args.video).resolve() if not Path(args.video).is_absolute() else Path(args.video)
    output_path = (base_dir / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read the first frame from the video.")

    first_frame = resize_frame(first_frame, args.max_width)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first_frame.shape[1], first_frame.shape[0]),
    )

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=60,
        varThreshold=24,
        detectShadows=False,
    )

    previous_bbox = None
    frame_index = 0

    while True:
        if frame_index == 0:
            frame = first_frame.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame(frame, args.max_width)

        detection, mask, mode = detect_ball(frame, bg_subtractor, previous_bbox=previous_bbox)

        if detection is not None:
            bbox = clamp_bbox(detection.bbox, frame.shape)
            bbox = smooth_bbox(bbox, previous_bbox)
            previous_bbox = bbox
            status = f"Auto detection: {mode}"
            label = "Ball auto detector"
        else:
            bbox = previous_bbox
            status = "Ball not found"
            label = "Ball auto detector"

        if bbox is not None:
            draw_overlay(frame, bbox, label, status)
        else:
            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        cv2.imshow("Lab 12 - Automatic Ball Detection", frame)

        if args.show_mask:
            cv2.imshow("Auto Detection Mask", mask)

        frame_index += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
