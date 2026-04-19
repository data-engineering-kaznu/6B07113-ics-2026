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
        description="Lab 12: ball detection and tracking with OpenCV"
    )
    parser.add_argument(
        "--video",
        default="../5275695-uhd_2160_4096_25fps.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--tracker",
        default="CSRT",
        choices=["CSRT", "KCF", "MOSSE"],
        help="OpenCV tracker to use",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize frames to this width for faster processing",
    )
    parser.add_argument(
        "--output",
        default="output_tracked.mp4",
        help="Path to output video with annotations",
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=8,
        help="Minimum detected ball radius in pixels after resize",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=10,
        help="How often to refresh the tracker from detector result",
    )
    return parser.parse_args()


def create_tracker(name: str):
    factories = {
        "CSRT": ("TrackerCSRT_create", "legacy"),
        "KCF": ("TrackerKCF_create", "legacy"),
        "MOSSE": ("TrackerMOSSE_create", "legacy"),
    }
    method_name, legacy_namespace = factories[name]

    if hasattr(cv2, method_name):
        return getattr(cv2, method_name)()

    legacy = getattr(cv2, legacy_namespace, None)
    if legacy is not None and hasattr(legacy, method_name):
        return getattr(legacy, method_name)()

    raise AttributeError(
        f"Tracker {name} is unavailable. Install opencv-contrib-python."
    )


def resize_frame(frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0

    scale = max_width / float(width)
    new_size = (max_width, int(height * scale))
    resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def clamp_bbox(bbox, shape):
    x, y, w, h = [int(v) for v in bbox]
    frame_h, frame_w = shape[:2]

    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def tracker_bbox(bbox):
    x, y, w, h = bbox
    return int(x), int(y), int(w), int(h)


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def expand_bbox(bbox, frame_shape, factor=1.8):
    x, y, w, h = bbox
    cx, cy = bbox_center(bbox)
    half_w = int((w * factor) / 2)
    half_h = int((h * factor) / 2)
    expanded = (
        cx - half_w,
        cy - half_h,
        2 * half_w,
        2 * half_h,
    )
    return clamp_bbox(expanded, frame_shape)


def hue_mask(hsv_frame, lower, upper):
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv_frame, lower, upper)

    lower_wrap = np.array([0, lower[1], lower[2]], dtype=np.uint8)
    upper_wrap = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
    lower_main = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
    upper_main = np.array([179, upper[1], upper[2]], dtype=np.uint8)

    mask_a = cv2.inRange(hsv_frame, lower_main, upper_main)
    mask_b = cv2.inRange(hsv_frame, lower_wrap, upper_wrap)
    return cv2.bitwise_or(mask_a, mask_b)


def learn_hsv_bounds(frame, roi):
    x, y, w, h = clamp_bbox(roi, frame.shape)
    roi_frame = frame[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape(-1, 3)
    pixels = pixels[pixels[:, 1] > 30]
    pixels = pixels[pixels[:, 2] > 30]

    if len(pixels) == 0:
        raise ValueError("Unable to learn ball color from selected ROI.")

    hue = pixels[:, 0].astype(np.int32)
    sat = pixels[:, 1]
    val = pixels[:, 2]

    median_hue = int(np.median(hue))
    hue_margin = 15
    lower_h = (median_hue - hue_margin) % 180
    upper_h = (median_hue + hue_margin) % 180

    lower_s = int(max(40, np.percentile(sat, 10) - 20))
    lower_v = int(max(40, np.percentile(val, 10) - 20))
    upper_s = int(min(255, np.percentile(sat, 99) + 10))
    upper_v = int(min(255, np.percentile(val, 99) + 10))

    lower = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
    return lower, upper


def detect_ball(frame, lower, upper, min_radius, search_bbox=None):
    if search_bbox is not None:
        sx, sy, sw, sh = clamp_bbox(search_bbox, frame.shape)
        work = frame[sy : sy + sh, sx : sx + sw]
        offset_x, offset_y = sx, sy
        search_center = (sx + sw // 2, sy + sh // 2)
    else:
        work = frame
        offset_x, offset_y = 0, 0
        h, w = frame.shape[:2]
        search_center = (w // 2, h // 2)

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    mask = hue_mask(hsv, lower, upper)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_detection = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        if radius < min_radius:
            continue

        center_x = int(cx) + offset_x
        center_y = int(cy) + offset_y
        distance = np.hypot(center_x - search_center[0], center_y - search_center[1])
        score = area * circularity - distance * 2.0

        bbox = (
            center_x - radius,
            center_y - radius,
            radius * 2,
            radius * 2,
        )
        detection = Detection(
            bbox=clamp_bbox(bbox, frame.shape),
            center=(center_x, center_y),
            radius=radius,
            score=score,
        )

        if best_detection is None or detection.score > best_detection.score:
            best_detection = detection

    return best_detection, mask


def draw_overlay(frame, bbox, source_label, status_text):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        source_label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read the first frame from the video.")

    first_frame, _ = resize_frame(first_frame, args.max_width)

    roi = cv2.selectROI("Select Ball", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Ball")
    if roi[2] == 0 or roi[3] == 0:
        raise RuntimeError("ROI selection was cancelled.")

    lower, upper = learn_hsv_bounds(first_frame, roi)
    detection, _ = detect_ball(first_frame, lower, upper, args.min_radius, search_bbox=roi)
    initial_bbox = detection.bbox if detection is not None else clamp_bbox(roi, first_frame.shape)

    tracker = create_tracker(args.tracker)
    tracker.init(first_frame, tracker_bbox(initial_bbox))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first_frame.shape[1], first_frame.shape[0]),
    )

    frame_index = 0
    last_bbox = initial_bbox

    while True:
        if frame_index == 0:
            frame = first_frame.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame, _ = resize_frame(frame, args.max_width)

        status_text = "Tracking stable"
        source_label = f"{args.tracker} tracker"

        if frame_index == 0:
            bbox = initial_bbox
        else:
            tracker_success, tracked_bbox = tracker.update(frame)

            if tracker_success:
                bbox = clamp_bbox(tracked_bbox, frame.shape)
                search_bbox = expand_bbox(bbox, frame.shape)
                detection, _ = detect_ball(
                    frame,
                    lower,
                    upper,
                    args.min_radius,
                    search_bbox=search_bbox,
                )

                if detection is not None and frame_index % args.refresh_interval == 0:
                    bbox = detection.bbox
                    tracker = create_tracker(args.tracker)
                    tracker.init(frame, tracker_bbox(bbox))
                    source_label = "Color detector + tracker"
                    status_text = "Tracker refreshed from detector"
            else:
                detection, _ = detect_ball(frame, lower, upper, args.min_radius)
                if detection is None:
                    bbox = last_bbox
                    source_label = "Target lost"
                    status_text = "Detector could not reacquire the ball"
                else:
                    bbox = detection.bbox
                    tracker = create_tracker(args.tracker)
                    tracker.init(frame, tracker_bbox(bbox))
                    source_label = "Color detector recovery"
                    status_text = "Ball reacquired after tracking loss"

        draw_overlay(frame, bbox, source_label, status_text)
        writer.write(frame)
        cv2.imshow("Lab 12 - Ball Detection and Tracking", frame)

        last_bbox = bbox
        frame_index += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
