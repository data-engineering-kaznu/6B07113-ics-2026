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
    source: str


@dataclass
class BallModel:
    lower: np.ndarray
    upper: np.ndarray
    hue: int
    saturation: int
    value: int
    radius: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid ball tracking: manual or automatic init + detector + tracker + Kalman"
    )
    parser.add_argument(
        "--video",
        default="../5275695-uhd_2160_4096_25fps.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        default="output_hybrid_tracked.mp4",
        help="Path to output video",
    )
    parser.add_argument(
        "--init-mode",
        default="manual",
        choices=["manual", "auto"],
        help="How to initialize the ball position",
    )
    parser.add_argument(
        "--tracker",
        default="CSRT",
        choices=["CSRT", "KCF", "MOSSE"],
        help="OpenCV tracker to use as fallback",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize frames to this width for faster processing",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show detector mask in a separate window",
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

    raise AttributeError(f"Tracker {name} is unavailable. Install opencv-contrib-python.")


def create_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.4
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


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


def tracker_bbox(bbox):
    x, y, w, h = bbox
    return int(x), int(y), int(w), int(h)


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def expand_bbox(bbox, shape, factor=2.4):
    x, y, w, h = bbox
    cx, cy = bbox_center(bbox)
    half_w = max(10, int((w * factor) / 2))
    half_h = max(10, int((h * factor) / 2))
    return clamp_bbox((cx - half_w, cy - half_h, 2 * half_w, 2 * half_h), shape)


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


def learn_ball_model(frame, roi):
    x, y, w, h = clamp_bbox(roi, frame.shape)
    roi_frame = frame[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (w // 2, h // 2),
        (max(1, int(w * 0.32)), max(1, int(h * 0.32))),
        0,
        0,
        360,
        255,
        -1,
    )

    pixels = hsv[mask == 255]
    pixels = pixels[pixels[:, 1] > 30]
    pixels = pixels[pixels[:, 2] > 30]
    if len(pixels) == 0:
        raise ValueError("Unable to learn the ball color from ROI.")

    hue = pixels[:, 0].astype(np.int32)
    sat = pixels[:, 1]
    val = pixels[:, 2]

    median_hue = int(np.median(hue))
    lower_h = (median_hue - 16) % 180
    upper_h = (median_hue + 16) % 180
    lower_s = int(max(55, np.percentile(sat, 20) - 15))
    lower_v = int(max(45, np.percentile(val, 20) - 15))
    upper_s = int(min(255, np.percentile(sat, 99) + 10))
    upper_v = int(min(255, np.percentile(val, 99) + 10))

    return BallModel(
        lower=np.array([lower_h, lower_s, lower_v], dtype=np.uint8),
        upper=np.array([upper_h, upper_s, upper_v], dtype=np.uint8),
        hue=median_hue,
        saturation=int(np.median(sat)),
        value=int(np.median(val)),
        radius=float(max(w, h) / 2.0),
    )


def default_ball_model():
    return BallModel(
        lower=np.array([4, 80, 45], dtype=np.uint8),
        upper=np.array([24, 255, 255], dtype=np.uint8),
        hue=14,
        saturation=150,
        value=130,
        radius=24.0,
    )


def build_detector_mask(frame, model, bg_subtractor=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = hue_mask(hsv, model.lower, model.upper)

    motion_mask = None
    if bg_subtractor is not None:
        fg = bg_subtractor.apply(frame)
        _, motion_mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_big = np.ones((5, 5), dtype=np.uint8)

    mask = color_mask
    if motion_mask is not None:
        combined = cv2.bitwise_and(color_mask, motion_mask)
        if cv2.countNonZero(combined) > 25:
            mask = combined

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
    mask = cv2.dilate(mask, kernel_small, iterations=1)
    return mask, hsv


def score_candidate(contour, hsv_roi, offset_x, offset_y, model, target_center=None):
    area = cv2.contourArea(contour)
    if area < 35:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.4:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    radius = int(radius)
    if radius < 8 or radius > 90:
        return None

    contour_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    pixels = hsv_roi[contour_mask == 255]
    if len(pixels) == 0:
        return None

    mean_hue = float(np.mean(pixels[:, 0]))
    mean_sat = float(np.mean(pixels[:, 1]))
    mean_val = float(np.mean(pixels[:, 2]))
    if mean_sat < 55 or mean_val < 50:
        return None

    center_x = int(cx) + offset_x
    center_y = int(cy) + offset_y

    hue_diff = abs(mean_hue - model.hue)
    hue_diff = min(hue_diff, 180 - hue_diff)
    radius_diff = abs(radius - model.radius)
    distance_penalty = 0.0
    if target_center is not None:
        distance_penalty = np.hypot(center_x - target_center[0], center_y - target_center[1]) * 1.7

    score = (
        area * circularity * 2.4
        + mean_sat * 1.0
        + mean_val * 0.4
        - hue_diff * 9.0
        - radius_diff * 2.0
        - distance_penalty
    )

    bbox = (
        center_x - radius,
        center_y - radius,
        radius * 2,
        radius * 2,
    )
    return Detection(
        bbox=bbox,
        center=(center_x, center_y),
        radius=radius,
        score=score,
        source="detector",
    )


def find_best_detection(frame, model, mask, hsv, predicted_center=None, search_bbox=None, label="global"):
    if search_bbox is not None:
        x, y, w, h = clamp_bbox(search_bbox, frame.shape)
        mask_roi = mask[y : y + h, x : x + w]
        hsv_roi = hsv[y : y + h, x : x + w]
        offset_x, offset_y = x, y
    else:
        mask_roi = mask
        hsv_roi = hsv
        offset_x, offset_y = 0, 0

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        detection = score_candidate(
            contour,
            hsv_roi,
            offset_x,
            offset_y,
            model,
            target_center=predicted_center,
        )
        if detection is None:
            continue
        detection.source = label
        if best is None or detection.score > best.score:
            best = detection
    return best


def update_model(model, frame, bbox):
    x, y, w, h = clamp_bbox(bbox, frame.shape)
    roi = frame[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    pixels = pixels[pixels[:, 1] > 30]
    pixels = pixels[pixels[:, 2] > 30]
    if len(pixels) == 0:
        return model

    hue = int(np.median(pixels[:, 0]))
    sat = int(np.median(pixels[:, 1]))
    val = int(np.median(pixels[:, 2]))

    alpha = 0.15
    mixed_hue = int((1.0 - alpha) * model.hue + alpha * hue)
    mixed_sat = int((1.0 - alpha) * model.saturation + alpha * sat)
    mixed_val = int((1.0 - alpha) * model.value + alpha * val)
    mixed_radius = (1.0 - alpha) * model.radius + alpha * (max(w, h) / 2.0)

    lower_h = (mixed_hue - 16) % 180
    upper_h = (mixed_hue + 16) % 180
    lower_s = max(50, mixed_sat - 70)
    lower_v = max(40, mixed_val - 70)
    upper_s = min(255, mixed_sat + 85)
    upper_v = min(255, mixed_val + 85)

    return BallModel(
        lower=np.array([lower_h, lower_s, lower_v], dtype=np.uint8),
        upper=np.array([upper_h, upper_s, upper_v], dtype=np.uint8),
        hue=mixed_hue,
        saturation=mixed_sat,
        value=mixed_val,
        radius=mixed_radius,
    )


def smooth_bbox(new_bbox, old_bbox, alpha=0.72):
    if old_bbox is None:
        return new_bbox
    return tuple(int(alpha * n + (1.0 - alpha) * o) for n, o in zip(new_bbox, old_bbox))


def draw_overlay(frame, bbox, label, status):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        frame,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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


def initialize_manual(frame, tracker_name, kalman):
    roi = cv2.selectROI("Select Ball", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Ball")
    if roi[2] == 0 or roi[3] == 0:
        raise RuntimeError("ROI selection was cancelled.")

    model = learn_ball_model(frame, roi)
    bbox = clamp_bbox(roi, frame.shape)
    center = bbox_center(bbox)

    kalman.statePost = np.array([[center[0]], [center[1]], [0], [0]], dtype=np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    tracker = create_tracker(tracker_name)
    tracker.init(frame, tracker_bbox(bbox))
    return bbox, model, tracker, "manual init"


def initialize_auto(frame, tracker_name, kalman, bg_subtractor):
    model = default_ball_model()

    for _ in range(5):
        bg_subtractor.apply(frame)

    mask, hsv = build_detector_mask(frame, model, bg_subtractor)
    detection = find_best_detection(frame, model, mask, hsv, label="auto init")
    if detection is None:
        raise RuntimeError("Automatic initialization failed: ball not found on the first frame.")

    bbox = clamp_bbox(detection.bbox, frame.shape)
    center = bbox_center(bbox)
    model = update_model(model, frame, bbox)

    kalman.statePost = np.array([[center[0]], [center[1]], [0], [0]], dtype=np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    tracker = create_tracker(tracker_name)
    tracker.init(frame, tracker_bbox(bbox))
    return bbox, model, tracker, "auto init"


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
        history=70,
        varThreshold=20,
        detectShadows=False,
    )
    kalman = create_kalman()

    if args.init_mode == "manual":
        current_bbox, model, tracker, init_label = initialize_manual(first_frame, args.tracker, kalman)
    else:
        current_bbox, model, tracker, init_label = initialize_auto(first_frame, args.tracker, kalman, bg_subtractor)

    frame_index = 0
    lost_counter = 0
    tracker_refresh_counter = 0

    while True:
        if frame_index == 0:
            frame = first_frame.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame(frame, args.max_width)

        prediction = kalman.predict()
        predicted_center = (int(prediction[0]), int(prediction[1]))

        mask, hsv = build_detector_mask(frame, model, bg_subtractor)

        local_detection = None
        if current_bbox is not None:
            local_detection = find_best_detection(
                frame,
                model,
                mask,
                hsv,
                predicted_center=predicted_center,
                search_bbox=expand_bbox(current_bbox, frame.shape),
                label="local detector",
            )

        global_detection = find_best_detection(
            frame,
            model,
            mask,
            hsv,
            predicted_center=predicted_center,
            label="global detector",
        )

        tracker_success = False
        tracker_bbox_value = None
        if tracker is not None:
            tracker_success, tracked = tracker.update(frame)
            if tracker_success:
                tracker_bbox_value = clamp_bbox(tracked, frame.shape)

        chosen_detection = local_detection if local_detection is not None else global_detection
        label = "Hybrid tracker"
        status = init_label if frame_index == 0 else "Tracking stable"

        if chosen_detection is not None:
            bbox = clamp_bbox(chosen_detection.bbox, frame.shape)

            if tracker_bbox_value is not None:
                tracker_center = bbox_center(tracker_bbox_value)
                detection_distance = np.hypot(
                    chosen_detection.center[0] - tracker_center[0],
                    chosen_detection.center[1] - tracker_center[1],
                )
                if detection_distance < max(50, chosen_detection.radius * 3):
                    blended = (
                        int((bbox[0] + tracker_bbox_value[0]) / 2),
                        int((bbox[1] + tracker_bbox_value[1]) / 2),
                        int((bbox[2] + tracker_bbox_value[2]) / 2),
                        int((bbox[3] + tracker_bbox_value[3]) / 2),
                    )
                    bbox = clamp_bbox(blended, frame.shape)
                    status = f"Detector + {args.tracker}"
                else:
                    status = chosen_detection.source
            else:
                status = chosen_detection.source

            bbox = smooth_bbox(bbox, current_bbox)
            measurement = np.array([[np.float32(bbox_center(bbox)[0])], [np.float32(bbox_center(bbox)[1])]])
            kalman.correct(measurement)
            model = update_model(model, frame, bbox)
            current_bbox = bbox
            lost_counter = 0
            tracker_refresh_counter += 1

            if tracker is None or tracker_refresh_counter >= 8:
                tracker = create_tracker(args.tracker)
                tracker.init(frame, tracker_bbox(current_bbox))
                tracker_refresh_counter = 0

        elif tracker_bbox_value is not None:
            bbox = smooth_bbox(tracker_bbox_value, current_bbox)
            measurement = np.array([[np.float32(bbox_center(bbox)[0])], [np.float32(bbox_center(bbox)[1])]])
            kalman.correct(measurement)
            current_bbox = bbox
            lost_counter = 0
            status = f"{args.tracker} fallback"

        else:
            lost_counter += 1
            if current_bbox is not None and lost_counter <= 6:
                px, py = predicted_center
                radius = max(12, int(model.radius))
                bbox = clamp_bbox((px - radius, py - radius, radius * 2, radius * 2), frame.shape)
                current_bbox = bbox
                status = "Kalman prediction"
            else:
                current_bbox = None
                tracker = None
                status = "Ball lost"

        if current_bbox is not None:
            draw_overlay(frame, current_bbox, label, status)
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
        cv2.imshow("Lab 12 - Hybrid Ball Tracking", frame)
        if args.show_mask:
            cv2.imshow("Hybrid Detector Mask", mask)

        frame_index += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
