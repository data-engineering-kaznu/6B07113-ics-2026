from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


FRAME_HEIGHT = 180
FRAME_WIDTH = 280
N_FRAMES = 70
OUTPUT_DIR = Path(__file__).resolve().parent


def generate_frame(
    center: tuple[float, float],
    radius: int,
    distractor_center: tuple[float, float],
    distractor_radius: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 35, dtype=np.uint8)
    noise = rng.integers(0, 18, size=frame.shape, dtype=np.uint8)
    frame = np.clip(frame + noise, 0, 255)

    yy, xx = np.ogrid[:FRAME_HEIGHT, :FRAME_WIDTH]

    target_mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    frame[target_mask] = np.array([220, 40, 40], dtype=np.uint8)

    distractor_mask = (
        (xx - distractor_center[0]) ** 2 + (yy - distractor_center[1]) ** 2
        <= distractor_radius**2
    )
    frame[distractor_mask] = np.array([40, 180, 70], dtype=np.uint8)

    return frame


def generate_video(random_state: int = 42) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    rng = np.random.default_rng(random_state)
    frames: list[np.ndarray] = []
    ground_truth: list[tuple[int, int]] = []

    for frame_index in range(N_FRAMES):
        center_x = 30 + frame_index * 3
        center_y = 35 + frame_index * 1.5 + 10 * np.sin(frame_index / 7.0)
        distractor_x = 220 - frame_index
        distractor_y = 130 + 8 * np.cos(frame_index / 6.0)

        center = (center_x, center_y)
        distractor = (distractor_x, distractor_y)
        frames.append(
            generate_frame(
                center,
                radius=14,
                distractor_center=distractor,
                distractor_radius=16,
                rng=rng,
            )
        )
        ground_truth.append((int(round(center_x)), int(round(center_y))))

    return frames, ground_truth


def detect_red_object(frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None]:
    red_mask = (
        (frame[:, :, 0] > 170)
        & (frame[:, :, 1] < 90)
        & (frame[:, :, 2] < 90)
    )

    labeled, count = ndimage.label(red_mask)
    if count == 0:
        return None, None

    objects = ndimage.find_objects(labeled)
    best_slice = None
    best_area = 0
    for object_slice in objects:
        if object_slice is None:
            continue
        y_slice, x_slice = object_slice
        area = (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)
        if area > best_area:
            best_area = area
            best_slice = object_slice

    if best_slice is None:
        return None, None

    y_slice, x_slice = best_slice
    x, y = x_slice.start, y_slice.start
    width = x_slice.stop - x_slice.start
    height = y_slice.stop - y_slice.start
    centroid = (x + width // 2, y + height // 2)
    return (x, y, width, height), centroid


def track_object(
    frames: list[np.ndarray],
) -> tuple[list[tuple[int, int, int, int] | None], list[tuple[int, int] | None]]:
    boxes: list[tuple[int, int, int, int] | None] = []
    trajectory: list[tuple[int, int] | None] = []

    previous_center = None
    for frame in frames:
        box, center = detect_red_object(frame)
        if center is not None and previous_center is not None:
            distance = np.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
            if distance > 35:
                center = previous_center
                box = boxes[-1]

        boxes.append(box)
        trajectory.append(center)
        if center is not None:
            previous_center = center

    return boxes, trajectory


def save_detection_preview(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(frame)
    if box is not None:
        x, y, width, height = box
        rectangle = plt.Rectangle((x, y), width, height, fill=False, color="yellow", linewidth=2)
        ax.add_patch(rectangle)
    ax.set_title("Detected object on the first frame")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_trajectory_plot(
    frames: list[np.ndarray],
    trajectory: list[tuple[int, int] | None],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(frames[-1])

    valid_points = [point for point in trajectory if point is not None]
    xs = [point[0] for point in valid_points]
    ys = [point[1] for point in valid_points]
    ax.plot(xs, ys, color="cyan", linewidth=2, marker="o", markersize=3)

    ax.set_title("Tracked object trajectory")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_tracking_gif(
    frames: list[np.ndarray],
    boxes: list[tuple[int, int, int, int] | None],
    trajectory: list[tuple[int, int] | None],
    output_path: Path,
) -> None:
    rendered_frames: list[Image.Image] = []
    history: list[tuple[int, int]] = []

    for frame, box, center in zip(frames, boxes, trajectory):
        image = Image.fromarray(frame.copy())
        canvas = image.load()

        if center is not None:
            history.append(center)

        if box is not None:
            x, y, width, height = box
            for px in range(x, x + width):
                if 0 <= px < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT:
                    canvas[px, y] = (255, 255, 0)
                if 0 <= px < FRAME_WIDTH and 0 <= y + height - 1 < FRAME_HEIGHT:
                    canvas[px, y + height - 1] = (255, 255, 0)
            for py in range(y, y + height):
                if 0 <= x < FRAME_WIDTH and 0 <= py < FRAME_HEIGHT:
                    canvas[x, py] = (255, 255, 0)
                if 0 <= x + width - 1 < FRAME_WIDTH and 0 <= py < FRAME_HEIGHT:
                    canvas[x + width - 1, py] = (255, 255, 0)

        for point_x, point_y in history:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px = point_x + dx
                    py = point_y + dy
                    if 0 <= px < FRAME_WIDTH and 0 <= py < FRAME_HEIGHT:
                        canvas[px, py] = (80, 255, 255)

        rendered_frames.append(image)

    rendered_frames[0].save(
        output_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=80,
        loop=0,
    )


def main() -> None:
    frames, ground_truth = generate_video()
    boxes, trajectory = track_object(frames)

    valid_pairs = [
        (predicted, actual)
        for predicted, actual in zip(trajectory, ground_truth)
        if predicted is not None
    ]
    mean_error = float(
        np.mean(
            [
                np.hypot(predicted[0] - actual[0], predicted[1] - actual[1])
                for predicted, actual in valid_pairs
            ]
        )
    )
    tracked_ratio = len(valid_pairs) / len(ground_truth)

    detection_image = OUTPUT_DIR / "detection_preview.png"
    trajectory_image = OUTPUT_DIR / "trajectory.png"
    animation_file = OUTPUT_DIR / "tracking_demo.gif"

    save_detection_preview(frames[0], boxes[0], detection_image)
    save_trajectory_plot(frames, trajectory, trajectory_image)
    save_tracking_gif(frames, boxes, trajectory, animation_file)

    print("Lab 12. Object detection and tracking demo")
    print(f"Frames processed: {len(frames)}")
    print(f"Tracked frames ratio: {tracked_ratio:.3f}")
    print(f"Mean tracking error: {mean_error:.2f} px")
    print(f"Detection preview saved to: {detection_image.name}")
    print(f"Trajectory image saved to: {trajectory_image.name}")
    print(f"Animation saved to: {animation_file.name}")


if __name__ == "__main__":
    main()
