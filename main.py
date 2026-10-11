#!/usr/bin/env python3
"""Estimate vehicle speed in a fixed-camera video using a calibrated road plane.

This is for measurement and analytics in controlled/private environments.
Validate its readings against an independent reference before relying on them.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--config", required=True, help="Calibration YAML path")
    parser.add_argument("--output", default="annotated.mp4", help="Annotated MP4 path")
    parser.add_argument("--csv", default="speeds.csv", help="Per-frame speed CSV path")
    parser.add_argument("--gate-csv", default="gate-speeds.csv", help="Gate-to-gate speed event CSV path")
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics model name or path")
    parser.add_argument("--confidence", type=float, default=0.20, help="Detection confidence; lower finds smaller distant vehicles")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detection image size; higher preserves distant vehicles")
    parser.add_argument("--tracker", default="traffic_bytetrack.yaml", help="Ultralytics tracker configuration")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames (useful for validation)")
    return parser.parse_args()


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray | None, float, int, dict[str, list[float]], dict | None]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    image = np.asarray(data["image_points"], dtype=np.float32)
    ground = np.asarray(data["ground_points"], dtype=np.float32)
    if image.shape != ground.shape or len(image) < 4:
        raise ValueError("image_points and ground_points must contain at least four matching points")
    matrix, _ = cv2.findHomography(image, ground, method=0)
    if matrix is None:
        raise ValueError("Could not calculate homography; check calibration points")
    polygon = data.get("road_polygon")
    polygon_array = np.asarray(polygon, dtype=np.int32) if polygon else None
    gate_pairs = data.get("gate_pairs_m", {})
    for axis, values in gate_pairs.items():
        if axis not in {"x", "y"} or len(values) != 2:
            raise ValueError("gate_pairs_m must contain x and/or y pairs with two metre values")
    screen_gates = data.get("screen_gates")
    if screen_gates is not None:
        for gate_name in ("gate_a", "gate_b"):
            if gate_name not in screen_gates or len(screen_gates[gate_name]) != 2:
                raise ValueError("screen_gates needs gate_a and gate_b, each with two image points")
        if float(screen_gates.get("distance_ft", 0)) <= 0:
            raise ValueError("screen_gates.distance_ft must be positive")
    return matrix, polygon_array, float(data.get("speed_smoothing_seconds", 1.0)), int(data.get("min_track_age_frames", 8)), gate_pairs, screen_gates


def ground_position(point: tuple[float, float], homography: np.ndarray) -> np.ndarray:
    image_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
    return cv2.perspectiveTransform(image_point, homography)[0, 0]


def in_road(point: tuple[float, float], polygon: np.ndarray | None) -> bool:
    return polygon is None or cv2.pointPolygonTest(polygon, point, False) >= 0


def crossed_gate(previous: float, current: float, gate: float) -> float | None:
    """Return fractional progress through a gate between two readings, if crossed."""
    delta = current - previous
    if delta == 0 or (previous - gate) * (current - gate) > 0:
        return None
    return (gate - previous) / delta


def segment_crossing_fraction(previous: tuple[float, float], current: tuple[float, float], start: np.ndarray, end: np.ndarray) -> tuple[float, int] | None:
    """Return movement fraction and side-change direction when crossing a finite gate segment."""
    direction = end - start
    old_side = float(np.cross(direction, np.asarray(previous) - start))
    new_side = float(np.cross(direction, np.asarray(current) - start))
    if old_side == new_side or old_side * new_side > 0:
        return None
    fraction = old_side / (old_side - new_side)
    crossing = np.asarray(previous) + fraction * (np.asarray(current) - np.asarray(previous))
    length_sq = float(np.dot(direction, direction))
    along = float(np.dot(crossing - start, direction) / length_sq)
    if not 0 <= along <= 1:
        return None
    return fraction, 1 if new_side > old_side else -1


def main() -> None:
    args = parse_args()
    homography, road_polygon, smoothing_seconds, min_age, gate_pairs, screen_gates = load_calibration(args.config)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(args.model)
    history: dict[int, deque[tuple[float, np.ndarray]]] = defaultdict(deque)
    track_ages: dict[int, int] = defaultdict(int)
    gate_first_crossing: dict[tuple[int, str], tuple[float, float]] = {}
    gate_speeds: dict[int, float] = {}
    screen_gate_a: dict[int, tuple[float, int]] = {}
    screen_gate_speeds: dict[int, float] = {}
    rows: list[list[object]] = []
    gate_rows: list[list[object]] = []
    frame_number = 0

    while True:
        if args.max_frames is not None and frame_number >= args.max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_number / fps
        result = model.track(frame, persist=True, classes=[2, 3, 5, 7], conf=args.confidence, imgsz=args.imgsz, tracker=args.tracker, verbose=False)[0]
        if road_polygon is not None:
            cv2.polylines(frame, [road_polygon], True, (255, 200, 0), 2)
        if screen_gates:
            a1, a2 = (tuple(map(int, point)) for point in screen_gates["gate_a"])
            b1, b2 = (tuple(map(int, point)) for point in screen_gates["gate_b"])
            cv2.line(frame, a1, a2, (255, 255, 0), 3)
            cv2.line(frame, b1, b2, (255, 255, 0), 3)
            cv2.putText(frame, "Gate A", a1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Gate B", b1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            for box, track_id, class_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = box
                road_point = (float((x1 + x2) / 2), float(y2))
                if not in_road(road_point, road_polygon):
                    continue
                track_ages[track_id] += 1
                position = ground_position(road_point, homography)
                points = history[track_id]
                if points and screen_gates:
                    previous_time, _, previous_pixel = points[-1]
                    previous_pixel = tuple(previous_pixel)
                    # Older in-memory tuples from no previous frame are not used.
                    if previous_pixel is not None:
                        gate_a = np.asarray(screen_gates["gate_a"], dtype=float)
                        gate_b = np.asarray(screen_gates["gate_b"], dtype=float)
                        crossing_a = segment_crossing_fraction(previous_pixel, road_point, gate_a[0], gate_a[1])
                        crossing_b = segment_crossing_fraction(previous_pixel, road_point, gate_b[0], gate_b[1])
                        if crossing_a:
                            fraction, direction = crossing_a
                            screen_gate_a[track_id] = (previous_time + fraction * (timestamp - previous_time), direction)
                        if crossing_b and track_id in screen_gate_a:
                            fraction, direction = crossing_b
                            gate_a_time, gate_direction = screen_gate_a.pop(track_id)
                            gate_b_time = previous_time + fraction * (timestamp - previous_time)
                            elapsed = gate_b_time - gate_a_time
                            if direction == gate_direction and 0.2 <= elapsed <= 10:
                                distance_ft = float(screen_gates["distance_ft"])
                                # Convert feet/second to kilometres/hour:
                                # 1 ft = 0.3048 m; 1 m/s = 3.6 km/h.
                                gate_speed = distance_ft / elapsed * 0.3048 * 3.6
                                screen_gate_speeds[track_id] = gate_speed
                                gate_rows.append([track_id, "screen_a_to_b", "A", "B", f"{gate_a_time:.3f}", f"{gate_b_time:.3f}", f"{elapsed:.3f}", f"{distance_ft * 0.3048:.3f}", f"{gate_speed:.2f}"])
                points.append((timestamp, position, road_point))
                while len(points) > 1 and timestamp - points[0][0] > smoothing_seconds:
                    points.popleft()
                speed_kmh = 0.0
                if len(points) > 1:
                    dt = timestamp - points[0][0]
                    if dt > 0:
                        speed_kmh = float(np.linalg.norm(position - points[0][1]) / dt * 3.6)

                # Record a speed only after a track crosses both gates on the
                # same physical axis. This avoids using distorted near-camera
                # pixel motion as the distance measurement.
                if len(points) > 1:
                    previous_time, previous_position, _ = points[-2]
                    for axis, pair in gate_pairs.items():
                        index = 0 if axis == "x" else 1
                        current_value, previous_value = position[index], previous_position[index]
                        for gate in pair:
                            fraction = crossed_gate(previous_value, current_value, gate)
                            if fraction is None:
                                continue
                            crossing_time = previous_time + fraction * (timestamp - previous_time)
                            key = (track_id, axis)
                            first = gate_first_crossing.get(key)
                            if first is None or first[0] == gate:
                                gate_first_crossing[key] = (gate, crossing_time)
                                continue
                            elapsed = abs(crossing_time - first[1])
                            distance_m = abs(pair[1] - pair[0])
                            if 0.2 <= elapsed <= 5.0:
                                gate_speed = distance_m / elapsed * 3.6
                                gate_speeds[track_id] = gate_speed
                                gate_rows.append([track_id, axis, f"{first[0]:.3f}", f"{gate:.3f}", f"{elapsed:.3f}", f"{distance_m:.3f}", f"{gate_speed:.2f}"])
                            gate_first_crossing.pop(key, None)
                label_speed = screen_gate_speeds.get(track_id, gate_speeds.get(track_id))
                completed_gate = label_speed is not None
                label = f"#{track_id}  {label_speed:.1f} km/h" if completed_gate else f"#{track_id} measuring"
                color = (0, 220, 0) if track_ages[track_id] >= min_age else (0, 180, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                if completed_gate:
                    scale, thickness, text_color = 1.05, 3, (0, 255, 255)
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                    text_x, text_y = int(x1), max(text_height + baseline + 12, int(y1) - 12)
                    cv2.rectangle(frame, (text_x - 7, text_y - text_height - baseline - 7), (text_x + text_width + 7, text_y + baseline + 7), (0, 0, 0), -1)
                    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)
                else:
                    cv2.putText(frame, label, (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if track_ages[track_id] >= min_age:
                    rows.append([frame_number, f"{timestamp:.3f}", track_id, model.names[class_id], f"{position[0]:.3f}", f"{position[1]:.3f}", f"{speed_kmh:.2f}"])
        writer.write(frame)
        frame_number += 1

    capture.release()
    writer.release()
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([["frame", "timestamp_s", "track_id", "vehicle_type", "ground_x_m", "ground_y_m", "speed_kmh"], *rows])
    with open(args.gate_csv, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([["track_id", "direction", "first_gate", "second_gate", "gate_a_time_s", "gate_b_time_s", "elapsed_s", "gate_distance_m", "speed_kmh"], *gate_rows])
    print(f"Processed {frame_number} frames. Wrote {args.output}, {args.csv}, and {args.gate_csv}.")


if __name__ == "__main__":
    main()
