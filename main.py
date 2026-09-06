from ultralytics import YOLO
import cv2, csv, numpy as np, math
from collections import deque

# ================= SETTINGS =================
VIDEO_PATH = "BellevueIntersection2.mov"
MODEL_PATH = "yolo26n.pt"

# Intersection polygon: TL, TR, BR, BL
INTERSECTION_QUAD = np.array([
    (731, 376), (1267, 376), (1692, 806), (641, 971)
], dtype=np.int32)


# Perspective calibration
IMAGE_POINTS = np.float32([
    [620, 280], [1040, 280],
    [1450, 850], [500, 850]
])

REFERENCE_WIDTH_FT = 72.0
REFERENCE_HEIGHT_FT = 84.0

REAL_WORLD_POINTS = np.float32([
    [0, 0],
    [REFERENCE_WIDTH_FT, 0],
    [REFERENCE_WIDTH_FT, REFERENCE_HEIGHT_FT],
    [0, REFERENCE_HEIGHT_FT]
])

# Top entry line — used for approach/TTC calculation
LINE_A = (744, 399)
LINE_B = (1313, 369)

SPEED_WINDOW = 0.75
MIN_SPEED_FPS = 2.0
REQUIRED_SAFE_GAP = 1.0  # 0.8 sec clearance + 0.2 sec buffer

# ================= CALIBRATION =================
H = cv2.getPerspectiveTransform(
    IMAGE_POINTS, REAL_WORLD_POINTS
)

def pixel_to_feet(x, y):
    p = np.float32([[[x, y]]])
    x_ft, y_ft = cv2.perspectiveTransform(p, H)[0][0]
    return float(x_ft), float(y_ft)

LAx, LAy = pixel_to_feet(*LINE_A)
LBx, LBy = pixel_to_feet(*LINE_B)

def distance_from_line(x, y):
    dx, dy = LBx - LAx, LBy - LAy
    length = math.hypot(dx, dy)

    if length == 0:
        return 0

    return abs(
        (dx * (y - LAy) - dy * (x - LAx)) / length
    )

def above_entry_line(x, y):
    dx = LINE_B[0] - LINE_A[0]

    if dx == 0:
        return y < LINE_A[1]

    line_y = LINE_A[1] + (
        (x - LINE_A[0]) *
        (LINE_B[1] - LINE_A[1]) / dx
    )

    return y < line_y

# ================= MODEL / VIDEO =================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)


# ================= COORDINATE SCALING =================
# Resolution used when selecting the points
PICKER_WIDTH = 1920
PICKER_HEIGHT = 1080

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale_x = actual_width / PICKER_WIDTH
scale_y = actual_height / PICKER_HEIGHT

INTERSECTION_QUAD = (
    INTERSECTION_QUAD.astype(np.float32)
    * np.array([scale_x, scale_y])
).astype(np.int32)

LINE_A = (
    int(LINE_A[0] * scale_x),
    int(LINE_A[1] * scale_y)
)

LINE_B = (
    int(LINE_B[0] * scale_x),
    int(LINE_B[1] * scale_y)
)

print(f"Video resolution: {actual_width} x {actual_height}")
print(f"Coordinate scale: {scale_x:.3f} x {scale_y:.3f}")


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Could not read video's FPS.")

csv_file = open("tracking_data.csv", "w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "frame",
    "time_seconds",
    "vehicle_id",
    "x_center_pixels",
    "y_bottom_pixels",
    "width_pixels",
    "height_pixels",
    "confidence",
    "real_x_ft",
    "real_y_ft",
    "distance_from_entry_line_ft",
    "closing_speed_ft_per_sec",
    "speed_mph",
    "estimated_time_to_entry_line_sec",
    "inside_intersection",
    "left_turn_status"
])

frame_number = 0
trajectories = {}
distance_history = {}

# ================= TRACKING LOOP =================
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video finished!")
            break

        result = model.track(
            frame,
            imgsz=640,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        output = result.plot()

        # Draw intersection polygon
        cv2.polylines(
            output,
            [INTERSECTION_QUAD],
            True,
            (255, 255, 0),
            3
        )

        LINE_A = (731, 376)
        LINE_B = (1267, 376)
        

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().numpy()

            for box, track_id, confidence in zip(
                boxes, ids, confs
            ):

                x1, y1, x2, y2 = box

                # Bottom-center of vehicle
                x = int((x1 + x2) / 2)
                y = int(y2)

                width = x2 - x1
                height = y2 - y1
                time = frame_number / fps

                # ---------- REAL-WORLD POSITION ----------
                real_x, real_y = pixel_to_feet(x, y)

                distance = distance_from_line(
                    real_x,
                    real_y
                )

                # ---------- INTERSECTION ----------
                inside = cv2.pointPolygonTest(
                    INTERSECTION_QUAD.astype(np.float32),
                    (float(x), float(y)),
                    False
                ) >= 0

                # ---------- SPEED / TTC ----------

                history = distance_history.setdefault(track_id, deque())
                history.append((time, distance))

                while len(history) > 1 and time - history[0][0] > SPEED_WINDOW:
                    history.popleft()

                closing_speed = 0.0
                speed_mph = 0.0
                ttc = None

                if len(history) >= 2:
                    old_time, old_distance = history[0]
                    dt = time - old_time

                    if dt > 0:
                        speed_fps = abs(old_distance - distance) / dt

                        if speed_fps >= MIN_SPEED_FPS:
                            closing_speed = speed_fps
                            speed_mph = speed_fps * 0.681818

                            # Only calculate TTC while approaching the entry line
                            if above_entry_line(x, y):
                                ttc = distance / speed_fps

                # ---------- SAFE / UNSAFE ----------
                if inside:
                    status = "UNSAFE"

                elif (
                    above_entry_line(x, y)
                    and ttc is not None
                ):
                    status = (
                        "SAFE"
                        if ttc > REQUIRED_SAFE_GAP
                        else "UNSAFE"
                    )

                else:
                    status = "NOT RELEVANT"

                # ---------- CSV ----------
                writer.writerow([
                    frame_number,
                    time,
                    track_id,
                    x,
                    y,
                    width,
                    height,
                    confidence,
                    real_x,
                    real_y,
                    distance,
                    closing_speed,
                    speed_mph,
                    ttc,
                    inside,
                    status
                ])

                # ---------- TRAJECTORY ----------
                points = trajectories.setdefault(
                    track_id,
                    []
                )

                points.append((x, y))

                if len(points) > 100:
                    points.pop(0)

                for i in range(1, len(points)):
                    cv2.line(
                        output,
                        points[i - 1],
                        points[i],
                        (0, 0, 255),
                        2
                    )

                cv2.circle(
                    output,
                    (x, y),
                    7,
                    (0, 0, 255),
                    -1
                )

                # =================================
                # STATS LABEL — ALWAYS STAYS
                # =================================
                if ttc is not None:
                    text = (
                        f"ID {track_id} | "
                        f"{distance:.1f} ft | "
                        f"{speed_mph:.1f} mph | "
                        f"{ttc:.1f} s"
                    )
                else:
                    text = (
                        f"ID {track_id} | "
                        f"{distance:.1f} ft | "
                        f"{speed_mph:.1f} mph"
                    )

                tx = x + 10
                ty = y - 10

                if tx + 420 > output.shape[1]:
                    tx = max(5, x - 420)

                if ty < 25:
                    ty = y + 30

                # Black background behind stats
                (tw, th), baseline = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2
                )

                cv2.rectangle(
                    output,
                    (tx - 5, ty - th - 5),
                    (tx + tw + 5, ty + baseline + 5),
                    (0, 0, 0),
                    -1
                )

                # Stats
                cv2.putText(
                    output,
                    text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                # =================================
                # STATUS — ONLY SAFE / UNSAFE
                # =================================
                if status in ("SAFE", "UNSAFE"):

                    status_color = (
                        (0, 255, 255)
                        if status == "SAFE"
                        else (0, 0, 255)
                    )

                    cv2.putText(
                        output,
                        status,
                        (tx, ty + 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        status_color,
                        4
                    )

        cv2.imshow(
            "YOLO + ByteTrack + Intersection Safety",
            output
        )

        if cv2.waitKey(1) & 0xFF in (
            ord("q"),
            ord("Q")
        ):
            print("Stopped by user.")
            break

        frame_number += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

print("Program terminated.")
print("Tracking data saved to tracking_data.csv")