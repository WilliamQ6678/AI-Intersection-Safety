import cv2

VIDEO_PATH = "BellevueIntersection2.mov"

cap = cv2.VideoCapture(VIDEO_PATH)



ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not open BellevueIntersection2.mov")

points = []
base_frame = frame.copy()

WINDOW_NAME = "Pick Intersection Quadrilateral"


def draw_points():
    display = base_frame.copy()

    # Draw selected points and their numbers.
    for index, point in enumerate(points):
        cv2.circle(
            display,
            point,
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            str(index + 1),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Draw edges between selected points.
    for i in range(1, len(points)):
        cv2.line(
            display,
            points[i - 1],
            points[i],
            (255, 255, 0),
            3
        )

    # Close the shape after all four points exist.
    if len(points) == 4:
        cv2.line(
            display,
            points[3],
            points[0],
            (255, 255, 0),
            3
        )

        # Green top edge = far entry/reference line.
        cv2.line(
            display,
            points[0],
            points[1],
            (0, 255, 0),
            5
        )

        cv2.putText(
            display,
            "TOP / ENTRY LINE",
            (points[0][0] + 10, points[0][1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return display


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")


print()
print("Click four points in CLOCKWISE order:")
print("1 = top-left")
print("2 = top-right")
print("3 = bottom-right")
print("4 = bottom-left")
print()
print("The green top edge is the far entry/distance line.")
print("U = undo last point")
print("R = reset all points")
print("Q = finish")
print()

cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    click_event
)

while True:
    display_frame = draw_points()

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key in [ord("u"), ord("U")] and points:
        points.pop()

    if key in [ord("r"), ord("R")]:
        points.clear()

    if key in [ord("q"), ord("Q")]:
        break

cv2.destroyAllWindows()

if len(points) == 4:
    print()
    print("Copy this into your tracking script:")
    print()
    print("INTERSECTION_QUAD = np.array([")
    print(f"    {points[0]},  # top-left")
    print(f"    {points[1]},  # top-right")
    print(f"    {points[2]},  # bottom-right")
    print(f"    {points[3]}   # bottom-left")
    print("], dtype=np.int32)")
else:
    print()
    print("You selected fewer than four points.")