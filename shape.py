import cv2
import numpy as np

def detect_shape(approx):
    vertices = len(approx)
    shape = "Unidentified"

    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        # Check for square, rectangle, rhombus, trapezium
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # Get side lengths
        sides = []
        for i in range(4):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % 4][0]
            side = np.linalg.norm(pt1 - pt2)
            sides.append(side)
        avg_side = sum(sides) / 4
        side_var = max(sides) - min(sides)

        if 0.95 <= aspect_ratio <= 1.05 and side_var < 0.15 * avg_side:
            shape = "Square"
        elif side_var < 0.2 * avg_side:
            shape = "Rhombus"
        else:
            shape = "Rectangle" if 0.7 <= aspect_ratio <= 1.4 else "Trapezium"
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices == 6:
        shape = "Hexagon"
    elif vertices == 8:
        shape = "Octagon"
    elif vertices > 8:
        shape = "Circle or Oval"
    return shape

# Load image
image = cv2.imread("shapes.png")  # <-- Replace with your file name
if image is None:
    print("❌ Could not load image. Check the path.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) < 300:
        continue

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    shape = detect_shape(approx)

    M = cv2.moments(contour)
    if M["m00"] == 0:
        continue
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX - 40, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Show and save result
cv2.imshow("Detected Shapes", image)
cv2.imwrite("labeled_shapes.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
