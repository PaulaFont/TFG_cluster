import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

# === CONFIGURATION ===

# Replace this with the path to your image
image_path = '/data/users/pfont/input/rsc37_rsc176_402_0.jpg'


json_path = "/data/users/pfont/layout_data/out_binary_hisam_inverted/results.json"
json_path = "/data/users/pfont/layout_data/input/results.json"
with open(json_path, "r") as f:
    json_data = json.load(f)

layout_data = json_data["rsc37_rsc176_402_0"][0]["bboxes"]


# Sort by reading order
layout_data = sorted(layout_data, key=lambda x: x["position"])

# === DRAWING LOGIC ===

# Load image using OpenCV
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Remove the region with position == 4
layout_data = [region for region in layout_data if region["position"] != 4]
# Draw polygons and labels
for region in layout_data:
    polygon = np.array(region["polygon"], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_rgb, [polygon], isClosed=True, color=(255, 0, 0), thickness=4)

    # Compute centroid for label and arrow
    cx = int(np.mean([p[0] for p in region["polygon"]]))
    cy = int(np.mean([p[1] for p in region["polygon"]]))
    region["centroid"] = (cx, cy)

    # Draw reading order slightly to the right of the centroid
    offset_x = 40  # Adjust this value to move further right
    offset_y = 0   # You can also adjust vertically if needed
    cv2.putText(
        img_rgb,
        str(region["position"]),
        (cx + offset_x, cy + offset_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2.5,
        color=(0, 0, 255),
        thickness=6
    )

# Draw arrows for reading order with consistent tip size
arrow_tip_length = 30  # pixels

for i in range(len(layout_data) - 1):
    pt1 = np.array(layout_data[i]["centroid"], dtype=np.float32)
    pt2 = np.array(layout_data[i + 1]["centroid"], dtype=np.float32)
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length == 0:
        continue
    tip_length = min(arrow_tip_length / length, 0.5)  # OpenCV expects a ratio (max 0.5)
    cv2.arrowedLine(
        img_rgb,
        tuple(pt1.astype(int)),
        tuple(pt2.astype(int)),
        color=(255, 0, 0),
        thickness=4,
        tipLength=tip_length
    )

# Save result
output_path = "layout_402_2.jpg"
Image.fromarray(img_rgb).save(output_path)
print(f"Saved overlay image to {output_path}")
