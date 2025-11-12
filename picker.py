import cv2
import numpy as np

# Global lists
points_left = []
points_right = []
current_list = points_left  

def mouse_callback(event, x, y, flags, param):
    global current_list
    if event == cv2.EVENT_LBUTTONDOWN:
        current_list.append((x, y))
        print(f"Point recorded: {(x,y)}")

number = 6
style = "old" if number <= 3 else "new"
name = f"Picture{str(number)}"

img_path = f"data/{style}/{name}.png"
img = cv2.imread(img_path)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

print("Click ~15 points for LEFT brow. Press 'r' for RIGHT brow, 'l' for LEFT brow, 'q' to finish.")

while True:
    display = img.copy()
    
    # Draw left brow points (blue)
    for p in points_left:
        cv2.circle(display, p, 3, (255,0,0), -1)
    # Draw right brow points (green)
    for p in points_right:
        cv2.circle(display, p, 3, (0,255,0), -1)

    cv2.imshow("image", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        print("Switched to RIGHT brow.")
        current_list = points_right
    elif key == ord("l"):
        print("Switched to LEFT brow.")
        current_list = points_left
    elif key == ord("q"):
        print("Finished selecting points.")
        annotated = display.copy()
        break

cv2.destroyAllWindows()

np.savetxt(f"{name}_{style}_left.csv", points_left, fmt="%d", delimiter=",")
np.savetxt(f"{name}_{style}_right.csv", points_right, fmt="%d", delimiter=",")
cv2.imwrite(f"{name}_{style}_with_points.jpg", annotated)
