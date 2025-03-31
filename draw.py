import cv2
import cvzone
import math
import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
import keyboard
import threading
from ultralytics import YOLO

WINDOW_TITLE = 'VALORANT'
CONFIDENCE_THRESHOLD = 0.2

model = YOLO("models/best.pt")

running = False


def toggle_running():
    global running
    running = not running
    if running:
        print("开始运行.")
    else:
        print("停止.")


keyboard.on_press_key('shift', lambda _: toggle_running())


def get_window_image(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    bbox = (window.left, window.top, window.right, window.bottom)
    img = ImageGrab.grab(bbox)
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np


def process_image():
    global running
    while True:
        if running:
            img = get_window_image(WINDOW_TITLE)
            if img is not None:
                results = model(img, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        currentClass = model.names[cls]

                        if currentClass == "head" and conf >= CONFIDENCE_THRESHOLD:
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.circle(img, (center_x, center_y), max(w, h) // 2, (0, 255, 0), 2)
                            cvzone.putTextRect(img, f'{currentClass} {conf}', (x1, max(35, y1)), scale=1, thickness=1,
                                               offset=5)
                            cvzone.cornerRect(img, (x1, y1, w, h), l=8)

                        if currentClass == "person" and conf >= CONFIDENCE_THRESHOLD:
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.circle(img, (center_x, center_y), max(w, h) // 2, (0, 255, 0), 2)
                            cvzone.putTextRect(img, f'{currentClass} {conf}', (x1, max(35, y1)), scale=1, thickness=1,
                                               offset=5)
                            cvzone.cornerRect(img, (x1, y1, w, h), l=8)

                resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Detected", resized_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    keyboard.unhook_all()


thread = threading.Thread(target=process_image)
thread.start()
