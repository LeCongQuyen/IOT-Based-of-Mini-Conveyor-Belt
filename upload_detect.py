from ultralytics import YOLO
import cv2
import firebase_admin
from firebase_admin import credentials, db
import time

# ============================
# Model
# ============================
model = YOLO(
    "D:/Learn/object_detect/Yolo1/runs/detect/train2/weights/best.pt"
)

# ============================
# Firebase
# ============================
cred = credentials.Certificate(
    r"D:\Learn\Unity_project\digitalbelttwin-firebase-adminsdk-fbsvc-71c79ce6ca.json"
)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://digitalbelttwin-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

latest_ref = db.reference("/detection/latest")
control_ref = db.reference("camera_run")

# ============================
# ROI (Conveyor middle zone)
# ============================
ROI_X = 50
ROI_Y = 130
ROI_W = 500
ROI_H = 250

# Trigger line (inside ROI, near right side)
TRIGGER_X = ROI_X + ROI_W - 250

# ============================
# State variables
# ============================
cam = None
camera_active = False

object_locked = False
prev_center_x = None
object_uid = 0

print("Camera service ready. Waiting for RUN button...")

# ============================
# Main loop
# ============================
while True:
    camera_run = control_ref.get()

    # ---------- START CAMERA ----------
    if camera_run and not camera_active:
        print("Camera STARTED")
        cam = cv2.VideoCapture(0)
        camera_active = True

    # ---------- STOP CAMERA ----------
    if not camera_run and camera_active:
        print("Camera STOPPED")
        cam.release()
        cv2.destroyAllWindows()
        cam = None
        camera_active = False
        object_locked = False
        prev_center_x = None
        time.sleep(0.2)
        continue

    if not camera_active:
        time.sleep(0.2)
        continue

    ret, frame = cam.read()
    if not ret:
        continue

    # ---------- ROI CROP ----------
    roi = frame[
        ROI_Y:ROI_Y + ROI_H,
        ROI_X:ROI_X + ROI_W
    ]

    results = model(roi, conf=0.5)[0]

    object_detected_this_frame = False

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Convert ROI coords → frame coords
        x1 += ROI_X
        x2 += ROI_X
        y1 += ROI_Y
        y2 += ROI_Y

        center_x = (x1 + x2) / 2
        label = model.names[cls]

        object_detected_this_frame = True

        # ---------- TRIGGER LOGIC ----------
        if not object_locked:
            if prev_center_x is not None:
                if prev_center_x < TRIGGER_X and center_x >= TRIGGER_X:
                    object_locked = True
                    object_uid += 1
                    print(f"TRIGGERED: {label} ({conf:.2f})")

                    data = {
                        "uid": object_uid,
                        "timestamp": int(time.time()),
                        "object_name": label,
                        "confidence": conf,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        }
                    }

                    latest_ref.set(data)
                    object_locked = True

        prev_center_x = center_x

        # ---------- DRAW bbox (for visualization only) ----------
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ---------- UNLOCK when object leaves ROI ----------
    if not object_detected_this_frame:
        object_locked = False
        prev_center_x = None

    # ---------- DRAW ROI ----------
    cv2.rectangle(
        frame,
        (ROI_X, ROI_Y),
        (ROI_X + ROI_W, ROI_Y + ROI_H),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "ROI - Conveyor Zone",
        (ROI_X, ROI_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # ---------- DRAW TRIGGER LINE ----------
    cv2.line(
        frame,
        (TRIGGER_X, ROI_Y),
        (TRIGGER_X, ROI_Y + ROI_H),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Trigger Line",
        (TRIGGER_X - 80, ROI_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        control_ref.set(False)
