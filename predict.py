import time
import firebase_admin
from firebase_admin import credentials, db

# ============================
# Firebase initialization
# ============================
cred = credentials.Certificate(
    r"D:\Learn\Unity_project\digitalbelttwin-firebase-adminsdk-fbsvc-71c79ce6ca.json"
)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://digitalbelttwin-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

print("Firebase initialized successfully")
control_ref = db.reference("prediction_run")
# ============================
# Prediction loop
# ============================
while True:
    run_pred = control_ref.get()
    if not run_pred:
        time.sleep(0.5)
        continue

    motor = db.reference("/realtime/motor").get()
    load = db.reference("/realtime/load").get()

    if motor is None:
        print("Waiting for motor data...")
        time.sleep(0.2)
        continue

    rpm = motor.get("speed_rpm", 0)
    current = motor.get("current_A", 0)
    temperature = motor.get("temperature_C", 0)

    status = "normal"
    reason = "System stable"

    if current > 0.6:
        status = "fault"
        reason = "Motor overload"
    elif rpm < 10  :
        status = "fault"
        reason = "Motor speed drop"
    elif temperature >= 40:
        status = "fault"
        reason = "Motor overheating"

    # Warning conditions
    elif current > 0.48:
        status = "warning"
        reason = "Increasing motor load"
    elif temperature > 33:
        status = "warning"
        reason = "Elevated motor temperature"
    

    db.reference("/prediction").set({
        "status": status,
        "reason": reason,
        "rpm": rpm,
        "current_A": current,
        "temperature_C": temperature
    })

    print(f"Prediction -> {status.upper()} | RPM={rpm} | Current={current} | Temperature={temperature}")

    time.sleep(1)
