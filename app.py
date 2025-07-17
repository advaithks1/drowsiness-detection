from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pyttsx3
import threading
import time
import platform
from tensorflow.keras.models import load_model
import json
from datetime import datetime
import winsound
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import base64
from io import BytesIO
from PIL import Image
import logging
import os


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
model = load_model("final_mobilenet_model.h5")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

previous_eye_state = "Open"
eye_missing_frames = 0
MISSING_THRESHOLD = 3
closed_start_time = None
alarm_active = False
detection_started = False

# Speed logic
current_speed = 10
speed_increment = 5
speed_decrement = 10

SLEEP_DATA_FILE = 'sleep_data.json'


def save_sleep_data(state, location):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"timestamp": timestamp, "state": state, "location": location}
    try:
        with open(SLEEP_DATA_FILE, 'a') as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"[ERROR] Saving sleep data: {e}")


def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()


def predict_eye(img):
    img = cv2.resize(img, (160, 160)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "Open" if pred > 0.5 else "Closed"


def play_alarm():
    try:
        winsound.PlaySound("/static/alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"Alarm sound error: {e}")
    speak("Wake up! Your eyes are closed")


def update_speed(eye_state):
    global current_speed
    if eye_state == "Open":
        current_speed = min(100, current_speed + speed_increment)
    else:
        current_speed = max(0, current_speed - speed_decrement)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_started
    detection_started = True
    return jsonify({"message": "Detection started"})


@app.route('/speed')
def get_speed():
    return jsonify({'speed': current_speed})


@app.route('/save_sleep_data', methods=['POST'])
def save_data():
    data = request.json
    save_sleep_data(data.get("state", "Unknown"), data.get("location", "Unknown"))
    return jsonify({"message": "Sleep data saved"})


def gen_frames():
    global previous_eye_state, eye_missing_frames, closed_start_time, alarm_active, detection_started

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(30, 30))
            detected_eye_state = None

            for (ex, ey, ew, eh) in eyes:
                if ew < 30 or eh < 30:
                    continue
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                detected_eye_state = predict_eye(eye_img)
                previous_eye_state = detected_eye_state
                eye_missing_frames = 0

                color = (0, 255, 0) if detected_eye_state == "Open" else (0, 0, 255)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                cv2.putText(roi_color, detected_eye_state, (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                break

            if detected_eye_state is None:
                eye_missing_frames += 1
                if eye_missing_frames >= MISSING_THRESHOLD:
                    previous_eye_state = "Closed"

            update_speed(previous_eye_state)

            if previous_eye_state == "Closed":
                if closed_start_time is None:
                    closed_start_time = time.time()
                elif time.time() - closed_start_time >= 3:
                    if not alarm_active and detection_started:
                        alarm_active = True
                        play_alarm()
                        save_sleep_data("Drowsy", "Driver's location")
            else:
                closed_start_time = None
                alarm_active = False

            status_color = (0, 255, 0) if previous_eye_state == "Open" else (0, 0, 255)
            cv2.putText(frame, previous_eye_state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/eye_state')
def eye_state():
    
    return previous_eye_state  # Return either 'Open' or 'Closed'


@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        data = request.json
        image_data = data.get('image')
        subject = data.get('subject')
        body = data.get('body')

        # Check if image data is missing
        if not image_data:
            raise ValueError("Image data is missing")

        app.logger.debug(f"Received data: {data}")
        image_data = image_data.split(",")[1]  # remove the data:image/png;base64 part
        image_bytes = base64.b64decode(image_data)

        # Create an image from the byte data
        image = Image.open(BytesIO(image_bytes))

        # Save the image to a temporary file
        image.save("temp_image.jpg")

        # Direct email credentials (replace these with your details)
        from_email = ""  # Replace with your Gmail email address
        smtp_password = ""  # Replace with your Gmail App Password
        to_email = ''  # Replace with the recipient's email address

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach the image to the email
        part = MIMEBase('application', 'octet-stream')
        with open("temp_image.jpg", "rb") as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="drowsy_image.jpg"')
        msg.attach(part)

        # Send the email using SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(from_email, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())

        return jsonify({"message": "Email sent successfully"}), 200

    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
