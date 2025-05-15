from flask import Flask, Response
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
import threading
import queue

app = Flask(__name__)

print("[INFO] Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")

class YUVStreamReader(threading.Thread):
    def __init__(self, cmd, width, height):
        super().__init__()
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        self.queue = queue.Queue(maxsize=1)
        self.running = True
        self.width = width
        self.height = height
        self.frame_size = int(self.width * self.height * 1.5)  # YUV420p

    def run(self):
        while self.running:
            raw_data = self.process.stdout.read(self.frame_size)
            if len(raw_data) != self.frame_size:
                continue

            yuv_frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((int(self.height * 1.5), self.width))
            bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(bgr_frame)

    def read(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def stop(self):
        self.running = False
        self.process.terminate()
        self.process.wait()

WIDTH = 640
HEIGHT = 480

print("[INFO] Starting libcamera-vid threaded YUV reader...")
reader = YUVStreamReader([
    "libcamera-vid", "-t", "0",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", "30",
    "--codec", "yuv420",
    "-o", "-"
], WIDTH, HEIGHT)
reader.start()

def generate_frames():
    while True:
        frame = reader.read()
        if frame is None:
            print("[WARNING] No frame received")
            continue

        # Optional brightness correction if needed
        # frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        # YOLO detection
        results = model(frame, imgsz=640)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            print(f"[INFO] Detections found: {len(results[0].boxes)}")
        else:
            print("[INFO] No objects detected")

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Pi AI Camera Feed with YOLOv8 Active Detection</h1><img src='/video_feed'>"

if __name__ == '__main__':
    print("[INFO] Starting Flask server at http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        print("[INFO] Stopping reader...")
        reader.stop()
