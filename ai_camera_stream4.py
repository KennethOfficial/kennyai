from flask import Flask, Response
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
import threading
import queue
import time

app = Flask(__name__)

print("[INFO] Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")
print("[INFO] Model loaded successfully!")

class YUVStreamReader(threading.Thread):
    def __init__(self, cmd, width, height):
        super().__init__()
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        self.queue = queue.Queue(maxsize=1)
        self.running = True
        self.width = width
        self.height = height
        self.frame_size = int(self.width * self.height * 1.5)  # YUV420p
        print(f"[INFO] YUVStreamReader initialized with frame size: {self.frame_size}")

    def run(self):
        print("[INFO] YUVStreamReader thread started")
        while self.running:
            raw_data = self.process.stdout.read(self.frame_size)
            if len(raw_data) != self.frame_size:
                print(f"[WARNING] Incomplete frame data: {len(raw_data)}/{self.frame_size} bytes")
                continue

            yuv_frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((int(self.height * 1.5), self.width))
            bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            print(f"[DEBUG] Frame converted: shape={bgr_frame.shape}, dtype={bgr_frame.dtype}")

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
        print("[INFO] YUVStreamReader thread stopped")

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
print("[INFO] Camera stream started")

def generate_frames():
    frame_count = 0
    while True:
        frame = reader.read()
        if frame is None:
            print("[WARNING] No frame received, waiting...")
            time.sleep(0.1)
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"[INFO] Processing frame #{frame_count}")

        # Apply detection with confidence threshold
        try:
            results = model(frame, conf=0.25, verbose=False)
            
            # Display detection information
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                print(f"[INFO] Detections found: {len(boxes)}, classes: {boxes.cls.tolist()}")
                
                # Detailed logging about detections
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = results[0].names[cls_id]
                    print(f"[DETECTION] #{i}: {class_name} (confidence: {conf:.2f})")
            else:
                if frame_count % 10 == 0:
                    print("[INFO] No objects detected")
                
            # Generate the annotated frame with bounding boxes
            annotated_frame = results[0].plot()
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {str(e)}")
            annotated_frame = frame  # Use original frame if detection fails

        # Convert the frame to JPEG for streaming
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("[ERROR] Failed to encode frame to JPEG")
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
        except Exception as e:
            print(f"[ERROR] Frame encoding error: {str(e)}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Pi AI Camera Feed</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
          h1 { color: #333; }
          .stream-container { margin-top: 20px; }
          img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        </style>
      </head>
      <body>
        <h1>Pi AI Camera Feed with YOLOv8 Object Detection</h1>
        <div class="stream-container">
          <img src='/video_feed'>
        </div>
      </body>
    </html>
    """

if __name__ == '__main__':
    print("[INFO] Starting Flask server at http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        print("[INFO] Stopping camera stream...")
        reader.stop()
