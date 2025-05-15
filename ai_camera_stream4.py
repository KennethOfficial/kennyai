from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time
import os

app = Flask(__name__)

print("[INFO] Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")
print("[INFO] Model loaded successfully!")

class CameraReader(threading.Thread):
    def __init__(self, width, height):
        super().__init__()
        self.queue = queue.Queue(maxsize=2)
        self.running = True
        self.width = width
        self.height = height
        print(f"[INFO] Camera reader initialized with resolution: {width}x{height}")

    def find_camera(self):
        """Try to find an available camera by testing different indices"""
        for i in range(10):  # Try first 10 indices
            print(f"[INFO] Trying camera index {i}...")
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[INFO] Successfully opened camera at index {i}")
                    return cap
                cap.release()
        return None

    def run(self):
        print("[INFO] Camera reader thread started")
        
        try:
            # Try to find an available camera
            cap = self.find_camera()
            if cap is None:
                print("[ERROR] No camera found. Please check your camera connection.")
                return

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify camera settings
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Camera settings - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}")
            
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARNING] Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Verify frame is valid
                if frame.size == 0:
                    print("[WARNING] Received empty frame")
                    continue
                    
                # Put the frame in the queue
                if not self.queue.full():
                    self.queue.put(frame)
                else:
                    # Discard oldest frame if queue is full
                    try:
                        self.queue.get_nowait()
                        self.queue.put(frame)
                    except queue.Empty:
                        pass
                        
            cap.release()
            
        except Exception as e:
            print(f"[ERROR] Camera reader thread error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def read(self):
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        print("[INFO] Camera reader thread stopped")

WIDTH = 640
HEIGHT = 480

print("[INFO] Starting camera stream...")
reader = CameraReader(WIDTH, HEIGHT)
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
        if frame_count % 30 == 0:
            print(f"[INFO] Processing frame #{frame_count}")

        # Apply detection with confidence threshold
        try:
            results = model(frame, conf=0.25, verbose=False)
            
            # Display detection information
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                print(f"[INFO] Detections found: {len(boxes)}, classes: {boxes.cls.tolist()}")
                
                # Detailed logging about detections (limit to avoid spam)
                for i, box in enumerate(boxes[:3]):  # Show only first 3 detections in log
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = results[0].names[cls_id]
                    print(f"[DETECTION] #{i}: {class_name} (confidence: {conf:.2f})")
                
                if len(boxes) > 3:
                    print(f"[DETECTION] ... and {len(boxes) - 3} more objects")
            else:
                if frame_count % 30 == 0:
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
