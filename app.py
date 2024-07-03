from flask import Flask, Response, render_template
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import datetime
import calendar
from shapely.geometry import Point, Polygon
import logging

# Setup Flask app
app = Flask(__name__)

# Initialize the YOLO model
model = YOLO("terbaru.pt")

# Define region of interest (ROI)
region_of_interest = [(300, 20), (302, 680), (280, 680), (280, 20)]
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True, reg_pts=region_of_interest, classes_names=model.names, draw_tracks=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Realtime Object Detection & Counting
@app.route('/realtime')
def index():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(count_object(), mimetype='multipart/x-mixed-replace; boundary=frame')

def count_object():
    cap = cv2.VideoCapture('data/video.mov')
    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        return

    tracked_ids = set()
    
    while True:
        success, im0 = cap.read()
        if not success:
            logging.info("End of video file reached or can't read frame.")
            break
        
        tracks = model.track(im0, persist=True, show=False)
        im0 = counter.start_counting(im0, tracks)
        
        # Process tracks and save to MongoDB if crossing the ROI
        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if track_id not in tracked_ids:
                    prev_position = counter.track_history[track_id][-2] if len(counter.track_history[track_id]) > 1 else None
                    current_position = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
                    
                    if len(region_of_interest) >= 3:
                        counting_region = Polygon(region_of_interest)
                        is_inside = counting_region.contains(Point(current_position))
                        
                        if prev_position and is_inside:
                            tracked_ids.add(track_id)

        # Menambahkan label jenis buah
        cv2.putText(im0, f"Ultralytics Analytics", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        counts_text = ', '.join([f"{model.names[cls]}: IN {count}" for cls, count in counter.counts.items()])
        cv2.putText(im0, counts_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
