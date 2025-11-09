from flask import Flask, render_template, request, url_for, redirect, Response
import os
import cv2
import numpy as np
from detector import detect_plates
from ocr_reader import read_license_plate

# ---------------- CONFIG -----------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
OUTPUT_IMG = os.path.join(UPLOAD_FOLDER, 'output.jpg')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ------------------------------------------


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handles image upload, YOLO detection, and OCR recognition."""
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Read uploaded image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "Error: Unable to read image file."

    # Run YOLOv11 detection
    boxes, crops = detect_plates(img)
    recognized = []

    # Process each detected plate
    for box, crop in zip(boxes, crops):
        if crop is None or crop.size == 0:
            continue

        text, score = read_license_plate(crop)
        if text is None:
            text = "UNKNOWN"

        x1, y1, x2, y2, conf = box
        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        recognized.append({
            'box': (x1, y1, x2, y2),
            'text': text,
            'conf': float(conf)
        })

    # Save output image
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    cv2.imwrite(OUTPUT_IMG, img)

    # Render result page
    return render_template(
        'result.html',
        output_image=url_for('static', filename='output.jpg'),
        texts=recognized
    )


# ------------------ WEBCAM DETECTION ------------------

def gen_frames():
    """Generate frames from webcam with YOLOv11 detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO detection on live frame
        boxes, crops = detect_plates(frame)

        for box, crop in zip(boxes, crops):
            if crop is None or crop.size == 0:
                continue

            text, score = read_license_plate(crop)
            if text is None:
                text = "UNKNOWN"

            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame to JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\\r\\n'
               b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame_bytes + b'\\r\\n')

    cap.release()


@app.route('/camera')
def camera():
    """Webcam demo page."""
    return render_template('index.html', camera=True)


@app.route('/video_feed')
def video_feed():
    """Stream webcam frames with detection."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
