from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from io import BytesIO
from ultralytics import YOLO
from test2 import process_image

app = Flask(__name__)

# Load model
model = YOLO("weights/best.pt")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        # Convert to OpenCV format
        in_memory = BytesIO()
        file.save(in_memory)
        npimg = np.frombuffer(in_memory.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Process image
        processed_img, damage_rate, parts = process_image(img, model)
        processed_img, damage_rate, parts = process_image(img, model)

        severity = get_severity(damage_rate)
        estimated_cost = estimate_cost(parts)

        # Encode original image
        _, buffer1 = cv2.imencode('.jpg', img)
        original_base64 = base64.b64encode(buffer1).decode('utf-8')

        # Encode processed image
        _, buffer2 = cv2.imencode('.jpg', processed_img)
        processed_base64 = base64.b64encode(buffer2).decode('utf-8')

        return render_template('index.html',
                       original=original_base64,
                       processed=processed_base64,
                       damage_rate=f"{damage_rate:.2f}",
                       parts=parts,
                       count=len(parts),
                       severity=severity,
                       cost=estimated_cost)

    return render_template('index.html')

def get_severity(damage_rate):
    if damage_rate < 10:
        return "Low"
    elif damage_rate < 30:
        return "Medium"
    else:
        return "High"

def estimate_cost(parts):
    cost_map = {
        "front-bumper-dent": 3000,
        "rear-bumper-dent": 3000,
        "doorouter-dent": 5000,
        "bonnet-dent": 6000,
        "boot-dent": 6000,
        "fender-dent": 4000,
        "roof-dent": 7000,
        "pillar-dent": 5000,
        "Sidemirror-Damage": 2000,
        "Headlight-Damage": 2500,
        "Taillight-Damage": 2500
    }

    total = 0
    for part in parts:
        total += cost_map.get(part, 3000)  # default cost

    return total
if __name__ == "__main__":
    app.run(debug=False, port=5001)