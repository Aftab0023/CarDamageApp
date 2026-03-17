import cv2
import cvzone

# ✅ ADD THIS (IMPORTANT)
class_labels = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage', 
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent', 
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent', 
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

def calculate_damage_rate(boxes, img_width, img_height):
    total_area = img_width * img_height
    damage_area = 0

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        damage_area += (x2 - x1) * (y2 - y1)

    return (damage_area / total_area) * 100


def process_image(img, model, threshold=0.4):
    results = model(img)

    final_boxes = []
    detected_parts = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < threshold:
                continue

            label = class_labels[cls]

            # 🔥 Check overlap (simple NMS)
            is_duplicate = False
            for fb in final_boxes:
                fx1, fy1, fx2, fy2 = map(int, fb.xyxy[0])

                # overlap check
                if abs(x1 - fx1) < 50 and abs(y1 - fy1) < 50:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_boxes.append(box)

                # Draw
                cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1))
                cvzone.putTextRect(img, f'{label} {conf:.2f}', (x1, y1-10))

                if label not in detected_parts:
                    detected_parts.append(label)

    # ✅ Use only filtered boxes
    damage_rate = calculate_damage_rate(final_boxes, img.shape[1], img.shape[0])

    return img, damage_rate, detected_parts
    results = model(img)
    all_boxes = []
    detected_parts = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > threshold:
                label = class_labels[cls]

                cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1))
                cvzone.putTextRect(img, f'{label} {conf:.2f}', (x1, y1-10))

                all_boxes.append(box)
                if label not in detected_parts:
                    detected_parts.append(label)

    damage_rate = calculate_damage_rate(all_boxes, img.shape[1], img.shape[0])

    return img, damage_rate, detected_parts

