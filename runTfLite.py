import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ===============================
# CONFIG
# ===============================
MODEL_PATH = '/home/chricke/Projects/runs/detect/train44/weights/best_saved_model/best_float32.tflite'
VIDEO_PATH = '/home/chricke/Projects/test1.webm'
CONF_THRESHOLD = 0.0

# ===============================
# LOAD TFLITE MODEL
# ===============================
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# ===============================
# LETTERBOX FUNCTION (YOLO STYLE)
# ===============================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # current shape (height, width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# ===============================
# POSTPROCESS (YOLO RAW OUTPUT)
# ===============================
def process_predictions(predictions, r, dwdh, orig_shape):
    boxes = []
    # predictions is (1, 14, 8400) → remove batch dim and transpose
    predictions = np.squeeze(predictions)      # (14, 8400)
    predictions = np.transpose(predictions)    # (8400, 14)

    for pred in predictions:
        print("Pred:", pred)
        conf = pred[4]  # objectness score
        if conf > CONF_THRESHOLD:
            cls_id = np.argmax(pred[5:])  # pick class with highest probability
            x, y, w, h = pred[0:4]

            # Undo padding (dwdh) and scaling (r) from preprocessing
            x -= dwdh[0]
            y -= dwdh[1]
            x /= r
            y /= r
            w /= r
            h /= r

            # Convert from center x,y,w,h → x1,y1,x2,y2
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            boxes.append((x1, y1, x2, y2, conf, cls_id))
    return boxes


# ===============================
# RUN INFERENCE ON VIDEO
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img, r, dwdh = letterbox(frame, (input_height, input_width))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)
    print(input_data.shape)
    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess
    boxes = process_predictions(output_data, r, dwdh, frame.shape)

    # Draw boxes
    for (x1, y1, x2, y2, conf, cls_id) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_id} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 TFLite Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

