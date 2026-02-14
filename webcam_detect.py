import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import json


IMG_SIZE = 96
CONFIDENCE_THRESHOLD = 65  
SMOOTHING_FRAMES = 10


model = tf.keras.models.load_model("classifier.h5")


with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)


prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)


cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape


    cv2.rectangle(frame,
                  (w//4, h//4),
                  (3*w//4, 3*h//4),
                  (255, 0, 0),
                  2)

 

    center = frame[h//4:3*h//4, w//4:3*w//4]

    resized = cv2.resize(center, (IMG_SIZE, IMG_SIZE))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    
    crop_size = int(IMG_SIZE * 0.8)
    start = (IMG_SIZE - crop_size) // 2
    resized = resized[start:start+crop_size, start:start+crop_size]

    resized = cv2.resize(resized, (IMG_SIZE, IMG_SIZE))

    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

  
    predictions = model.predict(img_array, verbose=0)[0]

    prediction_buffer.append(predictions)
    avg_prediction = np.mean(prediction_buffer, axis=0)

    max_index = np.argmax(avg_prediction)
    max_confidence = avg_prediction[max_index] * 100

    if max_confidence > CONFIDENCE_THRESHOLD:
        label = class_names[max_index]
        color = (0, 255, 0)
    else:
        label = "Uncertain"
        color = (0, 0, 255)


    cv2.putText(frame,
                f"{label} ({max_confidence:.2f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)


    y_offset = 90
    for i, (cls, prob) in enumerate(zip(class_names, avg_prediction)):
        percentage = prob * 100
        bar_length = int((percentage / 100) * 200)

        bar_color = (0, 255, 0) if i == max_index else (255, 255, 255)

        cv2.putText(frame,
                    f"{cls}: {percentage:.2f}%",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bar_color,
                    2)

        cv2.rectangle(frame,
                      (180, y_offset - 15),
                      (180 + bar_length, y_offset - 5),
                      bar_color,
                      -1)

        y_offset += 35

    cv2.imshow("Ultimate Gesture Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
