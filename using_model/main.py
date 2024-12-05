import mediapipe as mp
import cv2
import numpy as np


model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)
frame_skip_rate = 5

with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % frame_skip_rate == 0:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)

                    face = frame[y:y + height, x:x + width]
                    gray_face = cv2.cvtColor(cv2.resize(face, (48, 48)), cv2.COLOR_BGR2GRAY)
                    prediction = model.predict(np.expand_dims(np.expand_dims(gray_face, -1), 0))
                    
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    cv2.putText(frame, emotion_dict[np.argmax(prediction)], 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()