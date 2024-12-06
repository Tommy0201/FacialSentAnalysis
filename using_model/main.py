import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) ==2:
            inputs = tf.reshape(inputs,(-1, 1, 1, inputs.shape[-1]))
        attn_map = self.conv(inputs)
        return inputs * attn_map
    

# Define the EfficientNet model with attention
def create_model(num_classes=7):
    base_model = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=(48,48,3))
    base_model.trainable = False  # Freeze the base model weights

    inputs = tf.keras.Input(shape=(48,48,1))
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x, training=False)
    x = SpatialAttention()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs, outputs)
    return model
   
def face_mp_detection(model, emotion_dict, frame_skip_rate):
    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    frame_skip_rate = frame_skip_rate

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
    
    
if __name__ == "__main__":
    
    model = create_model()
    model.load_weights('training_models/eNet-model/model_checkpoint_eNet.keras')
    cv2.ocl.setUseOpenCL(False)

    # emotion_dict = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happy", 5: "Sad", 6: "Angry", 7: "Neutral"}
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    frame_skip = 5
    
    face_mp_detection(model,emotion_dict,frame_skip)