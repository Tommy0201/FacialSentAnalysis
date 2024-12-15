import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, MultiHeadAttention, LayerNormalization
from tensorflow.keras.applications import ResNet50


KERNAL_SIZE = 7 #only if enetB0-model-2 is used else = 7


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=KERNAL_SIZE, padding="same", activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.reshape(inputs,(-1, 1, 1, inputs.shape[-1]))
        attn_map = self.conv(inputs)
        return inputs * attn_map
    

# Define the EfficientNet model with attention
def create_eNetB0_model(num_classes=7):
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

def create_eNetB4_model(num_classes=7):
    base_model = EfficientNetB4(include_top=False, weights="imagenet", pooling="avg", input_shape=(224,224,3))
    base_model.trainable = False  # Freeze the base model weights

    inputs = tf.keras.Input(shape=(224,224,1))
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x, training=False)
    x = SpatialAttention()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs, outputs)
    return model

def create_customize_model():

    inputs = Input(shape=(48, 48, 1))

    # First convo block
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second convo block
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Third convo block (added to handle larger input)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Reshape for multi-head attention
    x = Reshape((-1, 128))(x) 

    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)  
    attention_output = LayerNormalization()(attention_output)  

    # Fully connected layers
    x = Flatten()(attention_output)
    x = Dense(128, activation='relu')(x)  
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax', dtype='float32')(x) 

    model = Model(inputs=inputs, outputs=outputs)
    return model
   
def create_resNet_model(num_classes=7):
    # Load ResNet-50 without the top layers
    base_model = ResNet50(include_top=False, weights="imagenet", pooling=None, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model weights

    inputs = tf.keras.Input(shape=(224, 224, 1))    
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x, training=False)    
    x = SpatialAttention()(x)    
    x = layers.GlobalAveragePooling2D()(x)    
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs, outputs)
    return model

def face_mp_detection(model, emotion_dict, frame_skip_rate, img_size):
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
                        # gray_face = cv2.cvtColor(cv2.resize(face, (48, 48)), cv2.COLOR_BGR2GRAY)
                        predict_image = cv2.cvtColor(cv2.resize(face, img_size), cv2.COLOR_BGR2GRAY)
                        prediction = model.predict(np.expand_dims(np.expand_dims(predict_image, -1), 0))
                        
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                        cv2.putText(frame, emotion_dict[np.argmax(prediction)], 
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame_counter +=1
            cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()   

    
if __name__ == "__main__":
    
    #STAGE 1: emo-model and eNet-model are the two first train with 48x48, image label 1-7
    #STAGE 2: emo-model-resize, eNetB4, resNet50 are the three models trained with 224x224, image label 1-7
    #STAGE 3: emo-model-2, eNetB0-2 are the two models trained with 48x48, image label 0-6 --> BOTH PERFORM BADLY
    
    model_name = "customize"
    image_size = (48,48)


    if "eNetB0" == model_name: #enetB0-2, 48x48, 1-0, kernal_size = 5  --> PERFORM BADLY
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        model = create_eNetB0_model()
        model.load_weights('training_models/stage3/eNetB0-model-2/final_model_eNet.keras')
        
        
    elif "eNetB4" == model_name: #enetB4, 224x224, 1-7, kernal_size = 7   --> PERFORM PRETTY WELL
        emotion_dict = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happy", 5: "Sad", 6: "Angry", 7: "Neutral"}
        model = create_eNetB4_model()
        model.load_weights('training_models/stage2/eNetB4-model/model_checkpoint_eNetB4.keras') 
        image_size = (224,224)
        
    elif "customize" == model_name:  #customize model 48x48, image label 0-6, kernal_size = 7, GOT MIXED UP 
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        model = create_customize_model()
        model.load_weights('training_models/stage3/emo-model-2/final_model_2.keras') 

    elif "eNetB42" == model_name:  #customize model 224x224, image label 0-6, kernal_size = 7
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        model = create_eNetB4_model()
        image_size = (224,224)
        model.load_weights('training_models/stage3/eNetB4-model2/model_checkpoint_eNetB4.keras') 
    
    elif "resNet" == model_name: #resNet model 224x224, image label 0-6, kernal_size = 5, #KINDA MIXED
        emotion_dict = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happy", 5: "Sad", 6: "Angry", 7: "Neutral"}
        model = create_resNet_model()
        model.load_weights('training_models/stage2/resNet50-model/model_checkpoint_resNet50.keras') 
        image_size = (224,224)
        # False since always happy`
    
    cv2.ocl.setUseOpenCL(False)
    
    frame_skip = 1
    face_mp_detection(model,emotion_dict,frame_skip, img_size=image_size)
