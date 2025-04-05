import tensorflow as tf
import numpy as np
from PIL import Image

# List of class labels (must match folder names)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load multi-class model architecture and weights
def load_model():
    input_img = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)  # multi-class output
    model = tf.keras.Model(inputs=input_img, outputs=x)
    
    model.load_weights("model/best_multiclass_model.weights.h5")
    return model

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((64, 64)).convert('RGB')
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 64, 64, 3))
    return img_array

# Predict emotion
def predict_emotion(model, image):
    processed = preprocess_image(image)
    probs = model.predict(processed)[0]
    predicted_index = np.argmax(probs)
    predicted_label = class_names[predicted_index]
    confidence = probs[predicted_index]
    return f"{predicted_label.capitalize()} 😃", confidence
