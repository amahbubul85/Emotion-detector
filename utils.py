import tensorflow as tf
import numpy as np
from PIL import Image

# Load model architecture
def load_model():
    input_img = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.ZeroPadding2D((3, 3))(input_img)
    x = tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn0')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pool0')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(x)
    
    model = tf.keras.Model(inputs=input_img, outputs=x)
    model.load_weights("model/happy_model.weights.h5")
    return model

# Preprocess image
def preprocess_image(image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 64, 64, 3))
    return img_array

# Predict
def predict_emotion(model, image):
    processed = preprocess_image(image)
    prob = model.predict(processed)[0][0]
    return "Happy 😄" if prob > 0.5 else "Sad 😢", prob
