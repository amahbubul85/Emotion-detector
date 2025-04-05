import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import h5py
import numpy as np

# 1. Load data
# Load training data
train_dataset = h5py.File("datasets/train_happy.h5", "r")
X_train_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
Y_train_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

# Load test data
test_dataset = h5py.File("datasets/test_happy.h5", "r")
X_test_orig = np.array(test_dataset["test_set_x"][:])     # your test set features
Y_test_orig = np.array(test_dataset["test_set_y"][:])     # your test set labels

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# 2. Define the functional model
def happy_model_functional():
    input_img = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(input_img)
    x = tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn0')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pool0')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='fc')(x)
    return tf.keras.Model(inputs=input_img, outputs=x)

# 3. Compile and train
from tensorflow.keras.callbacks import ModelCheckpoint

# 3. Compile model
model = happy_model_functional()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Define the checkpoint callback
if not os.path.exists("model"):
    os.makedirs("model")

checkpoint = ModelCheckpoint(
    filepath="model/best_model.weights.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min"
)

# 5. Train with callback
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, Y_test),
    callbacks=[checkpoint]
)
# Force save final weights manually


import tensorflow as tf

# Load happy/sad dummy faces
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "mini_dataset",
    image_size=(64, 64),
    batch_size=8,
    label_mode="binary"
)

# Split into train/test
total_batches = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * total_batches)
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

model.save_weights("model/final_model.weights.h5")
