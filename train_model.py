
# Transfer Learning
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# ✅ 1. Load train dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "mini_dataset/train",
    labels="inferred",
    label_mode="categorical",  # multi-class
    image_size=(96, 96),
    batch_size=32,
    shuffle=True,
    seed=42
)

# ✅ 2. Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "mini_dataset/test",
    labels="inferred",
    label_mode="categorical",
    image_size=(96, 96),
    batch_size=32,
    shuffle=False
)

# ✅ 3. Get class names and number of classes
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Detected classes: {class_names}")

# ✅ 4. Define transfer learning model using MobileNetV2
def transfer_emotion_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(96, 96, 3)))
    base_model.trainable = True  # Freeze base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=output)

# ✅ 5. Compile the model
model = transfer_emotion_model(num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),  # lower learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ✅ 6. Set up model checkpoint
if not os.path.exists("model"):
    os.makedirs("model")

checkpoint = ModelCheckpoint(
    filepath="model/best_transfer_model.weights.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min"
)

# ✅ 7. Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=60,
    callbacks=[checkpoint]
)



"""import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# ✅ 1. Load train dataset with all classes
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "mini_dataset/train",
    labels="inferred",
    label_mode="categorical",    # Multi-class (one-hot)
    image_size=(64, 64),
    batch_size=16,
    shuffle=True,
    seed=42
)

# ✅ 2. Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "mini_dataset/test",
    labels="inferred",
    label_mode="categorical",
    image_size=(64, 64),
    batch_size=16,
    shuffle=False
)

# ✅ 3. Get class names (automatically inferred)
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Detected classes: {class_names}")

# ✅ 4. Define multi-class CNN model
def emotion_model():
    input_img = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # ✅ FIXED

    return tf.keras.Model(inputs=input_img, outputs=x)


# ✅ 5. Compile model
model = emotion_model()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ 6. Checkpoint for best weights
if not os.path.exists("model"):
    os.makedirs("model")

checkpoint = ModelCheckpoint(
    filepath="model/best_multiclass_model.weights.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min"
)

# ✅ 7. Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=80,
    callbacks=[checkpoint]
)

# Force save final weights manually"""


"""import tensorflow as tf

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

model.save_weights("model/final_model.weights.h5")"""
