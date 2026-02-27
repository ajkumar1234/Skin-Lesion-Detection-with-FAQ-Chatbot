import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1️⃣ Model Building
# -----------------------------

input_shape = (128, 128, 3)

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze base models
for model in [resnet_model, inception_model, efficientnet_model]:
    model.trainable = False

input_tensor = Input(shape=input_shape)

resnet_features = GlobalAveragePooling2D()(resnet_model(input_tensor))
inception_features = GlobalAveragePooling2D()(inception_model(input_tensor))
efficientnet_features = GlobalAveragePooling2D()(efficientnet_model(input_tensor))

x = Concatenate()([resnet_features, inception_features, efficientnet_features])
x = Dense(256, activation='relu')(x)
output = Dense(8, activation='softmax')(x)

ensemble_model = Model(inputs=input_tensor, outputs=output)

ensemble_model.compile(
    optimizer=Adam(learning_rate=0.001),   # ✅ fixed (lr → learning_rate)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 2️⃣ Data Generator
# -----------------------------

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(   # ✅ was missing
    'dataset/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# -----------------------------
# 3️⃣ Training
# -----------------------------

history = ensemble_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    verbose=1
)

# -----------------------------
# 4️⃣ Save Model (Correct Way)
# -----------------------------

ensemble_model.save("ensemble_model.h5")  # ✅ Correct saving method

# -----------------------------
# 5️⃣ Plot Accuracy
# -----------------------------

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# -----------------------------
# 6️⃣ Confusion Matrix
# -----------------------------

validation_generator.shuffle = False
predictions = ensemble_model.predict(validation_generator)

y_true = validation_generator.classes
y_pred = np.argmax(predictions, axis=1)

confusion_mtx = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_true, y_pred))

# -----------------------------
# 7️⃣ Single Image Prediction
# -----------------------------

model = tf.keras.models.load_model("ensemble_model.h5")

image_path = "4.jpg"
img = image.load_img(image_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0   # ✅ match training rescaling

predictions = model.predict(img_array)

class_labels = list(train_generator.class_indices.keys())
predicted_class = class_labels[np.argmax(predictions)]

print("Predicted Class:", predicted_class)
print("Probabilities:", predictions)
