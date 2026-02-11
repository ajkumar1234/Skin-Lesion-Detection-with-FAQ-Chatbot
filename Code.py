import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
input_shape = (128, 128, 3)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in resnet_model.layers:
    layer.trainable = False

for layer in inception_model.layers:
    layer.trainable = False

for layer in efficientnet_model.layers:
    layer.trainable = False
input_tensor = Input(shape=input_shape)
resnet_features = GlobalAveragePooling2D()(resnet_model(input_tensor))
inception_features = GlobalAveragePooling2D()(inception_model(input_tensor))
efficientnet_features = GlobalAveragePooling2D()(efficientnet_model(input_tensor))
x = Concatenate()([resnet_features, inception_features, efficientnet_features])
x = Dense(256, activation='relu')(x)
output = Dense(8, activation='softmax')(x)
ensemble_model = Model(inputs=input_tensor, outputs=output)
ensemble_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
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
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical'
)
history = ensemble_model.fit(
    train_generator,
    epochs=100,  
    validation_data=validation_generator,
    verbose=1
)
import pickle
from multiprocessing import Lock

with open("model.pkl", "wb") as model_file:
    pickle.dump(ensemble_model, model_file)
    tf.saved_model.save(ensemble_model, 'ensemble_model')
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.get_logger().setLevel('ERROR')
batch_size = 32
num_classes = 7 

loaded_model = tf.saved_model.load('ensemble_model')

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False 
) 


y_true = []
y_pred = []

for i in range(len(train_generator)):
    batch_images, batch_labels = train_generator[i]
    y_true.extend(np.argmax(batch_labels, axis=1))
    
   
    batch_predictions = loaded_model(batch_images)
    y_pred.extend(np.argmax(batch_predictions, axis=1))

confusion_mtx = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion'],
            yticklabels=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Training Data)')
plt.show()
print(classification_report(y_true, y_pred, target_names=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion']))
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
tf.get_logger().setLevel('ERROR')

#loaded_model = tf.compat.v2.saved_model.load('ensemble_model', None)
loaded_model = tf.saved_model.load('ensemble_model')

image_path = '4.jpg'  

img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array = preprocess_input(img_array) 

predictions = loaded_model(img_array, training=False)  

class_labels = ['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','Melanocytic Nevi','healthy skin','melonama','Vascular skin lesion'] 
predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]

print("Predicted Class Label:", predicted_class_label)
print("Class Probabilities:", predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion'],
            yticklabels=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Training Data)')
plt.show()
print(classification_report(y_true, y_pred, target_names=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion']))
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
tf.get_logger().setLevel('ERROR')

#loaded_model = tf.compat.v2.saved_model.load('ensemble_model', None)
loaded_model = tf.saved_model.load('ensemble_model')

image_path = '4.jpg'  

img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array = preprocess_input(img_array) 
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array = preprocess_input(img_array) 
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.get_logger().setLevel('ERROR')
batch_size = 32
num_classes = 7 

loaded_model = tf.saved_model.load('ensemble_model')

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
class_labels = ['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','Melanocytic Nevi','healthy skin','melonama','Vascular skin lesion'] )
predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]

print("Predicted Class Label:", predicted_class_label)
print("Class Probabilities:", predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion'],
            yticklabels=['Actinic Keratoses', 'Basal Cell Carcinoma','Benign Keratosis', 'Dermatofibroma','healthy skin','Melanocytic Nevi','melonama','Vascular skin lesion'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Training Data)')
