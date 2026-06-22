import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Dataset Paths

train_dir = "dataset/train"
test_dir = "dataset/test"

# Data Preprocessing

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(48, 48),
batch_size=64,
class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(48, 48),
batch_size=64,
class_mode='categorical'
)

# CNN Model

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# Compile Model

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

# Train Model

history = model.fit(
train_generator,
validation_data=test_generator,
epochs=10
)

# Evaluate Model

loss, accuracy = model.evaluate(test_generator)

print("Accuracy:", accuracy)

# Accuracy Graph

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Save Model

model.save("emotion_recognition_model.h5")

print("Model saved successfully!")
