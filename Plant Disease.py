import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

predictions = model.predict(test_images)

import numpy as np
print("Predicted:", np.argmax(predictions[0]))
print("Actual:", test_labels[0])

img = image.load_img("shoe.jpg", target_size=(28,28), color_mode='grayscale')
img = image.img_to_array(img)
img = img/255.0

# 🔥 Important fix
img = 1 - img

img = img.reshape(1,28,28,1)

pred = model.predict(img)
print("Predicted:", np.argmax(pred))