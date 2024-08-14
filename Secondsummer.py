import cv2

print("Hello world")
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import os
class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB format
    img_resized = img.resize((32, 32))  # Resize to 32x32
    img_array = np.array(img_resized)  # Convert to NumPy array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def drop(event):
    # Get the file path from the drop event
    file_path = event.data.strip('{}')
    if os.path.isfile(file_path):
        try:
            # Open the image file
            img = Image.open(file_path)
            img.thumbnail((50, 50))  # Resize the image to fit in the label

            # Convert image to a format Tkinter can use
            img_tk = ImageTk.PhotoImage(img)

            # Update the label with the new image
            image_label.config(image=img_tk)
            image_label.image = img_tk  # Keep a reference to avoid garbage collection
            img_preprocessed = preprocess_image(file_path)


            # Predict using the model
            prediction = model.predict(img_preprocessed)
            predicted_class = class_names[np.argmax(prediction)]

            print(f'The Result is probably: {predicted_class}')



        except Exception as e:
            print(f"Error loading image: {e}")

# Create the main window
root = TkinterDnD.Tk()
root.title("Drag and Drop Image")
root.geometry("500x500")

# Create a label to display the image
image_label = tk.Label(root, text="Drag an image here", bg="lightgrey")
image_label.pack(expand=True, fill=tk.BOTH)

# Register the window for file drops
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)




images = tf.keras.datasets.cifar10.load_data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)

loss, accuracy

print(loss)
print(accuracy)
model.export('images.model')
root.mainloop()






