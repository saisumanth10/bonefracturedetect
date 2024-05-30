#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy -q')
get_ipython().system('pip install pandas -q')
get_ipython().system('pip install matplotlib -q')
get_ipython().system('pip install tensorflow -q')

get_ipython().system('pip install opendatasets -q')


# In[2]:


# import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import opendatasets as od


# ### Load Dataset

# In[3]:


# download dataset

od.download("https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays")


# In[4]:


BATCH_SIZE = 32
IMAGE_SIZE = (224,224)


# In[5]:


train_data_dir = "C:/Users/reddy/Downloads/archive/archive (6)/train"
test_data_dir = "C:/Users/reddy/Downloads/archive/archive (6)/val"

train_data = tf.keras.utils.image_dataset_from_directory(train_data_dir,
                                                         batch_size=BATCH_SIZE,
                                                         image_size=IMAGE_SIZE,
                                                         subset='training',
                                                         validation_split=0.1,
                                                         seed=42)

validation_data = tf.keras.utils.image_dataset_from_directory(train_data_dir,
                                                         batch_size=BATCH_SIZE,
                                                         image_size=IMAGE_SIZE,
                                                         subset='validation',
                                                         validation_split=0.1,
                                                         seed=42)

test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir,
                                                         batch_size=BATCH_SIZE,
                                                         image_size=IMAGE_SIZE)


# In[6]:


class_names = train_data.class_names
class_names


# In[7]:


for image_batch,label_batch in train_data.take(1):
    print(image_batch.shape)
    print(label_batch.shape)


# In[8]:


# plot data sample
plt.figure(figsize=(10,4))
for image,label in train_data.take(1):
    for i in range(10):
        ax = plt.subplot(2,5,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(class_names[label[i]])
        plt.axis('off')


# ### Scaling Images

# In[9]:


for image,label in train_data.take(1):
    for i in range(1):
      print(image)


# In[10]:


train_data = train_data.map(lambda x,y:(x/255,y))
validation_data = validation_data.map(lambda x,y:(x/255,y))
test_data = test_data.map(lambda x,y:(x/255,y))


# ### Data Augmentation

# In[11]:


data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal",input_shape=(224,224,3)),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
  ]
)


# ### Model Building

# In[12]:


model = tf.keras.models.Sequential()

model.add(data_augmentation)

model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[13]:


model.summary()


# In[14]:


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# ### Model Training

# In[15]:


start_time = time.time()

history = model.fit(train_data,
                    epochs=20,
                    validation_data=validation_data)

end_time = time.time()


# In[16]:


print(f'Total time for training {(end_time-start_time):.3f} seconds')


# ### Performance Analysis

# In[17]:


fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend()
plt.show()


# In[18]:


fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend()
plt.show()


# ### Model Evaluation

# In[19]:


precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
accuracy = tf.keras.metrics.BinaryAccuracy()


# In[20]:


for batch in test_data.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)


# In[21]:


precision.result()


# In[22]:


recall.result()


# In[23]:


accuracy.result()


# ### Test

# In[24]:


get_ipython().system('pip install opencv-python -q')


# In[25]:


import cv2


# In[26]:


img = cv2.imread('C:/Users/reddy/Downloads/archive/archive (6)/train/fractured/10-rotated1-rotated1.jpg')
plt.imshow(img)
plt.show()


# In[27]:


resized_image = tf.image.resize(img, IMAGE_SIZE)
scaled_image = resized_image/255


# In[28]:


scaled_image.shape


# In[29]:


np.expand_dims(scaled_image, 0).shape


# In[30]:


yhat = model.predict(np.expand_dims(scaled_image, 0))


# In[31]:


yhat


# In[32]:


class_names


# In[33]:


if yhat > 0.5:
    print(f'{class_names[1]}')
else:
    print(f'{class_names[0]}')


# In[34]:


get_ipython().system('pip install opencv-python-headless --user -q')


# In[35]:


pip install opencv-python-headless


# In[36]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np


# In[ ]:


class FractureDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fracture Detector")

        self.canvas = tk.Canvas(root, width=800, height=400)
        self.canvas.pack()

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        img = cv2.imread(file_path)
        resized_img = cv2.resize(img, (224, 224))
        scaled_img = resized_img / 255.0  # Normalize image

        yhat = model.predict(np.expand_dims(scaled_img, axis=0))[0][0]
        predicted_class = class_names[int(round(yhat))]

        self.display_image(file_path)
        self.result_label.config(text=f"Predicted: {predicted_class}")

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        self.img_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = FractureDetectorApp(root)
    root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




