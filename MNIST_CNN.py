#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ##  Visualizing the Image Data

# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


x_train.shape


# In[7]:


single_image = x_train[0]


# In[8]:


single_image


# In[9]:


single_image.shape


# In[10]:


plt.imshow(single_image)


# # PreProcessing Data
# 
# We first need to make sure the labels will be understandable by our CNN.

# ## Labels

# In[11]:


y_train


# In[12]:


y_test


# Hmmm, looks like our labels are literally categories of numbers. We need to translate this to be "one hot encoded" so our CNN can understand, otherwise it will think this is some sort of regression problem on a continuous axis. Luckily , Keras has an easy to use function for this:

# In[13]:


from tensorflow.keras.utils import to_categorical


# In[14]:


y_train.shape


# In[15]:


y_example = to_categorical(y_train)


# In[16]:


y_example


# In[17]:


y_example.shape


# In[18]:


y_example[0]


# In[19]:


y_cat_test = to_categorical(y_test,10)


# In[20]:


y_cat_train = to_categorical(y_train,10)


# ### Processing X Data
# 
# We should normalize the X data

# In[21]:


single_image.max()


# In[22]:


single_image.min()


# In[23]:


x_train = x_train/255
x_test = x_test/255


# In[24]:


scaled_single = x_train[0]


# In[25]:


scaled_single.max()


# In[26]:


plt.imshow(scaled_single)


# ## Reshaping the Data
# 
# Right now our data is 60,000 images stored in 28 by 28 pixel array formation. 
# 
# This is correct for a CNN, but we need to add one more dimension to show we're dealing with 1 RGB channel (since technically the images are in black and white, only showing values from 0-255 on a single channel), an color image would have 3 dimensions.

# In[27]:


x_train.shape


# In[28]:


x_test.shape


# Reshape to include channel dimension (in this case, 1 channel)

# In[29]:


x_train = x_train.reshape(60000, 28, 28, 1)


# In[30]:


x_train.shape


# In[31]:


x_test = x_test.reshape(10000,28,28,1)


# In[32]:


x_test.shape


# # Training the Model

# In[33]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[34]:


model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

# https://keras.io/metrics/
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) # we can add in additional metrics https://keras.io/metrics/


# In[35]:


model.summary()


# In[36]:


from tensorflow.keras.callbacks import EarlyStopping


# In[37]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# ## Train the Model

# In[38]:


model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])


# ## Evaluate the Model

# In[42]:


model.metrics_names


# In[43]:


losses = pd.DataFrame(model.history.history)


# In[44]:


losses.head()


# In[45]:


losses[['accuracy','val_accuracy']].plot()


# In[46]:


losses[['loss','val_loss']].plot()


# In[50]:


print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))


# In[52]:


from sklearn.metrics import classification_report,confusion_matrix


# In[53]:


predictions = model.predict_classes(x_test)


# In[54]:


y_cat_test.shape


# In[55]:


y_cat_test[0]


# In[56]:


predictions[0]


# In[57]:


y_test


# In[59]:


print(classification_report(y_test,predictions))


# In[60]:


confusion_matrix(y_test,predictions)


# In[61]:


import seaborn as sns


# In[72]:


plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
# https://github.com/matplotlib/matplotlib/issues/14751


# # Predicting a given image

# In[75]:


my_number = x_test[0]


# In[77]:


plt.imshow(my_number.reshape(28,28))


# In[79]:


# SHAPE --> (num_images,width,height,color_channels)
model.predict_classes(my_number.reshape(1,28,28,1))


# Looks like the CNN performed quite well!
