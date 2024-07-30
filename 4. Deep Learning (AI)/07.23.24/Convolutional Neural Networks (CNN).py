#!/usr/bin/env python
# coding: utf-8

# # Image Classification with Convolutional Neural Networks using CIFAR-10 Dataset
# 
# ### Project Overview
# 
# In this project, you will develop a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32 x 32 color images in 10 different classes, with 6,000 images per class. There are 50,000 training images and 10,000 testing images. The goal of this project is to build a CNN model that can accurately classify these images into their respective categories.
# 
# **Objectives**
# - Load and preprocess the CIFAR-10 dataset.
# - Visualize the dataset to understand the variety and distribution of images.
# - Build and train a CNN model for image classification.
# - Evaluate the model's performance on the test set.
# - Visualize the model's predictions on test images.

# In[5]:


#Install Important Library
get_ipython().system('pip install tensorflow numpy matplotlib')


# In[7]:


#Import all libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[9]:


#Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#Verify the data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5,i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
    
plt.show()


# ### Build the CNN Model
# 
# Design a CNN model that can learn the features of the images and classify them into the correct categories. The model typically includes convolutional layers, pooling layers, and dense (fully connected) layers.

# In[11]:


#Build the Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))


# In[13]:


model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])


# The model.compile method configures the model for training by specifying the optimizer, loss function, and metrics. Here's an explanation of each parameter in the given code:
# 
# 1. Optimizer: 'adam'
# 
# The optimizer parameter specifies the optimization algorithm to use during training. In this case, 'adam' stands for Adaptive Moment Estimation, which is an efficient and popular optimization algorithm. It adjusts the learning rate during training, which can help the model converge faster and avoid getting stuck in local minima.
# 
# *Key features of Adam:*
# - Combines the advantages of two other popular optimizers: AdaGrad and RMSProp.
# - Maintains per-parameter learning rates that are adapted based on the average of the first and second moments of the gradients.
# - Generally works well in practice and requires little tuning.
# 
# 2. Loss Function: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 
# The loss function parameter specifies the objective function that the model should minimize during training. Here, tf.keras.losses.SparseCategoricalCrossentropy is used, which is appropriate for multi-class classification problems where the labels are integers.
# 
# *Key aspects of SparseCategoricalCrossentropy:*
# - Computes the cross-entropy loss between the true labels and the predicted labels.
# - from_logits=True indicates that the model's output values are raw logits (i.e., the model's output layer does not include a softmax activation function). This parameter tells the loss function to apply the softmax activation function internally before computing the loss.
# 
# **Why use SparseCategoricalCrossentropy:**
# - Efficient for integer labels.
# - Automatically applies softmax activation when from_logits=True is set.
# 
# 3. Metrics: ['accuracy']
# 
# The metrics parameter specifies the metrics to be evaluated by the model during training and testing. In this case, ['accuracy'] indicates that we want to monitor the accuracy of the model.
# 
# *Accuracy Metric:*
# - Measures the proportion of correctly classified samples.
# - For classification problems, accuracy is a common metric to evaluate the performance of the model.

# In[15]:


history = model.fit(train_images, train_labels, epochs = 10, 
                    validation_data = (test_images, test_labels))


# 1. train_images and train_labels
# 
# These are the training data and corresponding labels that the model will learn from. The train_images are the input images, and train_labels are the true labels for these images.
# 
# 2. epochs = 10
# 
# The epochs parameter specifies the number of complete passes through the entire training dataset. In this case, the model will be trained for 10 epochs. Each epoch involves the model processing all training images and updating its weights accordingly.
# 
# 3. validation_data = (test_images, test_labels)
# 
# The validation_data parameter is used to provide a separate dataset on which the model will be evaluated at the end of each epoch. This allows you to monitor the model's performance on unseen data (validation data) during training.
# 
# - test_images: The input images for validation.
# - test_labels: The true labels for the validation images.
# 
# Using validation data helps in understanding how well the model generalizes to new data and can help in early stopping if the model starts to overfit.
# 
# **What happens during model.fit:**
# 
# => Training Loop:
# - For each epoch, the model processes each batch of training data.
# - It computes the predictions for each batch, compares them to the true labels, and calculates the loss.
# - The optimizer updates the model's weights based on the gradients of the loss.
# - This process continues for all batches in the training data.
# 
# => Validation Loop:
# - At the end of each epoch, the model is evaluated on the validation data.
# - The model computes predictions for the validation set and calculates the validation loss and metrics (e.g., accuracy).
# - The validation results are used to monitor the model's performance on unseen data.
# 
# => history Object
# - The history object returned by model.fit contains information about the training process, including the values of the loss and metrics at each epoch for both the training and validation datasets. This information is stored in a History object and can be used to plot the learning curves and analyze the training progress.

# In[19]:


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(f'\nTest accuracy: {test_acc}')


# ##### Interpretation
# - Model Performance: An accuracy of 69.75% indicates a reasonably good performance, especially considering that the CIFAR-10 dataset is relatively challenging due to its diverse set of images and classes. However, there's room for improvement. For high-performance tasks, you might aim for higher accuracy.
# 
# - Loss Value: The loss value of 0.8920, while informative, is not as directly interpretable as accuracy without additional context. It indicates how well the model's predictions match the true labels, with lower values indicating better matches.

# In[22]:


plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.show()


# In[24]:


model.save('my_cifar10_model.h5')


# The model.save('my_cifar10_model.h5') command is used to save the entire trained model to a file. Here's a detailed explanation of this process:
# 
# 1. Purpose of Saving a Model
# *Saving a model is important for various reasons:*
# - Reusability: You can reuse the trained model later without having to retrain it from scratch, which saves time and computational resources.
# - Deployment: You can deploy the saved model to a production environment where it can be used to make predictions on new data.
# - Sharing: You can share the trained model with others who can load and use it without needing access to the original training data or training code.
# - Backup: It provides a way to back up your work, ensuring you do not lose the trained model due to accidental data loss or changes.
# 
# 2. What Does model.save Do?
# 
# The model.save function saves the entire architecture, weights, and training configuration (including the optimizer, loss, and metrics) to a single file. In this case, the file is named my_cifar10_model.h5.
# 
# 3. File Format: .h5
# 
# The .h5 extension indicates that the model is saved in the HDF5 (Hierarchical Data Format version 5) format. HDF5 is a file format that is well-suited for storing large amounts of numerical data. It's efficient and widely used in scientific computing.
