import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
a = np.array([2 , 4, 5])
ap=tf.constant(a)
print(ap)
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
# Visualize it
plt.scatter(X, y);
plt.show(block=True)
plt.interactive(False)

tf.random.set_seed(42)
model = tf.keras.Sequential([
  tf.keras.layers.Input([1]),
  tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])
model.fit(X, y, epochs=5)
#pred = model.predict([27])
#print(pred)

### plot the values 2 times
#plt.figure(figsize=(10, 7))
#plt.scatter(X_train, y_train, c='b', label='Training data')
#plt.scatter(X_test, y_test, c='g', label='Testing data')
#plt.legend();

"""def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds):
   
  Plots training data, test data and compares predictions.
  
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend();"""
model.trainable = False
model.save("/Users/apple/Downloads/tensorflow notes/best_model_HDF5_format.h5")
model_load = tf.keras.models.load_model("/Users/apple/Downloads/tensorflow notes/best_model_HDF5_format.h5")
model_load.summary()

"""download from colab if its a directory
from google.colab import files
files.download("best_model_HDF5_format.h5")"""