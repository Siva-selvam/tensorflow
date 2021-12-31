import tensorflow as tf
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)
circles = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":y})
print(circles.head())
print(circles.label.value_counts())
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu);
plt.show(block=True)
plt.interactive(False)
print(X.shape, y.shape)

tf.random.set_seed(42)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.keras.activations.relu), # hidden layer 1, ReLU activation
  tf.keras.layers.Dense(4, activation=tf.keras.activations.relu), # hidden layer 2, ReLU activation
  tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid) # ouput layer, sigmoid activation
])
#uncomment this line and update the values
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) , callbacks=[lr_scheduler]
model.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy'])

history = model.fit(X, y, epochs=100, verbose=0)

pd.DataFrame(history.history).plot()
plt.title("Model_8 training curves")
plt.show(block=True)
plt.interactive(False)

lrs = 1e-4 * (10 ** (np.arange(100)/20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");
plt.show(block=True)
plt.interactive(False)

y_pred = model.predict(X)   #always add a dim, use squeeze
print(tf.squeeze(tf.round(y_pred)))
print(confusion_matrix(y, tf.squeeze(tf.round(y_pred))))