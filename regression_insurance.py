import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(insurance.head())
print(insurance.info())
insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot.head())
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tf.random.set_seed(42)

insurance_model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(100), # 100 units
  tf.keras.layers.Dense(10), # 10 units
  tf.keras.layers.Dense(1) # 1 unit (important for output layer)
])

insurance_model_1.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't
                          metrics=['mae'])

history = insurance_model_1.fit(X_train, y_train, epochs=100, verbose=0)

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs");
plt.show(block=True)
plt.interactive(False)
