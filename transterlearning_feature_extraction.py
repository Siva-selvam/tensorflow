import tensorflow as tf
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
#model.save("/Users/apple/Downloads/tensorflow notes/best_model_HDF5_format.h5")
model_load = tf.keras.models.load_model("/Users/apple/Downloads/tensorflow notes/best_model_HDF5_format.h5")
model_load.include_top = False
model_load.trainable = False
print(model_load.summary())
model = tf.keras.Sequential([
    model_load,  # use the feature extraction layer as the base
    tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')  # create our own output layer
  ])
model.build(input_shape=[1,])
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])
model.fit(X, y, epochs=5)
print(model.summary())
print(model.predict([11]))