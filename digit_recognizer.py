import tensorflow as tf 
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

current_directory = os.path.dirname(os.path.abspath(__file__))
final_directory = os.path.join(current_directory, 'saved_model/my_model12') #model 11 is not a bad model 12 works better
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

mnist = tf.keras.datasets.mnist

(training_data,training_labels), (test_data, test_labels) = mnist.load_data()
training_data, test_data = training_data/255, test_data/255

print(test_data.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(training_data, training_labels, epochs=100)
model.evaluate(test_data,test_labels)
predictions = model.predict(test_data)
np.set_printoptions(suppress=True)
print(test_labels[0])
print(predictions[0])


model.save(final_directory)