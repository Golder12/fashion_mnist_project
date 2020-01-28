import tensorflow as tf 
from tensorflow import keras
from PIL import Image

fashion_mnist = tf.keras.datasets.fashion_mnist 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=500)

test_loss= model.evaluate(test_images, test_labels)

#file= Image.open("shoe.jpeg")
#predictions = model.predict(file)

#print(predictions)