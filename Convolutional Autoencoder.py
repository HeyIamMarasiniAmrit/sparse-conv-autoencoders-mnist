import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

encoder_inputs = keras.Input(shape=(28, 28, 1), name="encoder_input")
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
encoder_outputs = layers.MaxPooling2D((2, 2), padding="same", name="encoder_output")(x)

encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")


decoder_inputs = keras.Input(shape=(7, 7, 8), name="decoder_input")
x = layers.Conv2DTranspose(8, (3, 3), strides=2, activation="relu", padding="same")(decoder_inputs)
x = layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="decoder_output")(x)

decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")


autoencoder_inputs = keras.Input(shape=(28, 28, 1), name="autoencoder_input")
latent_rep = encoder(autoencoder_inputs)
autoencoder_outputs = decoder(latent_rep)

autoencoder = keras.Model(autoencoder_inputs, autoencoder_outputs, name="autoencoder")


autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mae"])

history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)