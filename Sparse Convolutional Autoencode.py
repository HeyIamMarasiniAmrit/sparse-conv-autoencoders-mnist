# १. Target Sparsity (rho) र KL Divergence Regularizer
target_sparsity = 30 / (28 * 28)
print(f"Target Sparsity (rho): {target_sparsity:.4f}")

def kl_divergence_regularizer(inputs):
    rho_hat = tf.reduce_mean(inputs, axis=0)
    eps = 1e-10
    rho = target_sparsity
    kl_div = rho * tf.math.log((rho + eps) / (rho_hat + eps)) + \
             (1 - rho) * tf.math.log((1 - rho + eps) / (1 - rho_hat + eps))
    return 1e-4 * tf.reduce_sum(kl_div)

# २. Sparse Encoder
sparse_encoder_inputs = keras.Input(shape=(28, 28, 1), name="sparse_encoder_input")
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(sparse_encoder_inputs)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
sparse_encoder_outputs = layers.MaxPooling2D(
    (2, 2),
    padding="same",
    activity_regularizer=kl_divergence_regularizer,
    name="sparse_encoder_output"
)(x)

sparse_encoder = keras.Model(sparse_encoder_inputs, sparse_encoder_outputs, name="sparse_encoder")


sparse_autoencoder_inputs = keras.Input(shape=(28, 28, 1))
sparse_latent = sparse_encoder(sparse_autoencoder_inputs)
sparse_autoencoder_outputs = decoder(sparse_latent)

sparse_autoencoder = keras.Model(sparse_autoencoder_inputs, sparse_autoencoder_outputs, name="sparse_autoencoder")
sparse_autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mae"])


sample_pred = sparse_encoder.predict(x_test[0:1])
print("Sparse Latent Shape:", sample_pred.shape)


sparse_history = sparse_autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)