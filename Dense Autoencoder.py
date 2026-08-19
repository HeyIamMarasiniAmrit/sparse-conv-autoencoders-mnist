#Dense (Fully Connected) Stacked Autoencoder
stacked_autoencoder = keras.models.Sequential([
    layers.Flatten(input_shape=[28, 28, 1]),
    layers.Dense(100, activation="selu"),
    layers.Dense(30, activation="selu"),  # Bottleneck layer
    layers.Dense(100, activation="selu"),
    layers.Dense(28 * 28, activation="sigmoid"),
    layers.Reshape([28, 28, 1])
])

stacked_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
stacked_autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, validation_data=(x_test, x_test))


preds = stacked_autoencoder.predict(x_test[:8], verbose=0)

plt.figure(figsize=(20, 4))
for i in range(8):
    # Original Image
    plt.subplot(2, 8, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="binary")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed Image
    plt.subplot(2, 8, i + 9)
    plt.imshow(preds[i].reshape(28, 28), cmap="binary")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))

# Original
plt.subplot(1, 2, 1)
plt.imshow(x_test[0].reshape(28, 28), cmap="binary")
plt.title("Original")
plt.axis("off")

# Noisy Image
plt.subplot(1, 2, 2)
noise = np.random.rand(28, 28, 1) / 4.0
noisy_image = np.clip(x_test[0] + noise, 0.0, 1.0)  # values 0 देखि 1 को बिचमा राख्न clip गरिएको
plt.imshow(noisy_image.reshape(28, 28), cmap="binary")
plt.title("Noisy Input")
plt.axis("off")

plt.tight_layout()
plt.show()