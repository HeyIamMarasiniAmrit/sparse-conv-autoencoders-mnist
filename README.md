# Sparse Convolutional Autoencoders on MNIST

Implementation of several autoencoder variants in TensorFlow/Keras trained on the MNIST dataset:

- Basic Convolutional Autoencoder
- Sparse Convolutional Autoencoder (with KL-divergence activity regularizer)
- Stacked Dense Autoencoder
- Denoising Autoencoder (noise visualization + reconstruction setup)

## Features

- Encoder & Decoder built as separate models and then stacked
- Latent space of shape `(7, 7, 8)`
- Sparsity constraint via KL divergence toward target ρ ≈ 30 / (28×28)
- Binary cross-entropy + MAE metrics
- Reconstruction visualizations (original vs reconstructed)
- Simple noise injection example for denoising

## Requirements

```bash
tensorflow >= 2.x
matplotlib
numpy

# Data is loaded and normalized inside the notebook
# Train the sparse convolutional autoencoder
history = autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

Model Architecture (Sparse Conv AE)
Encoder

Conv2D(16) → MaxPool → Conv2D(8) → MaxPool (with KL regularizer)

Decoder

Conv2DTranspose(8) → Conv2DTranspose(16) → Conv2D(1, sigmoid)

Results
After training you can generate side-by-side original vs reconstructed digits and inspect the sparse latent activations.
