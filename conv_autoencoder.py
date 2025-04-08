import tensorflow as tf
from tensorflow.keras import layers, models


def build_conv_autoencoder(img_shape=(112, 112, 3), latent_dim=128):
    input_img = layers.Input(shape=img_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Flatten for Dense layer
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, activation="relu")(x)

    # Decoder
    x = layers.Dense(28 * 28 * 64, activation="relu")(x)
    x = layers.Reshape((28, 28, 64))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    output_img = layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)

    conv_autoencoder = models.Model(input_img, output_img)
    conv_autoencoder.compile(optimizer="adam", loss="mse")

    return conv_autoencoder
