import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(img_shape=(112, 112, 3), latent_dim=128):
    input_img = layers.Input(shape=img_shape)

    # Encoder
    x = layers.Flatten()(input_img)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(latent_dim, activation="relu")(x)

    # Decoder
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(112 * 112 * 3, activation="sigmoid")(x)
    output_img = layers.Reshape((112, 112, 3))(x)

    autoencoder = models.Model(input_img, output_img)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder
