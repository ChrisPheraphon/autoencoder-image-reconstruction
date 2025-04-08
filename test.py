import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_datasets

# โหลดโมเดลที่ Train แล้ว
model = tf.keras.models.load_model("trained_CONV_autoencoder.h5")

# โหลด Dataset
_, _, test_dataset = load_datasets()

# ทดสอบโมเดล
for images, _ in test_dataset.take(1):  # เอามา 1 batch
    reconstructed_images = model.predict(images)

    # Denormalize ภาพ (ถ้าโมเดลใช้ sigmoid output)
    images = images.numpy() * 255.0  # คืนค่ากลับเป็น 0-255
    reconstructed_images = np.clip(reconstructed_images * 255.0, 0, 255).astype(np.uint8)  # ป้องกันค่าเกิน

    # แสดงภาพต้นฉบับและที่ reconstruct
    plt.figure(figsize=(10, 4))
    for i in range(5):  # แสดง 5 ภาพแรก
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.axis("off")
        if i == 0:
            ax.set_title("Original")

        ax = plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed_images[i])
        plt.axis("off")
        if i == 0:
            ax.set_title("Reconstructed")

    plt.show()
    break
