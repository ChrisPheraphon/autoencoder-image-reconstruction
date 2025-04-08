import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_datasets
from autoencoder import build_autoencoder
from conv_autoencoder import build_conv_autoencoder

# โหลด Dataset
train_dataset, val_dataset, test_dataset = load_datasets()

# เลือกโมเดลที่จะเทรน
use_cnn = True

if use_cnn:
    model = build_conv_autoencoder()
else:
    model = build_autoencoder()

# ตั้งค่า Early Stopping เพื่อลด Overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]  # ใช้ Early Stopping
)

# บันทึกโมเดล
model.save("trained_CONV_autoencoder.h5")
print("✅ The model has been saved successfully: trained_autoencoder.h5")

# ==================== PLOT กราฟ LOSS ====================
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Train Loss", color="blue")
plt.plot(history.history["val_loss"], label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")  # บันทึกกราฟเป็นไฟล์
plt.show()
print("✅ Loss plot saved as loss_plot.png")

