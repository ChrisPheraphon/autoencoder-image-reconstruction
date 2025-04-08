import tensorflow as tf

def load_datasets(dataset_path="dataset", img_size=(112, 112), batch_size=32):
    def normalize(x):  # ฟังก์ชัน Normalize ค่าให้อยู่ในช่วง 0-1
        return x / 255.0

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + "/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode=None  # ไม่ใช้ labels
    ).map(lambda x: (normalize(x), normalize(x)))  # Normalize

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + "/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode=None
    ).map(lambda x: (normalize(x), normalize(x)))

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + "/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode=None
    ).map(lambda x: (normalize(x), normalize(x)))

    return train_dataset, val_dataset, test_dataset
