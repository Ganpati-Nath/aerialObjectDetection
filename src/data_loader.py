import tensorflow as tf
import os

def get_datasets(base_dir, img_size=(224,224), batch_size=32, seed=123):
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')
    test_dir  = os.path.join(base_dir, 'test')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds