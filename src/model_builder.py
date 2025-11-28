# src/model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_transfer_model(input_shape=(224, 224, 3)):
    """
    Transfer learning model using EfficientNetB0 backbone.
    Keep backbone trainable (fine-tuning).
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        classifier_activation=None
    )

    base_model.trainable = True  # fine-tune

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)
    x = base_model(x, training=True)

    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="effnet_transfer")
    return model
