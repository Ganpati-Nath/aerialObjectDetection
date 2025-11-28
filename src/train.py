import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from data_loader import get_datasets
from model_builder import build_transfer_model
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_dir", type=str, default="saved_models")
    return p.parse_args()


def main():
    args = parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("📂 Saving models to:", save_dir)

    print("📂 Loading dataset...")
    train_ds, val_ds, test_ds = get_datasets(args.data_dir, batch_size=args.batch_size)

    print("🧠 Building model...")
    model = build_transfer_model()

    optimizer = tf.keras.optimizers.Adam(1e-4)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("🚀 Training started...\n")

    # 🔥 FIX: Checkpoint uses .h5 (SAFE FORMAT)
    best_model_path = os.path.join(save_dir, "best_model.h5")
    final_model_path = os.path.join(save_dir, "final_model.keras")  # OK

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_format='h5',       # 🔥 FIXED LINE
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 🔥 FINAL MODEL — NEW .keras FORMAT (NO CHECKPOINT HERE)
    model.save(final_model_path)
    print(f"\n✅ FINAL MODEL SAVED AT: {final_model_path}")
    print(f"✅ BEST CHECKPOINT SAVED AT: {best_model_path}")


if __name__ == "__main__":
    main()