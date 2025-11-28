import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from data_loader import get_datasets
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--model_path", type=str, default="saved_models/best_model.h5")
    return p.parse_args()

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()

    print("📂 Loading dataset...")
    _, _, test_ds = get_datasets(args.data_dir)

    print("🧠 Loading model...")
    model = tf.keras.models.load_model(args.model_path)

    y_true = []
    y_pred = []

    print("🔍 Running evaluation on test set...")

    for images, labels in test_ds:
        preds = model.predict(images)
        preds = (preds.flatten() > 0.5).astype(int)

        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())

    print("\n📊 CONFUSION MATRIX:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plot_confusion_matrix(cm, ["Bird", "Drone"])

    print("\n📄 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=["Bird", "Drone"]))


if __name__ == "__main__":
    main()