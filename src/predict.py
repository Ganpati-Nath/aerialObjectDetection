# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# import argparse

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--model_path", type=str, default="../saved_models/best_model.h5")
#     p.add_argument("--img", type=str, required=True)
#     return p.parse_args()

# def main():
#     args = parse_args()
#     model = tf.keras.models.load_model(args.model_path)
#     img = image.load_img(args.img, target_size=(224,224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0) / 255.0
#     pred = model.predict(x)[0][0]
#     label = "drone" if pred > 0.5 else "bird"
#     print(f"Prediction: {label} ({pred:.4f})")

# if __name__ == "__main__":
#     main()

import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="saved_models/best_model.h5")
    p.add_argument("--img", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path)

    img = Image.open(args.img).convert("RGB")
    img_resized = img.resize((224, 224))

    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0][0]
    label = "Drone" if pred > 0.5 else "Bird"

    print(f"\nPrediction: {label}")
    print(f"Confidence: {pred:.4f}")

if __name__ == "__main__":
    main()