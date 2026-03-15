import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os


# Load trained model

model_path = "plant_classifier_model"
model = load_model(model_path)


# Class names (17)
CLASS_NAMES = [
    "aloevera","banana","coconut","corn","cucumber","curcuma","eggplant",
    "guava","mango","orange","paddy","peperchili","pineapple","shallot",
    "soybeans","sweetpotatoes","watermelon"
]


# Function to predict single image
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return


    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)


    img_array = preprocess_input(img_array)


    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_name = CLASS_NAMES[class_index]
    confidence = predictions[0][class_index]

    print(f"Predicted: {class_name} ({confidence*100:.2f}%)")


# Example
if __name__ == "__main__":

    img_path = "/.."
    predict_image(img_path)
