import pickle
import os

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, "pixel_classifier.pkl")

# Load the pixel classifier model
def load_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        fil_model = model.convert_to_fil_model()
    return fil_model

