from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

from model import SimpleANN

app = Flask(__name__)


# --------------------------------
# 2. Locate and load saved artifacts
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "wine_ann.pth")

model = SimpleANN()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)

model.eval()

PREPROCESSING_PATH = os.path.join(
    ARTIFACT_DIR, "preprocessing.joblib"
)

metadata = joblib.load(PREPROCESSING_PATH)

scaler = metadata["scaler"]
feature_names = metadata["feature_names"]
class_names = metadata["class_names"]

# --------------------------------
# 3. Define the home page
# --------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_names=feature_names
    )


# --------------------------------
# 4. Receive user input and predict
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Read all 13 values from the form
        user_values = []

        for i in range(len(feature_names)):
            value = request.form.get(f"feature_{i}", "").strip()

            if not value:
                raise ValueError(
                    f"Please enter a value for {feature_names[i]}."
                )

            user_values.append(float(value))

        # Convert input to a 2D NumPy array
        input_data = np.array(
            user_values,
            dtype=np.float64
        ).reshape(1, -1)

        # Apply the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # Convert the scaled input into a PyTorch tensor
        input_tensor = torch.tensor(
            input_scaled,
            dtype=torch.float32
        )

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs, dim=1
            )[0]

            predicted_index = int(
                torch.argmax(probabilities).item()
            )

        predicted_class = class_names[predicted_index]

        probability_results = [
            {
                "name": class_names[i],
                "probability": round(
                    float(probabilities[i].item()) * 100, 2
                )
            }
            for i in range(len(class_names))
        ]

        return render_template(
            "index.html",
            feature_names=feature_names,
            user_values=user_values,
            prediction=predicted_class,
            probabilities=probability_results
        )

    except ValueError as error:
        return render_template(
            "index.html",
            feature_names=feature_names,
            error=str(error),
            user_values=request.form
        ), 400


if __name__ == "__main__":
    app.run(debug=True)