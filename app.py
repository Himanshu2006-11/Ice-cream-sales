import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request


MODEL_PATH = Path("./model.pkl")

app = Flask(__name__)


def load_model(model_path: Path):
    """Load the trained model from disk."""
    if not model_path.exists():
        # Auto-train once if the serialized model is missing.
        from train_model import train_and_save_model

        train_and_save_model()

    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


model = load_model(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    temperature_input = request.form.get("temperature", "").strip()

    if not temperature_input:
        return render_template(
            "index.html", error="Please enter a temperature value."
        )

    try:
        temperature = float(temperature_input)
    except ValueError:
        return render_template(
            "index.html", error="Invalid input. Please enter a numeric temperature."
        )

    prediction_input = pd.DataFrame({"Temperature": [temperature]})
    prediction = model.predict(prediction_input)[0]

    return render_template(
        "index.html",
        prediction=round(float(prediction), 2),
        temperature=temperature,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
