import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


DATA_PATH = Path("./ice-cream.csv")
MODEL_PATH = Path("./model.pkl")


def load_and_prepare_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load data, handle missing values, and return feature/target columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names for resilient matching.
    normalized = {col.lower().strip(): col for col in df.columns}

    if "temperature" not in normalized:
        raise ValueError("Dataset must include a 'Temperature' column.")

    target_col = None
    for candidate in ["sales", "icecreamssold"]:
        if candidate in normalized:
            target_col = normalized[candidate]
            break

    if target_col is None:
        raise ValueError("Dataset must include a target column named 'Sales' or 'IceCreamsSold'.")

    feature_col = normalized["temperature"]

    # Keep only required columns and coerce to numeric.
    model_df = df[[feature_col, target_col]].copy()
    model_df[feature_col] = pd.to_numeric(model_df[feature_col], errors="coerce")
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")

    # Drop rows with missing values in required fields.
    model_df = model_df.dropna(subset=[feature_col, target_col])

    if model_df.empty:
        raise ValueError("No valid rows left after preprocessing.")

    x = model_df[[feature_col]]
    y = model_df[target_col]
    return x, y


def train_and_save_model() -> None:
    """Train a linear regression model and persist it to disk."""
    x, y = load_and_prepare_data(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    score = model.score(x_test, y_test)
    print(f"Model trained successfully. Test R^2: {score:.4f}")
    print(f"Saved model to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train_and_save_model()
