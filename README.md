# Ice Cream Sales Predictor (Flask + Linear Regression)

A production-ready machine learning web application that predicts ice cream sales from temperature data using a Linear Regression model.

## Features

- Trains a Linear Regression model with pandas and scikit-learn
- Handles missing and invalid values in training data
- Saves trained model as `model.pkl`
- Flask web app with form-based prediction endpoint
- User-friendly validation and error messages for invalid input
- Deployment-ready for Render using Gunicorn and Procfile

## Tech Stack

- Python
- Flask
- pandas
- scikit-learn
- NumPy
- Gunicorn
- HTML/CSS + Bootstrap

## Project Structure

- `app.py`
- `train_model.py`
- `model.pkl`
- `requirements.txt`
- `Procfile`
- `runtime.txt`
- `templates/index.html`
- `static/style.css`
- `ice-cream.csv`

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train model:
   ```bash
   python train_model.py
   ```
4. Run app:
   ```bash
   python app.py
   ```
5. Open browser:
   - http://127.0.0.1:10000

## Input Validation

- Empty temperature input returns a clear error message.
- Non-numeric input returns a clear error message.

## Render Deployment Guide

1. Push this project to a GitHub repository.
2. In Render, click **New +** -> **Web Service**.
3. Connect your GitHub repo and select this project.
4. Use these settings:
   - Build Command: `pip install -r requirements.txt && python train_model.py`
   - Start Command: `gunicorn app:app`
5. Deploy.

The app is configured to bind to:

- Host: `0.0.0.0`
- Port: `PORT` environment variable (defaults to `10000` locally)

## Notes

- The training script supports target column names `Sales` or `IceCreamsSold`.
- Dataset path defaults to `./ice-cream.csv`.
