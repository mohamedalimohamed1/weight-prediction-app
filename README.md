# Weight Prediction App

## Overview

Weight Prediction App is a Flask web application that estimates final weight in pounds from user-provided lifestyle and body-related inputs. The app serves a form-based interface, loads a trained machine learning pipeline with `joblib`, and returns the prediction together with basic model metrics as JSON.

The application code lives in the `app/` directory. It uses the trained SVM pipeline stored at `app/models/svm_model/svm_model_pipeline.joblib` and reads the dataset from `app/dataset/weight_change_dataset.csv`.

## Features

- Flask-based web interface for weight prediction
- Main page served from the `/` route
- Form submission handled by the `/predict` route
- Trained pipeline model loaded with `joblib`
- Prediction output in pounds
- JSON response containing prediction and evaluation metrics
- Weight converter for pounds and kilograms in the frontend

## Technologies

| Layer | Technologies |
| --- | --- |
| Backend | Python, Flask |
| Machine learning | scikit-learn, joblib |
| Data handling | pandas |
| Visualization scripts | matplotlib |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

```text
weight-prediction-app/
└── app/
    ├── app.py
    ├── dataset/
    │   └── weight_change_dataset.csv
    ├── models/
    │   ├── decision_tree_model/
    │   ├── knn_model/
    │   └── svm_model/
    │       └── svm_model_pipeline.joblib
    ├── static/
    │   ├── main.css
    │   └── main.js
    ├── templates/
    │   └── index.html
    ├── visualization/
    ├── dt.py
    ├── knn.py
    ├── svm.py
    └── graph_generator.py
```

## Installation

Create and activate a virtual environment, then install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the App

Run the Flask application from the `app/` directory so the relative model and dataset paths resolve correctly:

```powershell
cd app
python app.py
```

After starting the app locally, the main page is served by the `/` route.

## Input Features

The prediction form uses the following inputs:

- Participant ID
- Age
- Gender
- Current Weight (lbs)
- BMR (Calories)
- Daily Calories Consumed
- Daily Caloric Surplus/Deficit
- Weight Change (lbs)
- Duration (weeks)
- Physical Activity Level
- Sleep Quality
- Stress Level

## Prediction Flow

1. The `/` route renders the main prediction form.
2. The user submits the form from the browser.
3. The frontend sends the form data to `/predict`.
4. `app.py` converts the submitted values into a pandas DataFrame.
5. The feature columns are ordered to match the trained pipeline.
6. The SVM pipeline predicts final weight in pounds.
7. The app returns the prediction and model metrics as JSON.

## Model and Dataset Notes

- The loaded prediction model is `app/models/svm_model/svm_model_pipeline.joblib`.
- The dataset file is `app/dataset/weight_change_dataset.csv`.
- Additional training scripts for Decision Tree, KNN, and SVM models are present in the `app/` directory.
- Existing model and dataset files are part of the application structure and should remain available for the app to run.

## Limitations

- Predictions are educational/demo outputs and should not be treated as medical, nutrition, or fitness advice.
- The app uses a pre-trained local model and local dataset file.
- The `/predict` route calculates metrics from the available dataset for demonstration purposes.
- The project is not documented as a production deployment setup.

## Development Notes

- Keep the Flask app entry point in `app/app.py`.
- Run commands from the correct directory because model and dataset paths are relative to `app/`.
- Do not retrain or replace model files unless intentionally updating the machine learning workflow.
