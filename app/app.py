from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Flask app initialization
app = Flask(__name__)

# Load the trained pipeline model
pipeline = joblib.load('./models/svm_model/svm_model_pipeline.joblib')

# Define the expected features (excluding Participant ID)
expected_features = [
    'Age', 'Gender', 'Current Weight (lbs)', 'BMR (Calories)', 'Daily Calories Consumed',
    'Daily Caloric Surplus/Deficit', 'Weight Change (lbs)', 'Duration (weeks)',
    'Physical Activity Level', 'Sleep Quality', 'Stress Level'
]

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()

    # Extract Participant ID for logging purposes
    participant_id = data.pop('Participant ID')

    # Convert form data to DataFrame
    input_data = pd.DataFrame([data])

    # Ensure numerical fields are properly converted
    numerical_features = [
        'Age', 'Current Weight (lbs)', 'BMR (Calories)', 'Daily Calories Consumed',
        'Daily Caloric Surplus/Deficit', 'Weight Change (lbs)', 'Duration (weeks)', 'Stress Level'
    ]
    for col in numerical_features:
        input_data[col] = pd.to_numeric(input_data[col])

    # Reorder columns to match expected_features
    input_data = input_data[expected_features]

    # Make prediction using the loaded pipeline
    prediction = pipeline.predict(input_data)

    # Load a sample test dataset for evaluation purposes (replace this with real test data)
    # Here, we simulate the test set using the training data itself for demonstration
    test_data = pd.read_csv('./dataset/weight_change_dataset.csv')
    X_test = test_data.drop("Final Weight (lbs)", axis=1)
    y_test = test_data["Final Weight (lbs)"]

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    confidence = r2 * 100

    # Log the process to the console
    print(f"\nParticipant ID: {participant_id}")
    print(f"Received input: {data}")
    print(f"Prediction: {prediction[0]} lbs")
    print(f"Model Performance Metrics:")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - Mean Absolute Error (MAE): {mae:.4f}")
    print(f" - R² Score (Accuracy): {r2:.4f}")
    print(f" - Confidence: {confidence:.2f}%\n")

    # Return prediction and metrics to the user
    return jsonify({
        'Participant ID': participant_id,
        'Final Weight (lbs)': round(prediction[0], 2),
        'MSE': round(mse, 4),
        'MAE': round(mae, 4),
        'R² Score': round(r2, 4),
        'Confidence': f"{confidence:.2f}%"
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
