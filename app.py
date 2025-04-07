from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os

app = Flask(__name__)

# Predefined model path (CHANGE IF NEEDED)
MODEL_PATH = "./model/XGBmodel_WB.json"

# Feature extraction function (supports .xlsx)
def extract_features(file_path):
    """Compute features for XGBoost model."""
    selected_columns = ['RotVelRes', 'RotAccRes']
    
    print(f'Extracting features from {file_path}...')
    
    # Read the Excel file
    df = pd.read_excel(file_path, engine='openpyxl')

    feature_matrix = []
    
    # Calculate features for each column
    for column in selected_columns:
        data = df[column]

        # Calculate features
        max_min_feature = np.max(data) - np.min(data)
        sqrt_abs_max_feature = np.sqrt(np.abs(np.max(data)))

        # Append features to the feature matrix
        feature_matrix.append(max_min_feature)
        feature_matrix.append(sqrt_abs_max_feature)

    # Reshape feature matrix
    feature_matrix = np.array(feature_matrix).reshape((1, -1))

    print(f'Feature extraction completed. Shape: {feature_matrix.shape}')
    return feature_matrix


# Home page (renders HTML form)
@app.route('/')
def home():
    return render_template('index.html')


# Route to process input data and generate predictions
@app.route('/process', methods=['POST'])
def process_data():
    try:
        # Get uploaded file
        uploaded_file = request.files['xlsx_file']

        if not uploaded_file:
            return jsonify({'error': 'Please upload an Excel file (.xlsx)'}), 400

        # Save uploaded file temporarily
        file_path = os.path.join("uploads", uploaded_file.filename)
        uploaded_file.save(file_path)

        # Extract features from uploaded .xlsx file
        features = extract_features(file_path)

        # Load XGB model
        model = XGBRegressor()
        model.load_model(MODEL_PATH)

        # Run prediction
        predictions = model.predict(features)

        # Delete temporary file after processing
        os.remove(file_path)

        # Return predictions
        return render_template('results.html', prediction=predictions[0])

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs("uploads", exist_ok=True)

    # Run the Flask app
    app.run(debug=True)
