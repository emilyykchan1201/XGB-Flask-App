from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os
import tempfile
import glob  # For dynamically finding model files

app = Flask(__name__)

# Model directory path
MODEL_DIR = "./model/"

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


# Route to process input data and generate predictions using multiple models
@app.route('/process', methods=['POST'])
def process_data():
    try:
        # Get uploaded file
        uploaded_file = request.files['xlsx_file']

        if not uploaded_file:
            return jsonify({'error': 'Please upload an Excel file (.xlsx)'}), 400

        # Create a temporary file for uploading
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            file_path = temp_file.name
            uploaded_file.save(file_path)

            # Extract features from uploaded .xlsx file
            features = extract_features(file_path)

            # Find all XGB models in the folder
            model_files = glob.glob(os.path.join(MODEL_DIR, "XGB_*_model.json"))
            
            if not model_files:
                return jsonify({'error': 'No models found in the model directory'}), 400

            predictions = {}

            for model_file in model_files:
                # Extract region name from the file name
                region_name = model_file.split("XGB_")[1].split("_model.json")[0]

                # Load XGB model
                model = XGBRegressor()
                model.load_model(model_file)

                # Run prediction
                prediction = model.predict(features)[0]

                # Store prediction result with region name
                predictions[region_name] = prediction

            # Delete temporary file after processing
            os.remove(file_path)

            # Return predictions
            return render_template('results.html', predictions=predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
