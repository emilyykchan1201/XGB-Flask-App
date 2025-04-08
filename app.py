from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd
import os
import tempfile
import tensorflow as tf
import nibabel as nib
from xgboost import XGBRegressor
from cnn_run import preprocess_data, run_cnn_prediction, save_predictions_to_csv, save_predictions_to_nifti  # Import your functions

app = Flask(__name__)

# Paths to models
XGB_MODEL_FOLDER = "./model_XGB"
CNN_MODEL_PATH = "./model_CNN/cnn_trained_model.h5"
TEMPLATE_CSV = "./model_CNN/list_of_brain_voxels_MNI_size_3.csv"
NIFTI_TEMPLATE = "./model_CNN/MNI_size_3.nii.gz"

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

# Route to process input data and generate predictions using XGBoost models
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

            # Prepare to store predictions for each region
            predictions = {}

            # Load and predict with each XGBoost model in the folder
            for model_file in os.listdir(XGB_MODEL_FOLDER):
                if model_file.endswith(".json"):
                    model_path = os.path.join(XGB_MODEL_FOLDER, model_file)
                    model = XGBRegressor()
                    model.load_model(model_path)

                    # Extract region name from the model file name (e.g., XGB_region_model.json)
                    region = model_file.split('_')[1]

                    # Run prediction
                    prediction = model.predict(features)

                    # Store the prediction in the dictionary
                    predictions[region] = prediction[0]

            # Delete temporary file after processing
            os.remove(file_path)

            # Return predictions in the results page
            return render_template('xgb_results.html', predictions=predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route for CNN predictions (to process CSV data)
@app.route('/cnn_predict', methods=['POST'])
def cnn_predict():
    try:
        uploaded_file = request.files['csv_file']
        if not uploaded_file:
            return jsonify({'error': 'Please upload a CSV file'}), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.filename)
            uploaded_file.save(file_path)

            # Preprocess data
            X_data, file_names = preprocess_data(temp_dir)

            # Run CNN prediction
            predictions = run_cnn_prediction(X_data, CNN_MODEL_PATH)

            # Save output to CSV
            output_csv = os.path.join(temp_dir, "cnn_predictions.csv")
            save_predictions_to_csv(predictions, file_names, temp_dir, TEMPLATE_CSV)

            # Generate file URL for download
            file_url = f"/download/{os.path.basename(output_csv)}"

            return render_template('cnn_results.html', file_url=file_url)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route for downloading the CNN predictions file
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(tempfile.gettempdir(), filename), as_attachment=True)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)


# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Process Excel file for XGB models
@app.route('/xgb_predict', methods=['POST'])
def xgb_predict():
    try:
        uploaded_file = request.files['xlsx_file']
        if not uploaded_file:
            return jsonify({'error': 'Please upload an Excel file (.xlsx)'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
            file_path = temp_file.name
            uploaded_file.save(file_path)

            # Extract features
            selected_columns = ['RotVelRes', 'RotAccRes']
            df = pd.read_excel(file_path, engine='openpyxl')
            features = []

            for column in selected_columns:
                data = df[column]
                max_min_feature = np.max(data) - np.min(data)
                sqrt_abs_max_feature = np.sqrt(np.abs(np.max(data)))
                features.extend([max_min_feature, sqrt_abs_max_feature])

            features = np.array(features).reshape((1, -1))
            os.remove(file_path)

            # Load and predict with each model in the folder
            predictions = {}
            for model_file in os.listdir(XGB_MODEL_FOLDER):
                if model_file.startswith("XGB_") and model_file.endswith(".json"):
                    region = model_file.split("_")[1]  # Extract region name
                    model = XGBRegressor()
                    model.load_model(os.path.join(XGB_MODEL_FOLDER, model_file))
                    predictions[region] = float(model.predict(features)[0])

            return render_template('xgb_results.html', predictions=predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Process CSV files for CNN prediction
@app.route('/cnn_predict', methods=['POST'])
def cnn_predict():
    try:
        uploaded_file = request.files['csv_file']
        if not uploaded_file:
            return jsonify({'error': 'Please upload a CSV file'}), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.filename)
            uploaded_file.save(file_path)

            # Preprocess data
            X_data, file_names = preprocess_data(temp_dir)

            # Run CNN prediction
            predictions = run_cnn_prediction(X_data, CNN_MODEL_PATH)

            # Save output to CSV
            output_csv = os.path.join(temp_dir, "cnn_predictions.csv")
            save_predictions_to_csv(predictions, file_names, temp_dir, TEMPLATE_CSV)

            # Return the CSV file
            return send_file(output_csv, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
