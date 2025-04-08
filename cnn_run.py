import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf

def preprocess_data(directory_path):
    """
    Preprocess .xlsx files in the given directory to generate CNN-ready data.
    Rotational accelerations (RotAccX, RotAccY, RotAccZ) are scaled by 0.01.
    Tracks original filenames for output naming.

    Parameters:
        directory_path (str): Path to the directory containing .xlsx files.

    Returns:
        tuple: (numpy.ndarray, list of str)
            - Processed input data (X_data) ready for CNN.
            - List of original input filenames (without extensions).
    """
    all_data = []
    file_names = []  # To track filenames for output
    # required_columns = ['RotVelX', 'RotVelY', 'RotVelZ', 'RotAccX', 'RotAccY', 'RotAccZ']
    # required_columns = ['PAV_X_radsec', 'PAV_Y_radsec', 'PAV_Z_radsec', 'PAA_X_radsec_2', 'PAA_Y_radsec_2', 'PAA_Z_radsec_2']
    required_columns = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'ang_x', 'ang_y', 'ang_z']

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory_path, filename)
            
            try:
                # Attempt to read the file, first assuming there is a header
                df = pd.read_csv(filepath, header=0)

                # Check if all required columns are present
                if set(required_columns).issubset(df.columns):
                    df = df[required_columns]  # Use only required columns
                else:
                    # Retry with no header if required columns are missing
                    df = pd.read_excel(filepath, header=None)
                    if df.shape[1] >= 6:  # Ensure there are at least 6 columns
                        df = df.iloc[:, :6]  # Use the first 6 columns
                        df.columns = required_columns
                    else:
                        print(f"Skipping file {filename}: insufficient columns.")
                        continue
                
                # Scale RotAccX, RotAccY, RotAccZ by 0.01
                # df[['RotAccX', 'RotAccY', 'RotAccZ']] *= 0.01
                df[['ang_x', 'ang_y', 'ang_z']] *= 0.01

                # Ensure 104 rows (truncate or pad with zeros)
                if df.shape[0] > 104:
                    df = df.iloc[:104, :]  # Truncate to 104 rows
                elif df.shape[0] < 104:
                    # Pad with zeros if fewer than 104 rows
                    padding = pd.DataFrame(0, index=range(104 - df.shape[0]), columns=required_columns)
                    df = pd.concat([df, padding], axis=0)

                # Add the processed data to the list
                all_data.append(df.values)
                file_names.append(os.path.splitext(filename)[0])  # Store original filename without extension

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

    # Convert all data to a NumPy array for CNN processing
    X_data = np.array(all_data)  # Shape: (num_files, 104, 6)
    X_data = np.transpose(X_data, (0, 2, 1))  # Transpose to (num_files, 6, 104)
    X_data = np.expand_dims(X_data, axis=-1)  # Add channel dimension, shape: (num_files, 6, 104, 1)

    return X_data, file_names

def run_cnn_prediction(X_data, model_path):
    """
    Run predictions using a pre-trained CNN model.

    Parameters:
        X_data (numpy.ndarray): Preprocessed input data for CNN.
        model_path (str): Path to the saved CNN model.

    Returns:
        numpy.ndarray: Predictions from the CNN model.
    """
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X_data)
    return predictions

def save_predictions_to_nifti(predictions, file_names, output_directory, template_csv, nifti_template_path):
    """
    Save CNN predictions into NIfTI files using a template CSV and reference NIfTI file.
    Each output file is named using the original input filename.

    Parameters:
        predictions (numpy.ndarray): CNN predictions.
        file_names (list of str): List of original input filenames (without extensions).
        output_directory (str): Directory to save output NIfTI files.
        template_csv (str): Path to the CSV file with voxel coordinates.
        nifti_template_path (str): Path to the template NIfTI file.
    """
    voxel_df = pd.read_csv(template_csv)

    # Load the NIfTI template once
    template_nifti = nib.load(nifti_template_path)
    template_data = template_nifti.get_fdata()

    for i, (prediction, original_name) in enumerate(zip(predictions, file_names)):
        # Create a copy of the template data to store predictions
        nifti_data = np.zeros_like(template_data)

        # Map predictions to the NIfTI data using voxel coordinates
        for index, row in voxel_df.iterrows():
            x, y, z = int(row['X']), int(row['Y']), int(row['Z'])
            nifti_data[x, y, z] = prediction[index]

        # Save the prediction as a NIfTI file, using the original input filename
        output_path = os.path.join(output_directory, f"prediction_{original_name}.nii.gz")
        output_nifti = nib.Nifti1Image(nifti_data, template_nifti.affine, template_nifti.header)
        nib.save(output_nifti, output_path)
        print(f"Saved NifTi prediction: {output_path}")

def save_predictions_to_csv(predictions, file_names, output_directory, template_csv):
    """
    Save CNN predictions into CSV files using a template CSV.

    Parameters:
        predictions (numpy.ndarray): CNN predictions.
        file_names (list of str): List of original input filenames (without extensions).
        output_directory (str): Directory to save output CSV files.
        template_csv (str): Path to the CSV file with voxel coordinates.
    """
    voxel_df = pd.read_csv(template_csv)

    for i, (prediction, original_name) in enumerate(zip(predictions, file_names)):
        # Add the prediction values as a new column in the voxel DataFrame
        voxel_df['Strain'] = prediction

        # Save the DataFrame to a CSV file named using the original input filename
        csv_output_path = os.path.join(output_directory, f"prediction_{original_name}.csv")
        voxel_df.to_csv(csv_output_path, index=False)
        print(f"Saved CSV prediction: {csv_output_path}")

if __name__ == "__main__":
    # Command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python run.py <folder_path> <output_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_directory = sys.argv[2]

    # Validate paths
    if not os.path.isdir(folder_path):
        print(f"Error: Input directory not found: {folder_path}")
        sys.exit(1)
    os.makedirs(output_directory, exist_ok=True)

    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "cnn_trained_model.h5")
    template_csv = os.path.join(current_dir, "list_of_brain_voxels_MNI_size_3.csv")
    nifti_template_path = os.path.join(current_dir, "MNI_size_3.nii.gz")

    # Step 1: Preprocess data
    print("Preprocessing data...")
    X_data, file_names = preprocess_data(folder_path)
    if X_data.size == 0:
        print("No valid data found. Exiting.")
        sys.exit(1)
    print(f"Preprocessed data shape: {X_data.shape}")

    # Step 2: Run CNN prediction
    print("Running CNN prediction...")
    predictions = run_cnn_prediction(X_data, model_path)
    print(f"Predictions shape: {predictions.shape}")

    # Step 3: Save predictions to NIfTI files
    print("Saving predictions to NIfTI files...")
    save_predictions_to_nifti(predictions, file_names, output_directory, template_csv, nifti_template_path)

    # Step 4: Save predictions to CSV files
    print("Saving predictions to CSV files...")
    save_predictions_to_csv(predictions, file_names, output_directory, template_csv)

    print("All predictions saved.")
