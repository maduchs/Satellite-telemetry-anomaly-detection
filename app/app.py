# app.py
import os
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from preprocessor import DataPreprocessor
from isolation_forest_model import IsolationForestModel
from lstm_model import LSTMModel
from sklearn.model_selection import train_test_split
from visualization import plot_model_comparisons
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs('static', exist_ok=True)
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['csv_file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    filepath = None
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and validate data
        df = pd.read_csv(filepath)
        
        # Initialize preprocessor and process data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(df)
        
        # Calculate contamination (capped at 0.5)
        anomaly_ratio = float((processed_data['y_train'] == 1).mean())
        contamination = min(0.5, anomaly_ratio)
        logging.info(f"Calculated contamination: {contamination}")
        
        # Initialize and run Isolation Forest
        iforest = IsolationForestModel(contamination=contamination)
        iforest.train(processed_data['X_train'], processed_data['y_train'])
        iforest_results, iforest_metrics = iforest.predict(
            processed_data['X_test'], 
            processed_data['y_test']
        )
        
        # Get Isolation Forest raw predictions and confusion matrix
        iforest_raw_predictions = iforest.model.decision_function(processed_data['X_test'])
        iforest_predictions = (iforest_raw_predictions < 0).astype(int)
        iforest_conf_matrix = confusion_matrix(processed_data['y_test'], iforest_predictions)
        
        # Initialize and run LSTM
        lstm = LSTMModel(input_shape=(1, processed_data['X_train'].shape[1]))
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            processed_data['X_train_lstm'],
            processed_data['y_train'],
            test_size=0.2,
            random_state=42
        )
        lstm.train(
            X_train,
            y_train,
            X_val,
            y_val
        )
        lstm_results, lstm_metrics, roc_auc, lstm_conf_matrix = lstm.predict(
            processed_data['X_test_lstm'],
            processed_data['y_test']
        )
        
        # Get LSTM raw predictions
        lstm_raw_predictions = lstm.model.predict(processed_data['X_test_lstm']).flatten()
        
        # Generate visualizations
        plot_model_comparisons(
            iforest_metrics,
            lstm_metrics,
            iforest_results,
            lstm_results,
            lstm_conf_matrix,
            iforest_conf_matrix,
            processed_data['y_test'],
            lstm_raw_predictions,
            iforest_raw_predictions
        )
        
        # Prepare metrics for template
        metrics = {
            'isolation_forest': {
                'accuracy': round(iforest_metrics['accuracy'] * 100, 2),
                'precision': round(iforest_metrics['1']['precision'] * 100, 2),
                'recall': round(iforest_metrics['1']['recall'] * 100, 2),
                'f1': round(iforest_metrics['1']['f1-score'] * 100, 2),
                'anomalies': int(iforest_results['isolation_forest_anomaly'].value_counts().get('Anomaly', 0))
            },
            'lstm': {
                'accuracy': round(lstm_metrics['accuracy'] * 100, 2),
                'precision': round(lstm_metrics['1']['precision'] * 100, 2),
                'recall': round(lstm_metrics['1']['recall'] * 100, 2),
                'f1': round(lstm_metrics['1']['f1-score'] * 100, 2),
                'roc_auc': round(roc_auc * 100, 2),
                'anomalies': int(lstm_results['lstm_anomaly'].value_counts().get('Anomaly', 0))
            }
        }
        
        # Clean up
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        
        return render_template('results.html', metrics=metrics)
        
    except Exception as e:
        logging.error(f"Processing error: {e}")
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return f"Error processing file: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)