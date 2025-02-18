import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def validate_data(self, df):
        """Validate input data format and required columns."""
        required_columns = [
            'Timestamp', 'Telemetry_Value', 'Status', 
            'Error_Code', 'CPU_Usage (%)', 'Memory_Usage (MB)', 
            'Packet_Loss (%)', 'Anomaly_Label'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
    def preprocess(self, df):
        """Preprocess the telemetry data for anomaly detection."""
        try:
            logging.info("Starting data preprocessing")
            
            # Validate input data
            self.validate_data(df)
            
            # Convert Timestamp and create time features
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Hour'] = df['Timestamp'].dt.hour
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
            df['Month'] = df['Timestamp'].dt.month
            df['Weekday'] = df['Timestamp'].dt.weekday < 5  # True if weekday, False if weekend
            
            # Clean and transform numerical features
            df['Telemetry_Value'] = pd.to_numeric(df['Telemetry_Value'], errors='coerce')
            
            # Create binary features
            df['is_critical'] = (df['Status'] == 'Critical').astype(int)
            df['is_warning'] = (df['Status'] == 'Warning').astype(int)
            df['has_error'] = df['Error_Code'].notna().astype(int)
            df['has_packet_loss'] = (df['Packet_Loss (%)'] > 0).astype(int)
            df['high_cpu_usage'] = (df['CPU_Usage (%)'] > 80).astype(int)
            df['high_memory_usage'] = (df['Memory_Usage (MB)'] > 2000).astype(int)
            
            # Select features for modeling
            features = [
                'Hour', 'DayOfWeek', 'Month', 'Weekday', 'Telemetry_Value', 
                'is_critical', 'is_warning', 'has_error', 'has_packet_loss', 
                'high_cpu_usage', 'high_memory_usage', 'CPU_Usage (%)', 
                'Memory_Usage (MB)', 'Packet_Loss (%)'
            ]
            
            X = df[features]
            y = df['Anomaly_Label']
            
            # Handle missing values
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), 
                columns=X.columns
            )
            
            # Handle class imbalance only if needed
            if len(y[y==1]) / len(y) < 0.2:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
                X_scaled = pd.DataFrame(X_resampled, columns=X_scaled.columns)
                y = pd.Series(y_resampled)
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Prepare LSTM data (reshaping for LSTM input)
            X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            logging.info("Data preprocessing completed successfully")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_lstm': X_train_lstm,
                'X_test_lstm': X_test_lstm
            }
            
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise
