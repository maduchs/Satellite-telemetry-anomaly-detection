# isolation_forest_model.py
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IsolationForestModel:
    def __init__(self, contamination=0.1, n_estimators=250, random_state=42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples=256
        )
        logging.info(f"Initialized IsolationForest with contamination: {self.contamination}")
    
    def train(self, X_train, y_train):
        """Train the model on normal data only."""
        try:
            # Calculate actual anomaly ratio (capped at 0.5)
            anomaly_ratio = min((y_train == 1).mean(), 0.5)
            logging.info(f"Actual anomaly ratio: {anomaly_ratio:.4f}")
            
            # Update contamination if needed
            if anomaly_ratio != self.contamination:
                self.contamination = anomaly_ratio
                self.model = IsolationForest(
                    contamination=self.contamination,
                    n_estimators=self.model.n_estimators,
                    random_state=self.model.random_state,
                    max_samples='auto'
                )
                logging.info(f"Updated contamination to: {self.contamination}")
            
            # Use only normal samples for training
            normal_idx = y_train == 0
            self.model.fit(X_train[normal_idx])
            logging.info("Isolation Forest model trained successfully")
            
        except Exception as e:
            logging.error(f"Error training Isolation Forest: {e}")
            raise
            
    def predict(self, X_test, y_test):
        """Make predictions and return results with metrics."""
        try:
            # Get raw predictions
            raw_pred = self.model.predict(X_test)
            
            # Convert to binary labels (1 for anomaly, 0 for normal)
            y_pred = pd.Series(np.where(raw_pred == -1, 1, 0), index=y_test.index)
            
            # Convert to categorical labels for display
            y_pred_display = pd.Series(
                np.where(y_pred == 1, 'Anomaly', 'Normal'),
                index=y_test.index
            )
            
            # Prepare results DataFrame
            results = pd.DataFrame({
                'isolation_forest_anomaly': y_pred_display,
                'actual': np.where(y_test == 1, 'Anomaly', 'Normal')
            })
            
            # Calculate metrics
            metrics = classification_report(y_test, y_pred, output_dict=True)
            
            logging.info(f"Predictions complete. Detected {y_pred.sum()} anomalies")
            return results, metrics
            
        except Exception as e:
            logging.error(f"Error making Isolation Forest predictions: {e}")
            raise
