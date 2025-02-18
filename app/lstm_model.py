# lstm_model.py
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel:
    def __init__(self, input_shape, epochs=50, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        """Build LSTM model architecture with improved layers and tuning."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=False),  # Adjusted to non-return_sequences for final output
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification: Anomaly (1) or Normal (0)
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        return model

    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model with early stopping and model checkpoint."""
        try:
            # Calculate class weights to handle class imbalance
            class_weights = {
                0: 1,
                1: (len(y_train) - sum(y_train)) / sum(y_train)  # Balance the class weights
            }
            
            # Callbacks for early stopping and saving the best model
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
            model_checkpoint = ModelCheckpoint(
                'best_lstm_model.h5', 
                save_best_only=True, 
                monitor='val_loss', 
                mode='min'
            )
            
            # Train the model
            self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            logging.info("LSTM model trained successfully")
        except Exception as e:
            logging.error(f"Error training LSTM model: {e}")
            raise

    def predict(self, X_test, y_test):
        """Make predictions and evaluate with additional metrics."""
        try:
            # Get predictions
            y_pred_proba = self.model.predict(X_test).flatten()  # Probability predictions
            y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary labels

            # Convert to categorical for display
            y_pred_display = pd.Series(
                np.where(y_pred == 1, 'Anomaly', 'Normal'),
                index=y_test.index
            )

            # Prepare results DataFrame
            results = pd.DataFrame({
                'lstm_anomaly': y_pred_display,
                'actual': np.where(y_test == 1, 'Anomaly', 'Normal')
            })

            # Classification metrics
            metrics = classification_report(y_test, y_pred, output_dict=True)
            roc_auc = roc_auc_score(y_test, y_pred_proba)  # ROC AUC for model evaluation
            conf_matrix = confusion_matrix(y_test, y_pred)

            logging.info(f"Model Evaluation: ROC AUC: {roc_auc:.4f}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")

            return results, metrics, roc_auc, conf_matrix

        except Exception as e:
            logging.error(f"Error making LSTM predictions: {e}")
            raise