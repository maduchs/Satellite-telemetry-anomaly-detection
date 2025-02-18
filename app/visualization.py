# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_model_comparisons(iforest_metrics, lstm_metrics, iforest_results, lstm_results, lstm_conf_matrix, iforest_conf_matrix, y_test, lstm_predictions, iforest_predictions):
    """
    Create and save comparison visualizations for LSTM and Isolation Forest models
    """
    # Set style
    sns.set_style('darkgrid')
    
    # 1. Performance Metrics Comparison
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.35

    iforest_scores = [
        iforest_metrics['accuracy'],
        iforest_metrics['1']['precision'],
        iforest_metrics['1']['recall'],
        iforest_metrics['1']['f1-score']
    ]
    
    lstm_scores = [
        lstm_metrics['accuracy'],
        lstm_metrics['1']['precision'],
        lstm_metrics['1']['recall'],
        lstm_metrics['1']['f1-score']
    ]

    plt.bar(x - width/2, iforest_scores, width, label='Isolation Forest')
    plt.bar(x + width/2, lstm_scores, width, label='LSTM')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.savefig('static/performance_comparison.png')
    plt.close()

    # 2. Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(iforest_conf_matrix, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Isolation Forest Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(lstm_conf_matrix, annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title('LSTM Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('static/confusion_matrices.png')
    plt.close()

    # 3. ROC Curves
    plt.figure(figsize=(8, 6))
    
    # LSTM ROC
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_predictions)
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
    
    # Isolation Forest ROC
    fpr_if, tpr_if, _ = roc_curve(y_test, iforest_predictions)
    roc_auc_if = auc(fpr_if, tpr_if)
    
    plt.plot(fpr_lstm, tpr_lstm, color='darkorange', lw=2,
             label=f'LSTM (AUC = {roc_auc_lstm:.2f})')
    plt.plot(fpr_if, tpr_if, color='blue', lw=2,
             label=f'Isolation Forest (AUC = {roc_auc_if:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.savefig('static/roc_curves.png')
    plt.close()

    # 4. Anomaly Distribution
    plt.figure(figsize=(10, 6))
    
    lstm_anomalies = lstm_results['lstm_anomaly'].value_counts()
    iforest_anomalies = iforest_results['isolation_forest_anomaly'].value_counts()
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [lstm_anomalies.get('Normal', 0), lstm_anomalies.get('Anomaly', 0)], 
            width, label='LSTM')
    plt.bar(x + width/2, [iforest_anomalies.get('Normal', 0), iforest_anomalies.get('Anomaly', 0)], 
            width, label='Isolation Forest')
    
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Anomaly Distribution by Model')
    plt.xticks(x, ['Normal', 'Anomaly'])
    plt.legend()
    plt.savefig('static/anomaly_distribution.png')
    plt.close()