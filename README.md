# 🚀 SpaceTel Anomaly Detection Project

This repository contains all necessary files to train models and create a webpage for anomaly detection in satellite telemetry data.

## 📂 Directory Structure

```
📂 data
    ├── raw_data.csv         # Raw telemetry data
    ├── train_data.csv       # Training dataset
    └── test_data.csv        # Testing dataset

📂 models
    ├── isolation_forest_model.pkl  # Isolation Forest model for anomaly detection
    └── lstm_model.h5               # LSTM model for anomaly detection

📂 notebook
    └── models_spacetel.ipynb       # Jupyter notebook for model training

📂 web
    └── (placeholder for webpage files)

README.md
```

## ⚙️ How to Use

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook:**
    - Open `notebook/models_spacetel.ipynb` to view, edit, or retrain the models.

4. **Model Files:**
    - `models/isolation_forest_model.pkl` is a scikit-learn Isolation Forest model.
    - `models/lstm_model.h5` is a TensorFlow LSTM model.

5. **Data Files:**
    - `data/` contains raw data, as well as train and test datasets.

## 🌐 Web Integration
- Your teammates can use the models directly from the `models/` directory.
- Make sure to use libraries like `joblib` for `.pkl` files and `tensorflow.keras.models.load_model()` for `.h5` files.

## 🛠️ Helpful Commands

- Load Isolation Forest model:
    ```python
    import joblib
    iso_model = joblib.load('models/isolation_forest_model.pkl')
    ```

- Load LSTM model:
    ```python
    from tensorflow.keras.models import load_model
    lstm_model = load_model('models/lstm_model.h5')
    ```

## 🔑 Notes
- Ensure that the `models/` and `data/` folders remain intact for the webpage to function correctly.
- Update the `web/` directory with the necessary frontend files as needed.

