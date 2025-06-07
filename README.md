# Credit Card Fraud Detection

This project is a web application that detects fraudulent credit card transactions using a machine learning model. It is built with Flask and scikit-learn.

## Features

-   **Fraud Detection:** Classifies transactions as fraudulent or not fraudulent.
-   **Machine Learning Model:** Uses a RandomForestClassifier trained on a credit card fraud dataset.
-   **Imbalanced Data Handling:** Uses SMOTE (Synthetic Minority Over-sampling Technique) to handle the imbalanced nature of the dataset.
-   **Web Interface:** A simple web interface to input transaction data and get predictions.

## Dataset

The project uses the "Credit Card Fraud Detection" dataset from Kaggle. You can download it from [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kuppaladineshmano/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    -   Download the `creditcard.csv` file from the Kaggle link above and place it in the `Credit card fraud detection/` directory.

5.  **Run the Flask app:**
    ```bash
    python app.py
    ```

## Usage

-   Open your web browser and go to `http://127.0.0.1:5000`.
-   Enter the 29 transaction features (V1-V28 and Amount) as a comma-separated string in the input box.
-   Click the "Predict" button to see the prediction.

## Libraries Used

-   **Flask**
-   **Pandas**
-   **Scikit-learn**
-   **Imbalanced-learn**