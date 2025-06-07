from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Credit card fraud detection/creditcard.csv')

# Preprocessing
# Drop time as it is not relevant for prediction
X = data.drop(['Class', 'Time'], axis=1)
y = data['Class']

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        transaction_data = request.form['transaction']
        try:
            # Convert comma-separated string to a list of floats
            data_point = [float(x) for x in transaction_data.split(',')]
            
            # Ensure the data point has the correct number of features
            if len(data_point) != 29: # V1-V28 and Amount
                return render_template('index.html', prediction=f"Error: Please provide 29 features.")

            prediction = model.predict([data_point])
            result = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'
            return render_template('index.html', prediction=result)
        except ValueError:
            return render_template('index.html', prediction="Error: Invalid input. Please provide comma-separated numbers.")

if __name__ == '__main__':
    app.run(debug=True)