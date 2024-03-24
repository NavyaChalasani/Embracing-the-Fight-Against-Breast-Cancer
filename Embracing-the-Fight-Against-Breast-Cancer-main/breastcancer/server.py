from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the dataset
    df = pd.read_csv("C:/Users/chala/PycharmProjects/breastcancer/data/breast-cancer.csv")

    # Dropping unnecessary columns if any
    df.drop(columns=['id'], inplace=True)

    # Splitting features and target variable
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Implementing Logistic Regression
    log_reg_model = LogisticRegression()

    # Training the model
    log_reg_model.fit(X_train, y_train)

    # Getting user input from the form
    user_input = []
    for field in request.form.values():
        user_input.append(float(field))

    # Create a numpy array with the user input
    user_input = np.array([user_input])

    # Scale the user input using the same scaler used during preprocessing
    user_input_scaled = scaler.transform(user_input)

    # Make prediction using the trained model
    prediction = log_reg_model.predict(user_input_scaled)

    # Convert prediction to human-readable format
    result = "Malignant" if prediction == 'M' else "Benign"

    # Additional recommendations based on prediction
    recommendations = {
        'Malignant': [
            "Maintain a healthy weight, exercise regularly, and eat a healthy diet.",
            "Get regular mammogram screenings.",
            "Avoid hormone replacement therapy with estrogen and progestin after menopause."
        ],
        'Benign': [
            "Decrease caffeine intake and avoid smoking and alcohol.",
            "Get regular mammogram screenings.",
            "Maintain a weight that’s healthy for you and exercise regularly."
        ]
    }

    return render_template('result.html', result=result, recommendations=recommendations[result])

if __name__ == '__main__':
    app.run(debug=True)
