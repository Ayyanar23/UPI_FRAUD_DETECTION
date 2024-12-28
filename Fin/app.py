from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from io import StringIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using sessions

# Home route
@app.route('/')
def home():
    return render_template('login.html')

# Login route
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == "admin" and password == "admin":
        return redirect(url_for('upload'))
    return "Invalid credentials"

# Dataset Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            # Save the file to the current working directory
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            
            # Read the file into a DataFrame
            df = pd.read_csv(file_path) if file.filename.endswith('.csv') else pd.read_excel(file_path)

            # Store the dataset path in session for later use
            session['data_path'] = file_path

            # Train the model and save it in the session
            model, X_test, y_test = train_model(df)
            joblib.dump(model, 'model.pkl')  # Save the model to file
            
            return redirect(url_for('fraud_detection'))
    return render_template('upload.html')

# Train model
def train_model(df):
    # Add fraud column based on 'Status' (1 for 'FAILED', 0 for 'SUCCESS')
    df['fraud'] = df['Status'].apply(lambda x: 1 if x == 'FAILED' else 0)
    
    # Drop 'Status' column as it's now replaced by 'fraud' column
    df = df.drop(['Status'], axis=1)

    # Drop any non-numeric columns (e.g., identifiers, UPI IDs) or columns that should not be used
    df = df.drop(['Sender UPI ID', 'Receiver UPI ID', 'Sender Name', 'Receiver Name'], axis=1, errors='ignore')

    # Convert categorical features to numeric (e.g., 'Sender Name', 'Receiver Name')
    df = pd.get_dummies(df, drop_first=True)

    # Features (X) and target (y)
    X = df.drop('fraud', axis=1)
    y = df['fraud']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test

# Fraud Detection Route
@app.route('/detect', methods=['GET', 'POST'])
def fraud_detection():
    # Check if 'data_path' exists in session
    file_path = session.get('data_path')
    if not file_path:
        return redirect(url_for('upload'))  # Redirect to upload if no data in session

    # Load the model using joblib.load
    model = joblib.load('model.pkl')
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Ensure the 'fraud' column exists by replicating the train_model logic
    df['fraud'] = df['Status'].apply(lambda x: 1 if x == 'FAILED' else 0)
    df = df.drop(['Status'], axis=1)
    df = df.drop(['Sender UPI ID', 'Receiver UPI ID', 'Sender Name', 'Receiver Name'], axis=1, errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    
    if request.method == 'POST':
        # Extract features and predict fraud
        X = df.drop('fraud', axis=1)
        predictions = model.predict(X)
        df['predictions'] = predictions
        
        # Render the fraud detection results
        return render_template('fraud_results.html', data=df)
    
    return render_template('fraud_detection.html')

# Analysis Route
@app.route('/analysis')
def analysis():
    # Check if 'data_path' exists in session
    file_path = session.get('data_path')
    if not file_path:
        return redirect(url_for('upload'))  # Redirect to upload if no data in session
    
    # Load the data from the file path
    df = pd.read_csv(file_path)
    
    # Ensure the 'fraud' column exists by replicating the train_model logic
    df['fraud'] = df['Status'].apply(lambda x: 1 if x == 'FAILED' else 0)
    df = df.drop(['Status'], axis=1)
    df = df.drop(['Sender UPI ID', 'Receiver UPI ID', 'Sender Name', 'Receiver Name'], axis=1, errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    
    # Plotting the analysis
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.hist(df['Amount (INR)'], bins=30, color='blue', alpha=0.7)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.savefig('static/plot.png')
    
    # Calculate the number of fraud transactions
    num_fraud_transactions = df[df['fraud'] == 1].shape[0]
    total_transactions = df.shape[0]
    
    return render_template('analysis.html', plot_url='static/plot.png', 
                           num_fraud_transactions=num_fraud_transactions, 
                           total_transactions=total_transactions)

if __name__ == '__main__':
    app.run(debug=True)
