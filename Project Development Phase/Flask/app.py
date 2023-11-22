from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import pickle
from datetime import datetime, timedelta

# Load the trained model
model = pickle.load(open('Forecaster.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Perform prediction based on the current date
        next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        future = pd.DataFrame({'ds': [next_day]})
        forecast = model.predict(future)
        predicted_price = forecast['yhat'].item()

        return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)