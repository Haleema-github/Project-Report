# Importing libraries
from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import pickle

import warnings
warnings.filterwarnings('ignore')

# Creating an instance of Flask
app=Flask(__name__)

# Loading the model
model=pickle.load(open('randomcv_model.pkl','rb'))

# Creating homepage
@app.route('/')
def home():
    return render_template('index.html')

# Making Predictions
@app.route('/predict',methods=['POST'])
# @app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        Location = str(request.form['Location'])

        WindGustDir = str(request.form['WindGustDir'])

        WindDir9am = str(request.form['WindDir9am'])

        WindDir3pm = str(request.form['WindDir3pm'])

        MaxTemp = float(request.form['MaxTemp'])

        WindGustSpeed = float(request.form['WindGustSpeed'])

        Humidity3pm = float(request.form['Humidity3pm'])

        Pressure3pm = float(request.form['Pressure3pm'])

        RainToday = str(request.form['RainToday'])

        # Storing the data in 2-D array
        predict_list = [[Location,MaxTemp,WindGustDir,
                         WindGustSpeed,WindDir9am,WindDir3pm, 
                          Humidity3pm,Pressure3pm,RainToday,]]
                            

# Predicting the results using the model loaded from a pickle file(randomcv_model.pkl)
        output = model.predict(predict_list)

# oading the templates for respective outputs(0 or 1)
        if output == 1:
            return render_template('rainyday.html')
        else:
            return render_template('sunnyday.html')

    return render_template('index.html')


# Main driver function
if __name__ == '__main__':
    app.run(debug=True)