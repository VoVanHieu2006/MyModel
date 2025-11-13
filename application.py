from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


## import 
model = pickle.load(open('model/regression.pkl', 'rb'))
standar_scaler = pickle.load(open('model/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata", methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=="POST":
        medinc     = float(request.form['MedInc'])
        houseage   = float(request.form['HouseAge'])
        averooms   = float(request.form['AveRooms'])
        avebedrms  = float(request.form['AveBedrms'])
        population = float(request.form['Population'])
        aveoccup   = float(request.form['AveOccup'])
        latitude   = float(request.form['Latitude'])
        longitude  = float(request.form['Longitude'])

        newdata = standar_scaler.transform([[medinc, houseage, averooms, avebedrms, population, aveoccup, latitude, longitude]])  
        result = model.predict(newdata)
        return render_template('home.html', results=result)

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)