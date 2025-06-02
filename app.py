from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Load trained ML model
model_path = "best_rf_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define mappings for categorical values
dict_dest = {'Kolkata': 0, 'Hyderabad': 1, 'Delhi': 2, 'Banglore': 3, 'Cochin': 4}
dict_airlines = {'Trujet': 0, 'SpiceJet': 1, 'Air Asia': 2, 'IndiGo': 3, 'GoAir': 4, 
                 'Vistara': 5, 'Vistara Premium economy': 6, 'Air India': 7, 
                 'Multiple carriers': 8, 'Multiple carriers Premium economy': 9, 
                 'Jet Airways': 10, 'Jet Airways Business': 11}
stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}

# Homepage with form
@app.route("/")
def home():
    return render_template("index.html")


# Prediction logic
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract user inputs from form
        airline = request.form["airline"]
        destination = request.form["destination"]
        source = request.form["source"]
        total_stops = request.form["total_stops"]

        # Convert datetime fields from input
        dep_time = datetime.strptime(request.form["dep_time"], "%Y-%m-%dT%H:%M")
        arr_time = datetime.strptime(request.form["arr_time"], "%Y-%m-%dT%H:%M")

        # ✅ Validate: If times are same, prompt the user to enter valid input
        if dep_time == arr_time:
            return render_template("error.html", error="Departure and arrival times cannot be the same. Please enter valid times.")

        # ✅ Extract necessary features
        journey_day = dep_time.day  
        journey_month = dep_time.month  
        dep_hour = dep_time.hour  
        dep_minute = dep_time.minute  
        arr_hour = arr_time.hour  
        arr_minute = arr_time.minute  

        # ✅ Calculate Flight Duration
        duration = arr_time - dep_time
        duration_hours = duration.total_seconds() // 3600  
        duration_mins = (duration.total_seconds() // 60) % 60  

        # Handling edge cases
        if duration_hours == 0: 
            duration_hours = 1
        if duration_mins == 0:  
            duration_mins = 1

        # Convert categorical inputs to numerical representation
        airline_value = dict_airlines.get(airline, -1)  
        destination_value = dict_dest.get(destination, -1) 
        total_stops_value = stop_mapping.get(total_stops, -1)

        source_mapping = {"Banglore": [1, 0, 0, 0, 0], "Delhi": [0, 1, 0, 0, 0], 
                          "Kolkata": [0, 0, 1, 0, 0], "Chennai": [0, 0, 0, 1, 0], "Mumbai": [0, 0, 0, 0, 1]}
        source_values = source_mapping.get(source, [0, 0, 0, 0, 0])

        # ✅ Pass computed duration values to ML model
        input_features = np.array([airline_value, destination_value, total_stops_value, journey_day, journey_month,
                                   dep_hour, dep_minute, arr_hour, arr_minute, duration_hours, duration_mins] + source_values).reshape(1, -1)

        # Make prediction
        predicted_price = model.predict(input_features)[0]

        # Render result page
        return render_template("result.html", price=round(predicted_price, 2))

    except Exception as e:
        return render_template("error.html", error=str(e))



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



