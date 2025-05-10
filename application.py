from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the cleaned car dataset
car = pd.read_csv("Cleaned Car.csv")

# Load the trained ML model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = {}
    fuel_type_mapping = {}

    for company in companies:
        models = sorted(car[car['company'] == company]['name'].unique())
        car_models[company] = models
        fuel_type_mapping[company] = {}
        for model in models:
            fuels = sorted(car[(car['company'] == company) & (car['name'] == model)]['fuel_type'].unique())
            fuel_type_mapping[company][model] = fuels

    years = sorted(car['year'].unique(), reverse=True)

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        year=years,
        fuel_type_mapping=fuel_type_mapping
    )

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_data)[0]
    predicted_price = round(prediction, 2)

    return render_template('result.html', price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
