from flask import Flask, render_template, request,redirect
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)
cors=CORS(app)

model = pickle.load(open("/Users/lavishvaishnav/Desktop/PYTHON/CarPricePred/LinearRegressionModel.pkl",'rb'))
car=pd.read_csv("/Users/lavishvaishnav/Desktop/PYTHON/CarPricePred/cleaned car.csv")

@app.route('/')
def hello():
    companies = sorted(car['company'].unique())
    car_models= sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type= car['fuel_type'].unique()
    return render_template('index.html',companies= companies,car_models= car_models,years= year,fuel_types= fuel_type)



@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    print("request----",request)
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    print(fuel_type)
    driven=request.form.get('kilo_driven')
    print(company,car_model,year,fuel_type,driven)

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))




 

    


if __name__ == '__main__':
    app.run(debug=True)
