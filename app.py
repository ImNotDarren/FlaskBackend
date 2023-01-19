import pickle

from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin

from src.get_df import get_df
from src.set_x import set_X

app = Flask('cpp')
CORS(app)

def predict(url, year):
    # load the model from pickle file
    with open('models/ridge_model.pkl', 'rb') as f:
        lr = pickle.load(f)

    df = get_df(url, year)

    X = set_X(df)
    y = lr.predict(X)
    return y, df.iloc[0]['price']


@app.route('/')
def hello():
    return "<h1>It's alive!!!🧟‍♂️</h1>"


@app.route('/cpp/<url>/<year>')
@cross_origin()
def main(url, year):
    link = 'https://www.cars.com/vehicledetail/' + url + '/'
    predictedPrice, price = predict(link, year)
    return {'price': price, 'predictedPrice': predictedPrice[0], 'year': year}


if __name__ == '__main__':
    # Start the Flask app
    app.run()  # Local
