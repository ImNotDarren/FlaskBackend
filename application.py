import numpy as np
from torch.utils.data import DataLoader
from src.afib_model.resnet1d import Resnet34

import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
from src.get_df import get_df
from src.set_x import set_X
import torch

application = Flask(__name__)
CORS(application)


def predict(url, year):
    # load the model from pickle file
    with open('models/ridge_model.pkl', 'rb') as f:
        lr = pickle.load(f)

    df = get_df(url, year)

    X = set_X(df)
    y = lr.predict(X)
    return y[0]


@application.route('/')
@cross_origin()
def hello():
    return "<h1>It's alive!!!🧟‍♂️</h1>"


@application.route('/cpp/<url>/<year>')
@cross_origin()
def main(url, year):
    link = 'https://www.cars.com/vehicledetail/' + url + '/'
    predictedPrice = predict(link, year)
    return {'predictedPrice': predictedPrice}


@application.route('/afib', methods=['GET', 'POST'])
@cross_origin()
def afib():
    MODEL_PATH = 'src/afib_model/saved_models/epoch_30_ppglr_0.0001_lambda_0.9/PPG_best_1.pt'
    # PPG_model = Resnet34().cpu()
    # state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # PPG_model.load_state_dict(state_dict)
    # PPG_model.eval()
    # convert_torch2onnx(PPG_model, 'PPG_model.onnx', (1, 2400))

    uploaded_file = request.files['file_from_react']
    data = np.loadtxt(uploaded_file, delimiter=',', dtype='float32')
    # check if the data's format is correct
    # 1. less than 10 rows
    if data.shape[0] > 10:
        return {'error': 'Please include only 10 rows of data!'}

    # dataset = torch.from_numpy(data[:1])
    # PPG_feature, PPG_out = PPG_model(dataset)
    # PPG_predicted = PPG_out.argmax(1)
    # PPG_predicted_prob = PPG_out[:, 1]

    # return {'data': data.tolist(), 'pred': PPG_predicted_prob.tolist(), 'error': 0}
    return {'data': data.tolist(), 'pred': 'TODO', 'error': 0}


if __name__ == '__main__':
    # Start the Flask app
    application.run()  # Local
