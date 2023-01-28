import numpy as np
from torch.utils.data import DataLoader
from src.afib_model.resnet1d import Resnet34
from src.afib_model.dataset import Dataset_ori
# import onnxruntime as rt

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
    uploaded_file = request.files['file_from_react']
    data = np.loadtxt(uploaded_file, delimiter=',', dtype='float32')
    # check if the data's format is correct
    # 1. less than 10 rows
    if data.shape[0] > 10:
        return {'error': 'Please include only 10 rows of data!'}

    # onnx_session = rt.InferenceSession('PPG_model.onnx')

    # classification_list = []
    # prob_list = []

    # for d in data:
    #     input_ = np.array([[d.tolist()]]).astype(np.float32)
    #     PPG_out = onnx_session.run(['output'], {'input': input_})
    #     PPG_predicted = PPG_out.argmax(1)
    #     PPG_predicted_prob = PPG_out[:, 1]
    #     classification_list.append(PPG_predicted.detach().cpu().numpy().tolist()[0])
    #     prob_list.append(PPG_predicted_prob.detach().cpu().numpy().tolist()[0])

    MODEL_PATH = 'src/afib_model/saved_models/epoch_30_ppglr_0.0001_lambda_0.9/PPG_best_1.pt'
    PPG_model = Resnet34().cpu()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    PPG_model.load_state_dict(state_dict)
    PPG_model.eval()
    dataset = Dataset_ori(data)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    classification_list = []
    prob_list = []
    for batch_idx, (PPG) in enumerate(dataLoader):
        PPG = PPG.to('cpu').float()
        PPG_out = PPG_model(PPG)
        PPG_predicted = PPG_out.argmax(1)
        PPG_predicted_prob = PPG_out[:, 1]
        classification_list.append(PPG_predicted.detach().cpu().numpy().tolist()[0])
        prob_list.append(PPG_predicted_prob.detach().cpu().numpy().tolist()[0])
    #
    return {'data': data.tolist(), 'pred': classification_list, 'pred_prob': prob_list, 'error': 0}


if __name__ == '__main__':
    # Start the Flask app
    application.run()
