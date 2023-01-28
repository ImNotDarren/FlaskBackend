import torch
from src.afib_model.resnet1d import Resnet34
from torch.utils.data import DataLoader
from src.afib_model.dataset import Dataset_ori
import numpy as np


if __name__ == '__main__':
    MODEL_PATH = 'src/afib_model/saved_models/epoch_30_ppglr_0.0001_lambda_0.9/PPG_best_1.pt'
    PPG_model = Resnet34().cpu()

    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    PPG_model.load_state_dict(state_dict)

    PPG_model.eval()
    dummy_input = torch.randn(1, 1, 2400).cpu()
    data = np.loadtxt('./test2.csv', delimiter=',', dtype='float32')
    dataset = Dataset_ori(data)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for (PPG) in dataLoader:
        print(PPG)
        torch.onnx.export(PPG_model,
                          PPG,
                          'PPG_model.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          export_params=True)
        break
