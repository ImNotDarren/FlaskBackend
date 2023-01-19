import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pickle

def set_X(df):
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    X = df.drop(columns=['price'], axis=1)
    X_scaled = scaler.transform(X)
    poly = PolynomialFeatures(degree=2)
    X_scaled_poly = poly.fit_transform(X_scaled)
    return X_scaled_poly