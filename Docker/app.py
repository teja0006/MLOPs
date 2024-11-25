import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.datasets import load_diabetes

def train_model():
    data = load_diabetes()
    X,y = data.data,data.target

    model = RandomForestClassifier()
    model.fit(X,y)

    with open('model.pkl','wb') as file:
        pickle.dump(model,file)

    print("Model trained and saved as model.pkl")

def predict():
    with open('model.pkl','rb') as file:
        model = pickle.load(file)

    test = np.array([0.0381, 0.0507, 0.0617, 0.0219, -0.0442, -0.0348, -0.0434, -0.0026, 0.0199, -0.0176]).reshape(1,-1)
    prediction = model.predict(test)

    print(f"Prediction for {test}: {int(prediction[0])}")

if __name__ == "__main__":
    train_model()
    predict()