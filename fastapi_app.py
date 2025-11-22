from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import mlflow.sklearn
from sklearn.datasets import load_iris
import uvicorn
import mlflow
import os


# Define FastAPI app
app = FastAPI()

# Load dataset for labels
iris = load_iris()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

model_path = r"mlruns\759130372296487625\models\m-a2d06198053f4649ba7e0e2963aa5d6f\artifacts"
model = mlflow.sklearn.load_model(model_path)

@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    prediction = model.predict(data)
    return {"prediction": iris.target_names[prediction[0]]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080,reload=True,workers=4)