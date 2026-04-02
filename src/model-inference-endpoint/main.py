from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from text_loader import loader

mlflow.set_tracking_uri('data')

class InputText(BaseModel):
    input_texts: str

app = FastAPI()
with open("data/models/model_path.txt") as f:
    model_path = f.read().strip()

model = mlflow.pyfunc.load_model(model_path)

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    # TODO - task 2 
    # -----------------------------------
    # Goal: our goal is to complete the implementation of this function, 
    #       which takes input data and returns a prediction result from a pre-trained model.
    data_loader = loader.DataLoader()
    cleaned_data = data_loader.clean_text(input_data.input_texts)
    prediction = model.predict([cleaned_data])
    return {"prediction": prediction.tolist()}