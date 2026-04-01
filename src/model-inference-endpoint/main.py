from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from text_loader import loader

mlflow.set_tracking_uri('data')

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

with open("data/models/run_id.txt") as f:
    run_id = f.read().strip()

MODEL_URI = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    data_loader = loader.DataLoader()
    cleaned_data = data_loader.clean_text(input_data.input_texts)
    # Pipeline handles vectorization internally
    prediction = model.predict([cleaned_data])
    return {"prediction": prediction.tolist()}
