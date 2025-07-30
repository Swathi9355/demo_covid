import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

data = joblib.load("C:\\Users\\LENOVO\\OneDrive\\Desktop\\demo_covid\\ml_source\\covid_diag.pkl")

class inp_data(BaseModel):
    Age: int
    Gender: int
    Fever: int
    Cough: int
    Fatigue: int
    Breathlessness: int
    Comorbidity: int
    Stage: int
    Type: int
    Tumor_Size: float

app = FastAPI()

@app.get("/")
def root_msg():
    return {"Message": "Welcome to karikalan magic show"}

@app.post("/predict")
def prediction(Data: inp_data):
    inp = np.array([[Data.Age, Data.Gender, Data.Fever, Data.Cough, Data.Fatigue,
    Data.Breathlessness, Data.Comorbidity, Data.Stage, Data.Type, Data.Tumor_Size]])
    prdd = data.predict(inp)[0]
    return {"prediction": int(prdd)}