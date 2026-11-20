from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="PiuraTasador AI")
ia = joblib.load("piuratasador_ai.pkl")

class Inmueble(BaseModel):
    distrito: str
    zona: str
    tipo: str
    habitaciones: float
    banos: float
    area_m2: float
    cochera: float
    amoblado: float

@app.get("/")
def inicio():
    return {"estado": "PiuraTasador AI activo"}

@app.post("/")
@app.post("/predecir")
def predecir(datos: Inmueble):
    df = pd.DataFrame([datos.model_dump()])
    prediccion = ia.predict(df)
    return {"precio": round(float(prediccion[0]), 2)}
