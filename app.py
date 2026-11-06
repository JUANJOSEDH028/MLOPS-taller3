from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "models" / "pipefinal.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="API de predicción de entradas agotadas")


class DatosConcierto(BaseModel):
    Genre: str | None = Field(
        default="Rock",
        examples=["Rock"],
        description="Género musical del concierto",
    )
    Social_media_followers: float | None = Field(
        default=1_000_000,
        examples=[1_000_000],
        description="Seguidores en redes sociales",
    )


@app.get("/")
def home():
    return {"Universidad EIA": "MLOps"}


@app.post("/predict")
def predict(datos: DatosConcierto):
    entrada = pd.DataFrame([datos.model_dump()])
    prediccion = model.predict(entrada)[0]

    return {"Sold_out": int(prediccion)}
