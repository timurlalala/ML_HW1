from fastapi import FastAPI
from fastapi import UploadFile
from pydantic import BaseModel
from typing import List
from HW1_model import preprocess_dataframe
import pandas as pd
import pickle

# Загружаем пайплайн
with open('cars_pipeline.pkl', 'rb') as file:
    pipe = pickle.load(file)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


@app.post("/predict_item", response_model=float)
def predict_item(item: Item) -> float:
    data = preprocess_dataframe(pd.DataFrame(item.model_dump(), index=[0]))
    return pipe.predict(data)[0]


@app.post("/predict_items")
def predict_items(items: UploadFile) -> List[float]:
    data = preprocess_dataframe(pd.read_csv(items.file))
    return pipe.predict(data)
