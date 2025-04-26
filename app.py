from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np

import butterfly as data

model = tf.keras.models.load_model("butterfly_v1.keras")

routes = [{"get": ["/", "/list_all",]}, {"post": ["/scan",]}]

app = FastAPI()

@app.get("/")
async def home():
    return {"routes": routes}

@app.get("/list_all")
async def list():
    return data.Butterfly.butterfly_data

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    image = Image.open(file.file).resize(size=[256, 256])
    image_array = np.array(image)
    final_image = np.expand_dims(image_array, axis=0)
    pred = model.predict(final_image)
    pred_argmax = np.argmax(pred)
    return {"butterfly": data.Butterfly.butterfly_data[pred_argmax],}