from fastapi import FastAPI,UploadFile,File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os

from meter_reader.recognize_torch import TorchRecognizer
from meter_reader.pipeline import infer_image

app=FastAPI(title="Meter Reader API")

MODEL_PATH=os.getenv("MODEL_PATH","models/model_3300.pth")
EXPECTED_DIGITS=int(os.getenv("EXPECTED_DIGITS","3"))
MIN_SCORE=float(os.getenv("MIN_SCORE","0.7"))

recognizer=None

@app.on_event("startup")
def startup_event():
    global recognizer
    recognizer=TorchRecognizer(MODEL_PATH)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/infer")
async def infer_api(file:UploadFile=File(...)):
    data=await file.read()
    np_arr=np.frombuffer(data,np.uint8)
    img=cv2.imdecode(np_arr,cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"ok":False,"reason":"decode_failed"}
        )
    res=infer_image(
        img,
        recognizer,
        expected_digits=EXPECTED_DIGITS,
        min_score=MIN_SCORE
    )
    return res

